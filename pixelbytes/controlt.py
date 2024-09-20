#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import torch
import os
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from decimal import Decimal
import pandas as pd
import soundfile as sf
from datasets import Dataset, Features, Value, Audio
from huggingface_hub import login
import random

def generate_coefficients(start, end, steps):
    return torch.arange(start, end, steps)

def check_system_validity(A, B, C):
    order = A.shape[0]
    poles = torch.linalg.eigvals(A).real
    ctrb = torch.cat([B] + [torch.matrix_power(A, i) @ B for i in range(1, order)], dim=1)
    obsv = torch.cat([C] + [C @ torch.matrix_power(A, i) for i in range(1, order)], dim=0)
    return (torch.linalg.matrix_rank(ctrb) == order and 
            torch.linalg.matrix_rank(obsv) == order and 
            torch.all(poles < 0))

def generate_systems(order, coefs, d_coefs):
    non_validity_count = 0
    for A_coefs in tqdm(product(coefs, repeat=order*order), desc=f'Order {order}'):
        A = torch.tensor(A_coefs, dtype=torch.float).reshape(order, order)
        for B_coefs in product(coefs, repeat=order):
            B = torch.tensor(B_coefs, dtype=torch.float).reshape(order, 1)
            for C_coefs in product(coefs, repeat=order):
                C = torch.tensor(C_coefs, dtype=torch.float).reshape(1, order)
                for D in d_coefs:
                    if check_system_validity(A, B, C):
                        yield (A, B, C, torch.tensor([[D]], dtype=torch.float))
                    else:
                        non_validity_count += 1
    print(f"System rejected (unstable & non-controllable+observables systems): {non_validity_count}")

def lqr(A, B, Q, R):
    """Linear Quadratic Regulator design"""
    n = A.shape[0]
    P = torch.eye(n)
    for _ in range(100):  # Fixed-point iteration
        P_new = A.T @ P @ A - A.T @ P @ B @ torch.inverse(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if torch.allclose(P, P_new):
            break
        P = P_new
    K = torch.inverse(R + B.T @ P @ B) @ B.T @ P @ A
    return K, P

def lqe(A, G, C, V, W):
    """Linear Quadratic Estimator design"""
    n = A.shape[0]
    P = torch.eye(n)
    for _ in range(100):  # Fixed-point iteration
        P_new = A @ P @ A.T - A @ P @ C.T @ torch.inverse(W + C @ P @ C.T) @ C @ P @ A.T + G @ V @ G.T
        if torch.allclose(P, P_new):
            break
        P = P_new
    L = P @ C.T @ torch.inverse(W + C @ P @ C.T)
    return L, P

def lqg(A, B, C, D, QRV):
    n_states = A.shape[0]
    Q = torch.diag(torch.tensor([max(QRV[0], 1e-6)] + [1.] * (n_states - 1)))
    R = torch.tensor([[max(QRV[1], 1e-6)]])
    V = torch.diag(torch.tensor([max(QRV[2], 1e-6)] * n_states))
    try:
        K, _ = lqr(A, B, Q, R)
        L, _ = lqe(A, torch.eye(n_states), C, V, torch.tensor([[1.]]))
        A_cl = torch.block_diag(A - B @ K, A - L @ C)
        B_cl = torch.vstack([B, torch.zeros_like(B)])
        C_cl = torch.hstack([C, torch.zeros_like(C)])
        return (A_cl, B_cl, C_cl, D), K
    except:
        return None, None

def forced_response(A, B, C, D, T, U):
    n = len(T)
    x = torch.zeros((A.shape[0], n))
    y = torch.zeros((C.shape[0], n))
    I = torch.eye(A.shape[0])
    for i in range(1, n):
        dt = T[i] - T[i-1]
        # Semi-implicit Euler method
        x[:, i] = torch.linalg.solve(I - dt * A, x[:, i-1] + dt * B @ U[:, i-1])
        y[:, i] = C @ x[:, i] + D @ U[:, i]
    return T, y, x

def gaussian_filter1d(input, sigma):
    kernel_size = int(4 * sigma + 1)
    kernel = torch.exp(-torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1)**2 / (2*sigma**2))
    kernel = kernel / kernel.sum()
    padding = kernel_size//2
    padded_input = torch.nn.functional.pad(input, (padding, padding), mode='reflect')
    return torch.nn.functional.conv1d(padded_input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze()

def optimize_system(args):
    sys, T = args
    setpoint = 0.5
    A, B, C, D = sys
    n_states = A.shape[0]
    
    U_step = torch.ones((B.shape[1], len(T)), requires_grad=True)
    def performance(QRV):
        cl_sys, _ = lqg(A, B, C, D, QRV)
        if cl_sys is None: return torch.tensor(float('inf'))
        try:
            _, Y, _ = forced_response(*cl_sys, T[::50], U_step[::50])
            return torch.mean((Y - setpoint)**2)
        except:
            return torch.tensor(float('inf'))
    
    QRV = torch.tensor([10., 1., 1.], requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([QRV], lr=0.01)
    with torch.set_grad_enabled(True):
        for _ in range(10):
            optimizer.zero_grad()
            loss = performance(QRV)
            loss.backward()
            optimizer.step()
    optimal_QRV = QRV.detach()
    cl_sys_optimal, K_optimal = lqg(A, B, C, D, optimal_QRV)
    if cl_sys_optimal is None: return None
    noise = torch.randn(T.shape)[None]
    U_noise_optimal = 1 + noise/3. #gaussian_filter1d(noise, 2)/3.
    T_optimal, Y_optimal, X_optimal = forced_response(*cl_sys_optimal, T, U_noise_optimal)
    if not (0.5*setpoint < Y_optimal.mean() < 1.5*setpoint): return None
    U_optimal = -K_optimal @ (X_optimal[:n_states] - setpoint * torch.ones((n_states, len(T))))
    return sys, optimal_QRV, T_optimal, U_optimal.T, Y_optimal

def process_systems(all_systems, T, sample=None):
    if sample is not None:
        all_systems = random.sample(all_systems, sample)
    results = []
    for sys in tqdm(all_systems, total=len(all_systems)):
        result = optimize_system((sys, T))
        results.append(result)
    """ # OLD
    with Pool() as p:
        args = [(sys, T) for sys in all_systems]
        results = list(tqdm(p.imap(optimize_system, args), total=len(args)))
    """
    return results

def generate_dataset(results):
    os.makedirs('csv_database', exist_ok=True)
    os.makedirs('audio_database', exist_ok=True)
    valid_systems = 0
    for result in results:
        if result is None: continue
        sys, QRV, T, U, Y = result
        A, B, C, D = sys
        n = A.shape[0]
        # Ensure U and Y are 1D tensors
        U, Y = U.flatten(), Y.flatten()
        # Normalize U and Y
        U = 2 * ((U - U.min()) / (U.max() - U.min())) - 1
        Y = 2 * ((Y - Y.min()) / (Y.max() - Y.min())) - 1
        # Calculate transfer function coefficients
        num = torch.tensor([C @ torch.linalg.matrix_power(A, i) @ B for i in range(n)]).flatten()
        den = torch.zeros(n + 1, dtype=A.dtype, device=A.device)
        den[-1] = 1
        for k in range(1, n + 1):
            den[n-k] = -torch.trace(torch.linalg.matrix_power(A, k)) / k
        # Save CSV
        pd.DataFrame({
            'numerator': [num.tolist()],
            'denominator': [den.tolist()],
            "QRV (LQG)": [QRV.tolist()],
            'u': [[float(U.min()), float(U.max())]],
            'y': [[float(Y.min()), float(Y.max())]]
        }).to_csv(f'csv_database/{valid_systems}.csv', index=False)
        # Save audio
        sf.write(f'audio_database/{valid_systems}.ogg', torch.stack((U, Y), dim=1).numpy(), 16000, format='ogg', subtype='vorbis')
        valid_systems += 1
    print(f"Total valid systems generated and saved: {valid_systems}")
    
def create_audio_dataset(dataset_dir=os.getcwd()):
    # Get file lists
    audio = os.listdir(os.path.join(dataset_dir,"audio_database"))
    idx = [os.path.splitext(a)[0] for a in audio]
    # Create data dictionary
    data = {
        "audio": [os.path.join(dataset_dir, "audio_database", f"{i}.ogg") for i in idx],
        "text": [open(os.path.join(dataset_dir, "csv_database", f"{i}.csv"), 'r').read().strip() for i in idx]
    } # Create and cast dataset
    dataset = Dataset.from_dict(data)
    features = Features({"audio": Audio(sampling_rate=16000,mono=False), "text": Value("string")})
    return dataset.cast(features)

def push_control_to_hub(dataset, repo_name='PixelBytes-Control'):
    token = input("Please enter your Hugging Face token: ")
    login(token=token)
    # Push to Hub
    dataset.push_to_hub(repo_name)

if __name__ == "__main__":
    ## DATASET
    coefs1 = generate_coefficients(-1, 1.1, 5./10.)
    coefs2 = generate_coefficients(-1, 1.1, 2./3.)
    coefs3 = generate_coefficients(-0.5, +0.6, 1.)
    d_coefs = [0.] #generate_coefficients(-0.1, 0.11, 0.1)
    
    systems1 = []#list(generate_systems(1, coefs1, d_coefs))
    print(f"Systems 1 generated: {len(systems1)}")
    systems2 = []#list(generate_systems(2, coefs2, d_coefs))
    print(f"Systems 2 generated: {len(systems2)}")
    systems3 = list(generate_systems(3, coefs3, d_coefs))
    print(f"Systems 3 generated: {len(systems3)}")
    
    all_systems = systems1 + systems2 + systems3
    print(f"Total systems generated: {len(all_systems)}")
    
    T = torch.linspace(0, 10, 1000)
    results = process_systems(all_systems, T, sample=100)
    #generate_dataset(results)