#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import numpy as np
import control as ct
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
from tqdm import tqdm
from decimal import Decimal
import random, os
import pandas as pd
from scipy.io import wavfile


def generate_coefficients(start, end, steps):
    return np.arange(start, end, steps)

def check_system_validity(A, B, C):
    order = A.shape[0]
    poles = np.linalg.eigvals(A)
    return (np.linalg.matrix_rank(ct.ctrb(A, B)) == order and 
            np.linalg.matrix_rank(ct.obsv(A, C)) == order and 
            np.all(poles < 0))

def generate_systems(order, coefs, d_coefs):
    non_validity_count = 0
    for A_coefs in tqdm(product(coefs, repeat=order*order), desc=f'Order {order}'):
        A = np.array(A_coefs).reshape(order, order)
        for B_coefs in product(coefs, repeat=order):
            B = np.array(B_coefs).reshape(order, 1)
            for C_coefs in product(coefs, repeat=order):
                C = np.array(C_coefs).reshape(1, order)
                for D in d_coefs:
                    if check_system_validity(A, B, C):
                        yield (A, B, C, D)
                    else :
                        non_validity_count+=1
    print(f"System rejected (unstable & non-controllable+observables systems): {non_validity_count}")

def lqg(A, B, C, D, QRV):
    n_states = A.shape[0]
    Q = np.diag([max(QRV[0], 1e-6)] + [1] * (n_states - 1))
    R = np.array([[max(QRV[1], 1e-6)]])
    V = np.diag([max(QRV[2], 1e-6)] * n_states)
    try:
        K, _, _ = ct.lqr(A, B, Q, R)
        L, _, _ = ct.lqe(A, np.eye(n_states), C, V, np.array([[1]]))
        A_cl = np.block([[A - B @ K, B @ K], [np.zeros_like(A), A - L @ C]])
        return ct.ss(A_cl, np.vstack([B, np.zeros_like(B)]), np.hstack([C, np.zeros_like(C)]), D), K
    except np.linalg.LinAlgError:
        return None, None

def optimize_system(args):
    sys, T = args
    setpoint=0.5
    A, B, C, D = sys
    n_states = A.shape[0]
    def performance(QRV):
        cl_sys, _ = lqg(A, B, C, D, QRV)
        if cl_sys is None: return np.inf
        try:
            _, Y = ct.step_response(cl_sys, T=T[::50])
            return np.mean((Y - setpoint)**2)
        except: return np.inf
    initial_guess = [10, 1, 1] # QRV
    result = minimize(performance, initial_guess, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
    if result.fun == np.inf: return None
    optimal_QRV = result.x
    cl_sys_optimal, K_optimal = lqg(A, B, C, D, optimal_QRV)
    if cl_sys_optimal is None: return None
    noise = np.random.normal(size=T.shape)
    U_noise_optimal = 1 + gaussian_filter1d(noise, 2)/3.
    T_optimal, Y_optimal, X_optimal = ct.forced_response(cl_sys_optimal, T=T, U=U_noise_optimal, return_x=True)
    if not (0.5*setpoint < Y_optimal.mean() < 1.5*setpoint): return None
    U_optimal = -K_optimal @ (X_optimal[:n_states] - setpoint * np.ones((n_states, len(T))))
    return sys, optimal_QRV, T_optimal, U_optimal.T, Y_optimal

def process_systems(all_systems, T, sample=None):
    if not(sample is None) :
        all_systems = random.sample(all_systems, sample)
    with Pool() as p:
        args = [(sys, T) for sys in all_systems]
        results = list(tqdm(p.imap(optimize_system, args), total=len(args)))
    return results

def generate_dataset(results):
    os.makedirs('csv_database', exist_ok=True)
    os.makedirs('audio_database', exist_ok=True)
    valid_systems = 0
    for i, result in enumerate(results):
        if result is None: continue
        sys, QRV, T, U, Y = result
        # Normalize U & Y
        U_dec = np.array([[Decimal(str(x[0])) for x in U]])
        Umin, Umax, Ymin, Ymax = U_dec.min(), U_dec.max(), Y.min(), Y.max()
        U = np.array([float(2 * (x - Umin) / (Umax - Umin) - 1) for x in U_dec[0]])
        Y = 2 * ((Y - Ymin) / (Ymax - Ymin)) - 1
        # Convert system to transfer function
        A, B, C, D = sys
        tf_sys = ct.ss2tf(A, B, C, D)
        # Save system data
        pd.DataFrame({'numerator': [tf_sys.num[0][0]], 'denominator': [tf_sys.den[0][0]],
                      'u': [[Umin, Umax]], 'y': [[Ymin, Ymax]]}).to_csv(f'csv_database/system_{valid_systems}.csv', index=False)
        # Prepare and save audio signal (limited to 1 second)
        sample_rate = 8000
        max_duration = 1  # 1 second
        # Find the index corresponding to 1 second in the original time array
        one_second_index = np.searchsorted(T, 1.0)
        # Limit Y to 1 second
        Y_limited = Y[:one_second_index]
        T_limited = T[:one_second_index]
        # Normalize the limited Y & Resample to exactly 1 second duration
        y_normalized = Y_limited / np.max(np.abs(Y_limited))
        y_resampled = np.interp(np.linspace(0, 1, sample_rate), T_limited, y_normalized)
        # Save the audio file
        wavfile.write(f'audio_database/signal_{valid_systems}.wav', sample_rate, y_resampled.astype(np.float32))
        valid_systems += 1
    print(f"Total valid systems generated and saved: {valid_systems}")

if __name__ == '__main__':
    coefs1 = generate_coefficients(-1, 1.1, 5./10.)
    coefs2 = generate_coefficients(-1, 1.1, 2./3.)
    coefs3 = generate_coefficients(-0.5, +0.6, 1.)
    d_coefs = [0] #generate_coefficients(-0.1, 0.11, 0.1)
    
    systems1 = list(generate_systems(1, coefs1, d_coefs))
    print(f"Systems 1 generated: {len(systems1)}")
    systems2 = list(generate_systems(2, coefs2, d_coefs))
    print(f"Systems 2 generated: {len(systems2)}")
    systems3 = list(generate_systems(3, coefs3, d_coefs))
    print(f"Systems 3 generated: {len(systems3)}")
    
    all_systems = systems1 + systems2 + systems3
    print(f"Total systems generated: {len(all_systems)}")
    
    T = np.linspace(0, 100, 1000)
    results = process_systems(all_systems, T, sample=3)

    generate_dataset(results)

        