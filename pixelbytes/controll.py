#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfr
"""

import numpy as np
import control as ct
import pandas as pd
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

def generate_coefficients(start, end, steps):
    return np.arange(start, end, steps)

def check_system_validity(A, B, C):
    order = A.shape[0]
    return (np.linalg.matrix_rank(ct.ctrb(A, B)) == order and 
            np.linalg.matrix_rank(ct.obsv(A, C)) == order)

def generate_systems(order, coefs, d_coefs):
    for A_coefs in tqdm(product(coefs, repeat=order*order), desc=f'Order {order}'):
        A = np.array(A_coefs).reshape(order, order)
        for B_coefs in product(coefs, repeat=order):
            B = np.array(B_coefs).reshape(order, 1)
            for C_coefs in product(coefs, repeat=order):
                C = np.array(C_coefs).reshape(1, order)
                for D in d_coefs:
                    if check_system_validity(A, B, C):
                        yield (A, B, C, D)

def solve_optimal_control(sys, t, setpoint=0.5):
    ss_sys = ct.ss(*sys)
    n_states = ss_sys.A.shape[0]
    Q, R = np.eye(n_states), np.array([[1]])
    K, _, _ = ct.lqr(ss_sys, Q, R)
    cl_sys = ct.ss(ss_sys.A - ss_sys.B @ K, ss_sys.B * setpoint, ss_sys.C, ss_sys.D)
    T, Y = ct.step_response(cl_sys, T=t)
    X = ct.initial_response(cl_sys, T=t, X0=np.zeros(n_states))[1]
    U = np.array([setpoint - K @ X[:, i] for i in range(X.shape[1])])
    return T, U.flatten(), Y.flatten(), X, K.flatten()

def plot_system_response(sys, t, U, y, x, setpoint, show_states=True, show_frequency=True, show_pz=True):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    axs[0].plot(t, U.flatten())
    axs[0].set_title('Control Input (U)')
    
    axs[1].plot(t, y.flatten())
    axs[1].set_title('System Response (Y)')
    axs[1].axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    axs[1].legend()
    
    if show_states:
        for i in range(x.shape[0]):
            axs[2].plot(t, x[i, :], label=f'State {i+1}')
        axs[2].set_title('System States')
        axs[2].legend()
    
    if show_frequency:
        w, mag, _ = ct.bode(ct.ss(*sys), dB=True, Hz=True, plot=False)
        axs[3].semilogx(w, mag)
        axs[3].set_title('Frequency Response')
    elif show_pz:
        ct.pzmap(ct.ss(*sys), plot=True, ax=axs[3])
        axs[3].set_title('Pole-Zero Map')
    
    for ax in axs:
        ax.set_xlabel('Time' if ax != axs[3] else 'Frequency [Hz]')
        ax.set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()

def generate_dataset(systems, display_plots=False):
    os.makedirs('csv_database', exist_ok=True)
    os.makedirs('audio_database', exist_ok=True)
    t = np.linspace(0, 10, 1000)
    for i, sys in enumerate(systems):
        setpoint = np.random.uniform(0.5, 1.5)
        t, U, y, x, K = solve_optimal_control(sys, t, setpoint)
        tf_sys = ct.ss2tf(*sys)
        pd.DataFrame({'numerator': [tf_sys.num[0][0]], 'denominator': [tf_sys.den[0][0]]}).to_csv(f'csv_database/system_{i}.csv', index=False)
        y_normalized = y / np.max(np.abs(y))
        y_resampled = np.interp(np.linspace(0, t[-1], int(t[-1] * 44100)), t, y_normalized.flatten())
        wavfile.write(f'audio_database/signal_{i}.wav', 44100, y_resampled.astype(np.float32))
        print(f"System {i} generated and saved")
        if display_plots:
            plot_system_response(sys, t, U, y, x, setpoint)

if __name__ == '__main__':
    coefs1 = generate_coefficients(-1, 1.1, 2./10.)
    coefs2 = generate_coefficients(-1, 1.1, 2./3.)
    coefs3 = generate_coefficients(0, 1.1, 1.0)
    d_coefs = generate_coefficients(-1, 1.1, 1.0)
    
    systems1 = list(generate_systems(1, coefs1, d_coefs))
    systems2 = []#list(generate_systems(2, coefs2, d_coefs))
    systems3 = list(generate_systems(3, coefs3, d_coefs))
    
    all_systems = systems1 + systems2 + systems3
    print(f"Total systems generated: {len(all_systems)}")
    
    generate_dataset(all_systems[:5], display_plots=True)  # Generate and plot first 5 systems
    # generate_dataset(all_systems, display_plots=False)  # Generate all systems without plotting