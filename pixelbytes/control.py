#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien
"""

import numpy as np
import control as ct
import pandas as pd
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

def generate_transfer_function():
    num_coefs = den_coefs = [round(x, 1) for x in np.arange(-1, 1.1, 0.1) if abs(x) > 1e-10]
    num = np.random.choice(num_coefs, size=np.random.randint(1, 4))
    den = np.random.choice(den_coefs, size=np.random.randint(2, 5))
    while len(den) <= len(num):
        den = np.append(den, np.random.choice(den_coefs))
    return ct.tf(num, den)

def solve_optimal_control(sys, t, setpoint=1.0, use_solve_ocp=False):
    ss_sys = ct.tf2ss(sys)
    n_states = ss_sys.A.shape[0]
    Q, R = np.eye(n_states), np.array([[1]])
    
    if use_solve_ocp:
        aug_sys = ct.augw(ss_sys, Q, R)
        x0 = np.zeros((aug_sys.nstates, 1))
        x0[-1] = setpoint
        
        ocp = ct.optimal.OptimalControlProblem(aug_sys, t, x0=x0)
        result = ct.optimal.solve_ocp(ocp)
        
        U = result.inputs
        Y = result.outputs[:n_states]
        X = result.states[:n_states]
        
        return t, U, Y, X, None
    else:
        K, _, _ = ct.lqr(ss_sys, Q, R)
        
        # Create closed-loop system manually
        A_cl = ss_sys.A - ss_sys.B @ K
        B_cl = ss_sys.B * setpoint
        C_cl = ss_sys.C
        D_cl = ss_sys.D
        cl_sys = ct.ss(A_cl, B_cl, C_cl, D_cl)
        
        # Simulate the closed-loop system
        T, Y = ct.step_response(cl_sys, T=t)
        
        # Calculate states
        X = np.zeros((n_states, len(T)))
        x = np.zeros(n_states)
        for i in range(len(T)):
            X[:, i] = x
            x += (A_cl @ x + B_cl.flatten()) * (T[1] - T[0])
        
        # Calculate control input
        U = setpoint - K @ X
        
        return T, U, Y, X, K

def plot_system_response(sys, t, U, y, x, setpoint, show_states=True, show_frequency=True, show_pz=True):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].plot(t, U.T), axs[0].set_title('Control Input (U)')
    axs[1].plot(t, y.T), axs[1].set_title('System Response (Y)')
    axs[1].axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    axs[1].legend()

    if show_states:
        for i in range(x.shape[0]):
            axs[2].plot(t, x[i, :], label=f'State {i+1}')
        axs[2].set_title('System States'), axs[2].legend()

    if show_frequency:
        w, mag, _ = ct.bode(sys, dB=True, Hz=True, plot=False)
        axs[3].semilogx(w, mag), axs[3].set_title('Frequency Response')
    elif show_pz:
        ct.pzmap(sys, plot=True, ax=axs[3])
        axs[3].set_title('Pole-Zero Map')

    for ax in axs:
        ax.set_xlabel('Time' if ax != axs[3] else 'Frequency [Hz]')
        ax.set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def generate_dataset(num_systems, use_solve_ocp=False, display_plots=False, show_states=True, show_frequency=True, show_pz=True):
    os.makedirs('csv_database', exist_ok=True)
    os.makedirs('audio_database', exist_ok=True)
    t = np.linspace(0, 10, 1000)

    for i in range(num_systems):
        sys = generate_transfer_function()
        setpoint = np.random.uniform(0.5, 1.5)
        t, U, y, x, K = solve_optimal_control(sys, t, setpoint, use_solve_ocp)
        
        pd.DataFrame({'numerator': [ct.tfdata(sys)[0][0]], 'denominator': [ct.tfdata(sys)[1][0]]}).to_csv(f'csv_database/system_{i}.csv', index=False)
        
        y_normalized = y / np.max(np.abs(y))
        y_resampled = np.interp(np.linspace(0, t[-1], int(t[-1] * 44100)), t, y_normalized.flatten())
        wavfile.write(f'audio_database/signal_{i}.wav', 44100, y_resampled.astype(np.float32))
        
        print(f"System {i} generated and saved")

        if display_plots:
            plot_system_response(sys, t, U, y, x, setpoint, show_states, show_frequency, show_pz)

# Usage example:
generate_dataset(5, use_solve_ocp=False, display_plots=True, show_states=True, show_frequency=True, show_pz=False)
# generate_dataset(100, use_solve_ocp=False, display_plots=False)