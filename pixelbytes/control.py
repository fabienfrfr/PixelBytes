#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import torch, os, warnings
import torch.nn as nn
import warnings
import pandas as pd
import soundfile as sf
from datasets import Dataset, Features, Value, Audio
from huggingface_hub import login

def generate_coefficients(start, end, steps):
    return torch.arange(start, end, steps)

def generate_systems(order, coefs, d_coefs):
    non_validity_count = 0
    for A_coefs in tqdm(product(coefs, repeat=order*order), desc=f'Order {order}'):
        A = torch.tensor(A_coefs, dtype=torch.float).reshape(order, order)
        for B_coefs in product(coefs, repeat=order):
            B = torch.tensor(B_coefs, dtype=torch.float).reshape(order, 1)
            for C_coefs in product(coefs, repeat=order):
                C = torch.tensor(C_coefs, dtype=torch.float).reshape(1, order)
                for D in d_coefs:
                    yield (A, B, C, torch.tensor([[D]], dtype=torch.float))

class ControlSystem(nn.Module):
    def __init__(self, n_states, n_inputs, n_outputs):
        super().__init__()
        self.n_states, self.n_inputs, self.n_outputs = n_states, n_inputs, n_outputs
        self.params = nn.Parameter(torch.tensor([1., 1., 1.], dtype=torch.float32))
        self.debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
        self.methods = {'lqg': self.lqg, 'mpc': self.mpc, 'smc': self.smc} # ADDING TRAINED NN
        self.solvers = {'euler_adaptive': self._euler_adaptive, 'crank_nicolson': self._crank_nicolson}

    def debug_warn(self, message): 
        if self.debug_mode: warnings.warn(message)
    
    def check_system_validity(self, A, B, C):
        order = A.shape[0]
        poles = torch.linalg.eigvals(A).real
        ctrb = torch.cat([B] + [torch.matrix_power(A, i) @ B for i in range(1, order)], dim=1)
        obsv = torch.cat([C] + [C @ torch.matrix_power(A, i) for i in range(1, order)], dim=0)
        return (torch.linalg.matrix_rank(ctrb) == order and 
                torch.linalg.matrix_rank(obsv) == order and 
                torch.all(poles < 0))
    
    def forward(self, A, B, C, D, t, u, x0=None, method='lqg', solver='crank_nicolson'):
        self.debug_warn(f"Forward input shapes: A:{A.shape}, B:{B.shape}, C:{C.shape}, D:{D.shape}, t:{t.shape}, u:{u.shape}")
        if u.dim() == 1: u = u.unsqueeze(-1)
        if not self.check_system_validity(A,B,C): raise ValueError(f"Unsupported LTI: {(A,B,C)} unstable")
        if method not in self.methods: raise ValueError(f"Unsupported control method: {method}")
        if solver not in self.solvers: raise ValueError(f"Unsupported solver: {solver}")
        if x0 is None: x0 = torch.zeros(self.n_states, device=A.device)
        cl_sys, controller = self.methods[method](A, B, C, D, t, u)
        y, x = self.system_response(cl_sys, t, u, x0, solver)
        self.debug_warn(f"Forward output shapes: y:{y.shape}, x:{x.shape}")
        return t, y, x, controller

    def optimize(self, A, B, C, D, setpoint, method='lqg', solver='crank_nicolson', num_iterations=10, lr=0.01):
        optimizer = torch.optim.Adam([self.params], lr=lr)
        t = torch.linspace(0, 10, 100)
        u = torch.ones(100, 1) * setpoint
        x0 = torch.zeros(self.n_states)
        for _ in range(num_iterations):
            optimizer.zero_grad()
            _, y, _, _ = self(A, B, C, D, t, u, x0, method=method, solver=solver)
            loss = torch.mean((y - setpoint)**2) + 0.01 * torch.mean(u**2)
            loss.backward()
            optimizer.step()
        self.debug_warn(f"Optimize: final params: {self.params}")
        return self.params

    def lqg(self, A, B, C, D, t=None, u=None): # in dev
        Q = torch.diag(torch.cat([self.params[0].unsqueeze(0), torch.ones(self.n_states-1)]))
        R = self.params[1].unsqueeze(0).unsqueeze(0)
        V = torch.eye(self.n_states) * self.params[2]
        P_lqr = torch.linalg.solve(A.T @ Q @ A - Q, -A.T @ Q @ B @ torch.inverse(R + B.T @ Q @ B) @ B.T @ Q @ A - A.T - A)
        P_lqe = torch.linalg.solve(A @ V @ A.T - V, -A @ V @ C.T @ torch.inverse(torch.eye(self.n_outputs) + C @ V @ C.T) @ C @ V @ A.T - A.T - A)
        K, L = torch.inverse(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A, P_lqe @ C.T @ torch.inverse(torch.eye(self.n_outputs) + C @ P_lqe @ C.T)
        A_cl = A - B @ K
        return (A_cl, B, C, D), (K, L)

    def mpc(self, A, B, C, D, t, u, horizon=10): # not working for now
        x = torch.zeros((self.n_states, horizon+1), requires_grad=True)
        u_mpc = torch.zeros((self.n_inputs, horizon), requires_grad=True)
        mpc_loss = lambda: torch.sum(torch.diagonal(x.T @ x) + torch.diagonal(u_mpc.T @ u_mpc)) + 1000 * torch.sum((x[:, 1:] - (A @ x[:, :-1] + B @ u_mpc))**2)
        torch.optim.LBFGS([x, u_mpc]).step(mpc_loss)
        return (A, B, C, D), u_mpc.detach()

    def smc(self, A, B, C, D, t=None, u=None, order=1): # not working for now
        S = torch.cat([torch.ones(order), torch.tensor([1.0])])
        control_law = lambda x, x_d: -torch.sign(S @ (x - x_d)) * self.params[0]
        return (A, B, C, D), (control_law, S)

    def _euler_adaptive(self, A, B, C, D, u, x0, t):
        x_list = [x0]
        y_list = [C @ x0 + D @ u[0]]
        current_t = t[0].item()
        i = 1
        tol = 1e-3
        min_dt = 1e-3
        max_dt = (t[-1] - t[0]).item()
        while current_t < t[-1].item() and i < len(t):
            dt = min(t[i].item() - current_t, max_dt)
            while dt > min_dt:
                x_half = x_list[-1] + (dt/2) * (A @ x_list[-1] + B @ u[i-1])
                x_full = x_list[-1] + dt * (A @ x_list[-1] + B @ u[i-1])
                x_double = x_list[-1] + (dt/2) * (A @ x_list[-1] + B @ u[i-1]) + (dt/2) * (A @ x_half + B @ u[i-1])
                if torch.norm(x_double - x_full) < tol:
                    x_list.append(x_double)
                    y_list.append(C @ x_double + D @ u[i])
                    current_t += dt
                    break
                dt /= 2
            i += 1 if current_t >= t[i].item() else 0
        return torch.stack(y_list).squeeze(-1), torch.stack(x_list)

    def _crank_nicolson(self, A, B, C, D, u, x0, t):
        x_list = [x0]
        y_list = [(C @ x0 + D @ u[0]).squeeze()]
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            f1 = A @ x_list[-1] + B @ u[i-1]
            x_new = x_list[-1] + dt * 0.5 * (f1 + A @ (x_list[-1] + dt * f1) + B @ u[i])
            x_list.append(x_new)
            y_list.append((C @ x_new + D @ u[i]).squeeze())
        return torch.stack(y_list), torch.stack(x_list)

    def system_response(self, cl_sys, t, u, x0=None, solver='crank_nicolson'):
        A, B, C, D = cl_sys
        if x0 is None: x0 = torch.zeros(A.shape[0], device=A.device)
        # in dev (separate in 2 block to get output controler) --> add NN control and yield for dynamic control (temporary : use Gym-setpoint library for test)
        y, x = self.solvers[solver](A, B, C, D, u, x0, t) $
        self.debug_warn(f"System response: y shape: {y.shape}, x shape: {x.shape}")
        return y, x

class ControlDataset:
    def __init__(self, dataset_dir=os.getcwd()):
        self.dataset_dir = dataset_dir
        self.csv_dir = os.path.join(dataset_dir, 'csv_database')
        self.audio_dir = os.path.join(dataset_dir, 'audio_database')
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        self.data = []

    def generate_dataset(self, results):
        self.data = [self._process_result(result) for result in results if result is not None]
        self._save_data()
        print(f"Total valid systems generated and saved: {len(self.data)}")

    def _process_result(self, result):
        A, B, C, D, method, T, U, Y, controller_params = result
        n = A.shape[0]

        U, Y = self._normalize_signals(U.flatten(), Y.flatten())
        num, den = self._calculate_transfer_function(A, B, C, n)

        return {
            'numerator': num.tolist(),
            'denominator': den.tolist(),
            'method': method,
            'controller_params': [c.detach().tolist() for c in controller_params],
            'u': [float(U.min()), float(U.max())],
            'y': [float(Y.min()), float(Y.max())],
            'audio': torch.stack((U, Y), dim=1).detach().numpy()
        }

    @staticmethod
    def _normalize_signals(U, Y):
        U = 2 * ((U - U.min()) / (U.max() - U.min())) - 1
        Y = 2 * ((Y - Y.min()) / (Y.max() - Y.min())) - 1
        return U, Y

    @staticmethod
    def _calculate_transfer_function(A, B, C, n):
        num = torch.tensor([C @ torch.matrix_power(A, i) @ B for i in range(n)]).flatten()
        den = torch.zeros(n + 1, dtype=A.dtype, device=A.device)
        den[-1] = 1
        den[:-1] = -torch.tensor([torch.trace(torch.matrix_power(A, k)) / k for k in range(1, n+1)])
        return num, den

    def _save_data(self):
        for i, item in enumerate(self.data):
            pd.DataFrame({k: [v] for k, v in item.items() if k != 'audio'}).to_csv(
                os.path.join(self.csv_dir, f'{i}.csv'), index=False)
            sf.write(os.path.join(self.audio_dir, f'{i}.ogg'), 
                     item['audio'], 16000, format='ogg', subtype='vorbis')

    def create_hf_dataset(self):
        audio_files = os.listdir(self.audio_dir)
        idx = [os.path.splitext(a)[0] for a in audio_files]
        data = {
            "audio": [os.path.join(self.audio_dir, f"{i}.ogg") for i in idx],
            "text": [open(os.path.join(self.csv_dir, f"{i}.csv"), 'r').read().strip() for i in idx]}
        dataset = Dataset.from_dict(data)
        features = Features({
            "audio": Audio(sampling_rate=16000, mono=False),
            "text": Value("string")})
        return dataset.cast(features)

def push_control_to_hub(dataset, repo_name='PixelBytes-OptimalControl'):
    token = input("Please enter your Hugging Face token: ")
    login(token=token)
    # Push to Hub
    dataset.push_to_hub(repo_name)


if __name__ == "__main__":
    ## TEST
    os.environ['DEBUG'] = 'True'
    A = torch.tensor([[0, 1], [-1, -1]], dtype=torch.float32)
    B = torch.tensor([[0], [1]], dtype=torch.float32)
    C = torch.tensor([[1, 0]], dtype=torch.float32)
    D = torch.tensor([[0]], dtype=torch.float32)
    t = torch.linspace(0, 10, 200)
    u = torch.ones((1000, 1)) + 0.3 * torch.randn((1000, 1))
    
    method = 'lqg'
    system = ControlSystem(n_states=2, n_inputs=1, n_outputs=1)

    setpoint = 0.5
    system.optimize(A, B, C, D, setpoint, method=method, solver='euler_adaptive')
    t, y, x, controller = system(A, B, C, D, t, u, method='lqg', solver='crank_nicolson')
    
    import pylab as plt
    plt.plot(t.detach(), x[:,1].detach())
    plt.plot(t.detach(), y.detach())
    plt.show()
    
    
    results = [(A, B, C, D, method, t, x[:,1], y, controller),]
    dataset_generator = ControlDataset()
    dataset_generator.generate_dataset(results)
    hf_dataset = dataset_generator.create_hf_dataset()
    print("Hugging Face dataset created successfully.")

    ## DATASET
    '''
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
    '''