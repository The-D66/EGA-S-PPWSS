import numpy as np
import json
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

def linearize_tank(tank_path, h_range):
    with open(tank_path, 'r') as f:
        data = json.load(f)
    para = data['para']
    
    def quad_v(h):
        return para[0]*h**2 + para[1]*h + para[2]
    
    h_samples = np.linspace(h_range[0], h_range[1], 50)
    v_samples = quad_v(h_samples)
    
    popt, _ = curve_fit(linear_func, h_samples, v_samples)
    k, b = popt
    
    popt_inv, _ = curve_fit(linear_func, v_samples, h_samples)
    m, n = popt_inv
    
    return {'k': k, 'b': b, 'm': m, 'n': n}

def linearize_pipe(pipe_path, q_range):
    with open(pipe_path, 'r') as f:
        data = json.load(f)
    para = data['para']
    
    def quad_hl(q):
        return para[0]*q**2 + para[1]*q + para[2]
    
    q_samples = np.linspace(q_range[0], q_range[1], 50)
    hl_samples = quad_hl(q_samples)
    
    popt, _ = curve_fit(linear_func, q_samples, hl_samples)
    return {'k': popt[0], 'b': popt[1]}

if __name__ == "__main__":
    # Test for Luotian Tank
    sB_lin = linearize_tank('data/tank_curve/sB.json', [28.24, 35.26])
    print(f"Luotian Tank Linearization: V = {sB_lin['k']:.4f}*H + {sB_lin['b']:.4f}")
    
    # Test for SX-LT Pipe
    pipe_lin = linearize_pipe('data/pipe_curve/sE-sB.json', [10, 60])
    print(f"SX-LT Pipe Linearization: HL = {pipe_lin['k']:.4f}*Q + {pipe_lin['b']:.4f}")
