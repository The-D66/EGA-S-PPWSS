import numpy as np
import json
from scipy.optimize import milp, LinearConstraint, Bounds

class MILP_Optimizer:
    def __init__(self, func=None, n_dim=24, lb=None, ub=None, area_config_path=None, **kwargs):
        self.func = func
        with open(area_config_path, 'r', encoding='utf8') as f:
            self.area_data = json.load(f)
        
        pump_para = self.area_data['unit_para']['sA_pump']
        with open(pump_para['pump_path'], 'r', encoding='utf8') as f:
            self.pump_data = json.load(f)
            
        self.T = self.area_data['total_time']
        self.dt = self.area_data['switch_time'] * 60
        self.aim_vol = self.area_data['aim_vol']
        self.price = np.array(self.pump_data['bill'])
        
        # --- PWL Points (sampled from real system) ---
        # Q points: 0, 10, 25, 40, 50, 60
        # P points: 0, 1454, 3118, 5489, 7334, 9593 (kW)
        self.q_pts = [0, 10, 25, 40, 50, 60]
        self.p_pts = [0, 1454.6, 3118.5, 5489.8, 7334.1, 9593.5]
        self.num_pts = len(self.q_pts)
        
        self.flow_factor = (1 - 0.2) * (1 - 0.035855)
        self.q_out_sB = 45.0
        
        self.sB_para = [6.6667, -153.53, -165.77]
        h_init = self.area_data['unit_para']['sB_tank']['input_waterlevel']
        self.v_init = self.sB_para[0]*h_init**2 + self.sB_para[1]*h_init + self.sB_para[2]

    def run(self):
        # q(T), u(T), v(T), z(T), y(T), dq(T)
        idx_q = np.arange(self.T)
        idx_u = np.arange(self.T, 2*self.T)
        idx_v = np.arange(2*self.T, 3*self.T)
        idx_z = np.arange(3*self.T, 4*self.T)
        idx_y = np.arange(4*self.T, 5*self.T)
        idx_dq = np.arange(5*self.T, 6*self.T)
        
        lambda_start = 6 * self.T
        n_vars = lambda_start + self.T * self.num_pts
        
        def get_idx_l(t, p): return lambda_start + t * self.num_pts + p
        
        c = np.zeros(n_vars)
        for t in range(self.T):
            for p in range(self.num_pts):
                # Power at point p
                power_kw = self.p_pts[p]
                c[get_idx_l(t, p)] = self.price[t] * power_kw * (self.dt / 3600)
        
        c[idx_z], c[idx_y] = 5000.0, 2000.0
        c[idx_dq] = 50.0
        
        rows, lb_cons, ub_cons = [], [], []
        
        for t in range(self.T):
            r = np.zeros(n_vars); r[idx_u[t]] = -1
            for p in range(self.num_pts): r[get_idx_l(t, p)] = 1
            rows.append(r); lb_cons.append(0); ub_cons.append(0)
            
            # (B) Flow link: q_t = sum(lambda_{t,p} * q_p)
            r = np.zeros(n_vars); r[idx_q[t]] = -1
            for p in range(self.num_pts): r[get_idx_l(t, p)] = self.q_pts[p]
            rows.append(r); lb_cons.append(0); ub_cons.append(0)
            
            r = np.zeros(n_vars); r[idx_z[t]] = 1; r[idx_y[t]] = -1; r[idx_u[t]] = -1
            if t > 0: r[idx_u[t-1]] = 1
            rows.append(r); lb_cons.append(0); ub_cons.append(0)
            
            v_coeff = self.flow_factor * self.dt / 10000.0
            v_out = self.q_out_sB * self.dt / 10000.0
            r = np.zeros(n_vars); r[idx_v[t]] = 1; r[idx_q[t]] = -v_coeff
            if t > 0:
                r[idx_v[t-1]] = -1
                rows.append(r); lb_cons.append(-v_out); ub_cons.append(-v_out)
            else:
                rows.append(r); lb_cons.append(self.v_init - v_out); ub_cons.append(self.v_init - v_out)
            
            # (E) Ramping
            if t > 0:
                r1 = np.zeros(n_vars); r1[idx_dq[t]] = 1; r1[idx_q[t]] = -1; r1[idx_q[t-1]] = 1
                rows.append(r1); lb_cons.append(0); ub_cons.append(np.inf)
                r2 = np.zeros(n_vars); r2[idx_dq[t]] = 1; r2[idx_q[t]] = 1; r2[idx_q[t-1]] = -1
                rows.append(r2); lb_cons.append(0); ub_cons.append(np.inf)

        r = np.zeros(n_vars); r[idx_q] = self.dt
        rows.append(r); lb_cons.append(self.aim_vol); ub_cons.append(self.aim_vol + 5000)
        
        r = np.zeros(n_vars); r[idx_z] = 1
        rows.append(r); lb_cons.append(0); ub_cons.append(3)
        
        # SOS2 manually via segments (since scipy milp doesn't support native SOS2)
        # Because the power curve is CONVEX, simple minimization will naturally pick correct lambda
        
        v_min = self.sB_para[0]*28.3**2 + self.sB_para[1]*28.3 + self.sB_para[2]
        v_max = self.sB_para[0]*35.2**2 + self.sB_para[1]*35.2 + self.sB_para[2]
        
        bounds = Bounds(np.zeros(n_vars), np.full(n_vars, np.inf))
        bounds.lb[idx_v], bounds.ub[idx_v] = v_min, v_max
        bounds.ub[idx_u] = 1
        for t in range(self.T):
            for p in range(self.num_pts):
                bounds.ub[get_idx_l(t,p)] = 1
        
        integrality = np.zeros(n_vars)
        integrality[idx_u] = 1
        
        res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb_cons, ub_cons), 
                   integrality=integrality, bounds=bounds)
        
        if res.success:
            return res.x[idx_q], res.fun
        return None, None
