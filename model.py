import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from catboost import CatBoostRegressor

class HybridPowerModel:
    def __init__(self, model_path="catboost_model.cbm"):
        self.params = {
            'P_base':     0.0500,
            'a_cpu':      6.3312,
            'a_scr_base': 0.1500,
            'a_scr_br':   1.6375,
            'a_net':      0.00001,
            'gamma':      2.2
        }
        
        try:
            self.cb_model = CatBoostRegressor()
            self.cb_model.load_model(model_path)
            self.use_ai = True
        except Exception as e:
            print(f"CatBoost model not found: {e}. Using Physics-Only mode.")
            self.use_ai = False

    def predict_power(self, u, voltage_now):
        p = self.params
        
        br_phys = (u['br'] / 255.0) ** p['gamma'] if u['br'] > 0 else 0
        
        term_base = p['P_base']
        term_cpu  = p['a_cpu'] * u['cpu']
        term_scr  = u['s_on'] * (p['a_scr_base'] + p['a_scr_br'] * br_phys)
        term_net  = p['a_net'] * u['net_kbps']
        
        P_mech = term_base + term_cpu + term_scr + term_net
        
        P_res = 0.0
        if self.use_ai:
            features = [u['cpu'], u['s_on'], u['br'], voltage_now, u['net_kbps']]
            P_res = self.cb_model.predict(features)
            if P_mech + P_res > 20.0: 
                print(f"[Warning] High Power! P_mech={P_mech:.2f}, P_res={P_res:.2f}, Input={features}")
        
        P_total = max(P_mech + P_res, p['P_base'])
        return P_total

class BatteryECM:
    def __init__(self, capacity_Ah=4.575, soh=1.0):
        self.capacity_new = capacity_Ah 
        self.R_ref_new = 0.04
        self.soh = soh
        self.Q_rated = (self.capacity_new * soh) * 3600.0 
        self.beta = 2.0
        self.R_ref = self.R_ref_new * (1.0 + self.beta * (1.0 - soh))
        self.k = [3.655205, 0.374104, 0.210354, -0.033974]
        
        self.Ea_eV = 0.059130
        self.T_ref = 298.15
        
        self.tau = 100.0
        self.C1 = 5000.0

        self.z = 1.0
        self.vp = 0.0
        self.t = 0.0


    def get_ocv(self, z):
        z = np.clip(z, 0.001, 0.999)
        return self.k[0] + self.k[1]*z + self.k[2]*np.log(z) + self.k[3]*np.log(1-z)

    def get_resistance(self, temp_c):
        temp_k = temp_c + 273.15
        kB = 8.617e-5 
        exponent = (self.Ea_eV / kB) * (1.0/temp_k - 1.0/self.T_ref)
        return self.R_ref * np.exp(exponent)

    def solve_current(self, P_req, temp_c):
        R = self.get_resistance(temp_c)
        V_virt = self.get_ocv(self.z) - self.vp
        
        delta = V_virt**2 - 4 * R * P_req
        
        if delta < 0:
            return None, 0.0
        
        I = (V_virt - np.sqrt(delta)) / (2 * R)
        
        V_term = V_virt - I * R
        return I, V_term

    def dynamics(self, state, I):
        vp = state[1]
        dz = -I / self.Q_rated
        dvp = -vp / self.tau + I / self.C1
        return np.array([dz, dvp])

    def step(self, dt, P_req, temp_c):
        I, V_term = self.solve_current(P_req, temp_c)
        if I is None: return False, self.z, 0.0, 0.0
        if temp_c < 25.0:
            drop_per_deg = 0.008 # 0.8%
            capacity_eff = 1.0 - drop_per_deg * (25.0 - temp_c)
            capacity_eff = max(0.2, capacity_eff)
        else:
            capacity_eff = 1.0
        
        def dynamics(state, I_curr, eff):
            vp_curr = state[1]
            dz = -I_curr / (self.Q_rated * eff) 
            dvp = -vp_curr / self.tau + I_curr / self.C1
            return np.array([dz, dvp])

        s = np.array([self.z, self.vp])
        
        k1 = dynamics(s, I, capacity_eff)
        k2 = dynamics(s + 0.5*dt*k1, I, capacity_eff)
        k3 = dynamics(s + 0.5*dt*k2, I, capacity_eff)
        k4 = dynamics(s + dt*k3, I, capacity_eff)
        
        s_new = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        self.z = np.clip(s_new[0], 0, 1)
        self.vp = s_new[1]
        
        return True, self.z, V_term, I

class DualModeEKF:
    def __init__(self, battery_model):
        self.batt = battery_model
        self.P = np.diag([1e-4, 1e-4]) 
        self.Q = np.diag([1e-6, 1e-5])
        self.R = 1e-3
        
    def get_jacobian_A(self, dt):
        A = np.eye(2)
        A[1, 1] = np.exp(-dt / self.batt.tau)
        return A
        
    def get_jacobian_H(self, z):
        # H = dV/dx = [dOCV/dz, -1]
        k = self.batt.k
        z_safe = np.clip(z, 1e-3, 1-1e-3)
        # d(ln z) = 1/z, d(ln(1-z)) = -1/(1-z)
        d_ocv = k[1] + k[2]/z_safe - k[3]/(1-z_safe)
        return np.array([[d_ocv, -1]])

    def predict(self, dt, u_current=0.0):
        A = self.get_jacobian_A(dt)
        self.P = A @ self.P @ A.T + self.Q
        return self.P

    def update(self, V_meas):
        H = self.get_jacobian_H(self.batt.z)
        V_virt = self.batt.get_ocv(self.batt.z) - self.batt.vp
        pass

    def correct_state(self, V_meas, V_pred):
        H = self.get_jacobian_H(self.batt.z)
        y = V_meas - V_pred # Innovation
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        dx = K.flatten() * y
        
        new_z = self.batt.z + dx[0]
        new_vp = self.batt.vp + dx[1]
        
        self.batt.z = np.clip(new_z, 0, 1)
        self.batt.vp = new_vp
        
        self.P = (np.eye(2) - K @ H) @ self.P
        return dx


def run_scenario_simulation(csv_path, power_model, batt_model):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    print(f"--> Simulating: {os.path.basename(csv_path)} ({len(df)} steps)...")
    
    results = {'time': [], 'soc': [], 'voltage': [], 'power': [], 'temp': []}
    dt = 1.0
    
    batt_model.z = 1.0
    batt_model.vp = 0.0
    
    alive = True
    
    for _, row in df.iterrows():
        t = row['time_s']
        u_load = {
            'cpu': row['cpu'],
            's_on': row['s_on'],
            'br': row['br'],
            'net_kbps': row['net_kbps']
        }
        
        current_temp = row['temp_c'] if 'temp_c' in row else 35.0
        v_now_guess = batt_model.get_ocv(batt_model.z) - batt_model.vp
        p_req = power_model.predict_power(u_load, v_now_guess)
        alive, soc, v_term, i_load = batt_model.step(dt, p_req, current_temp)
        
        results['time'].append(t / 3600.0)
        results['soc'].append(soc * 100)
        results['voltage'].append(v_term)
        results['power'].append(p_req)
        results['temp'].append(current_temp)
        
        if not alive or soc <= 0:
            print(f"    [End] {t/3600.0:.2f} hours")
            break
            
    return pd.DataFrame(results)


if __name__ == "__main__":
    SCENARIO_DIR = "scenarios"
    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    
    power_model = HybridPowerModel("catboost_model.cbm")
    
    scenarios_to_run = [
        "scenario_gamer.csv",
        "scenario_binge_watcher.csv", 
        "scenario_scroller.csv",
        "scenario_office.csv",
        "scenario_polar.csv"
    ]
    
    colors = {
        "scenario_gamer": "tab:red", 
        "scenario_binge_watcher": "tab:orange", 
        "scenario_scroller": "tab:purple",
        "scenario_office": "tab:green",
        "scenario_polar": "tab:cyan"
    }
    
    plt.figure(figsize=(12, 8))
    
    final_report = []

    for fname in scenarios_to_run:
        path = os.path.join(SCENARIO_DIR, fname)
        label_name = fname.replace(".csv", "")
        batt_model = BatteryECM(capacity_Ah=4.575, soh=1.0)
        
        res = run_scenario_simulation(path, power_model, batt_model)
        
        if res is not None and len(res) > 0:
            tte = res['time'].iloc[-1]
            final_soc = res['soc'].iloc[-1]
            avg_temp = res['temp'].mean()
            
            final_report.append({
                "Scenario": label_name,
                "TTE (Hours)": tte,
                "Final SOC": final_soc,
                "Avg Temp": avg_temp
            })
            
            color = colors.get(label_name, 'black')
            plt.plot(res['time'], res['soc'], label=f"{label_name} (TTE={tte:.1f}h)", color=color, linewidth=2)

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("State of Charge (%)", fontsize=12)
    plt.title("Battery Comparison: 5 User Scenarios (Hybrid Model Simulation)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-5, 105)
    
    plot_path = os.path.join(RESULT_DIR, "comparison_soc.png")
    plt.savefig(plot_path, dpi=300)
    
    print("\n" + "="*50)
    print(f"{'Scenario':<25} | {'TTE (h)':<10} | {'Avg Temp':<10}")
    print("-" * 50)
    for row in final_report:
        print(f"{row['Scenario']:<25} | {row['TTE (Hours)']:<10.2f} | {row['Avg Temp']:<10.1f}")
    print("="*50)

    plt.show()