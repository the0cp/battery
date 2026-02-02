import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RATED_CAPACITY = 4.861208 

K_BY_TEMP = {
    5:  (3.660153, 0.339290, 0.216901, -0.039509),
    25: (3.655205, 0.374104, 0.210354, -0.033974),
    35: (3.616168, 0.434726, 0.202936, -0.027255),
}

def ocv_model(soc, k):
    soc = np.clip(soc, 1e-6, 1 - 1e-6)
    k0, k1, k2, k3 = k
    return k0 + k1 * soc + k2 * np.log(soc) + k3 * np.log(1 - soc)

def process_file(path, temp_c):
    df = pd.read_excel(path, engine='openpyxl')
    if 'Step_Index' in df.columns and 5 in df['Step_Index'].values:
        df = df[df['Step_Index'] == 5].copy()

    df['dt'] = df['Test_Time(s)'].diff().fillna(0)
    df['Ah'] = (df['Current(A)'].abs() * df['dt']).cumsum() / 3600.0
    df['SOC'] = 1 - df['Ah'] / RATED_CAPACITY
    df_calc = df[(df['SOC'] > 0.2) & (df['SOC'] < 0.8)].copy()
    
    if df_calc.empty:
        print(f"Warning: {path} has no data in SOC 0.2-0.8 range.")
        return None

    i = df_calc['Current(A)'].abs().to_numpy()
    v_meas = df_calc['Voltage(V)'].to_numpy()
    soc = df_calc['SOC'].to_numpy()

    m = i > 0.1
    if not np.any(m):
        return None
    
    i = i[m]
    v_meas = v_meas[m]
    soc = soc[m]
    v_est = ocv_model(soc, K_BY_TEMP[temp_c])
    r_inst = (v_est - v_meas) / i
    return float(np.median(r_inst))

def arrhenius_func(T_k, R_ref, Ea_eV):
    kB = 8.617e-5  # Boltzmann constant eV/K
    T_ref = 298.15 # 25 C
    return R_ref * np.exp((Ea_eV / kB) * (1 / T_k - 1 / T_ref))

files = {
    5:  'NMC_k1_1C_05degC.xlsx',
    25: 'NMC_k1_1C_25degC.xlsx',
    35: 'NMC_k1_1C_35degC.xlsx',
}

temps_val, rs_val = [], []

print(f"{'Temp(C)':<10} | {'R_est (Ohm)':<15}")
print("-" * 30)

for t_c, path in files.items():
    if os.path.exists(path):
        r = process_file(path, t_c)
        if r is not None:
            temps_val.append(t_c + 273.15)
            rs_val.append(r)
            print(f"{t_c:<10} | {r:.6f}")
    else:
        print(f"File not found: {path}")

if len(temps_val) >= 3:
    try:
        p0 = [0.05, 0.2] 
        popt, pcov = curve_fit(arrhenius_func, temps_val, rs_val, p0=p0)
        
        print("-" * 30)
        print(f"self.R_ref (at 25C) = {popt[0]:.6f} Ohm")
        print(f"self.Ea_R (Activation) = {popt[1]:.6f} eV")
        
        plt.figure(figsize=(8, 5))
        t_plot = np.linspace(0, 45, 100) + 273.15
        r_plot = arrhenius_func(t_plot, *popt)
        
        plt.plot(np.array(temps_val)-273.15, rs_val, 'ro', label='Calculated R')
        plt.plot(t_plot-273.15, r_plot, 'b--', label=f'Arrhenius Fit (Ea={popt[1]:.3f}eV)')
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Internal Resistance (Ohm)")
        plt.title("Arrhenius Parameter Extraction")
        plt.grid(True)
        plt.legend()
        plt.savefig("arrhenius_final.png")
        print("Plot saved to arrhenius_final.png")
        
    except Exception as e:
        print(f"Fitting failed: {e}")
else:
    print("Insufficient data for fit.")