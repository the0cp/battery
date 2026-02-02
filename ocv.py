import re
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

file_paths = [
    "NMC_k1_0_05C_05degC.xlsx",
    "NMC_k1_0_05C_25degC.xlsx",
    "NMC_k1_0_05C_35degC.xlsx",
]
soc_min, soc_max = 0.01, 0.99

def ocv(soc, k0, k1, k2, k3):
    return k0 + k1*soc + k2*np.log(soc) + k3*np.log(1 - soc)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

results = []

for file_path in file_paths:
    df = pd.read_excel(file_path, engine="openpyxl")
    data = df[df["Step_Index"] == 5].copy()

    data["dt"] = data["Test_Time(s)"].diff().fillna(0)
    data["Ah"] = (data["Current(A)"].abs() * data["dt"]).cumsum() / 3600
    total_cap = float(data["Ah"].max())
    data["SOC"] = 1 - (data["Ah"] / total_cap)

    m = re.search(r"_0_05C_(\d+)degC", file_path)
    temp_c = int(m.group(1)) if m else None

    print("\n" + "=" * 40)
    print(f"File: {file_path}")
    if temp_c is not None:
        print(f"Temp: {temp_c}C")
    print(f"Rated Capacity: {total_cap:.6f} Ah")

    fit_data = data[(data["SOC"] > soc_min) & (data["SOC"] < soc_max)].copy()

    p0 = [3.5, 0.5, 0.1, -0.1]
    popt, _ = curve_fit(
        ocv,
        fit_data["SOC"].to_numpy(),
        fit_data["Voltage(V)"].to_numpy(),
        p0=p0,
        maxfev=10000,
    )

    rmse = float(np.sqrt(np.mean((fit_data["Voltage(V)"] - ocv(fit_data["SOC"], *popt)) ** 2)))

    print(f"K0: {popt[0]:.6f}")
    print(f"K1: {popt[1]:.6f}")
    print(f"K2: {popt[2]:.6f}")
    print(f"K3: {popt[3]:.6f}")
    print(f"RMSE: {rmse:.6f} V")

    soc_line = np.linspace(soc_min, soc_max, 400)
    v_line = ocv(soc_line, *popt)

    results.append(
        dict(
            file=file_path,
            temp_c=temp_c,
            total_cap_ah=total_cap,
            k0=float(popt[0]),
            k1=float(popt[1]),
            k2=float(popt[2]),
            k3=float(popt[3]),
            rmse_v=rmse,
            data=data,
            soc_line=soc_line,
            v_line=v_line,
        )
    )

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(
        data["SOC"],
        data["Voltage(V)"],
        color="#34495E",
        linewidth=2,
        label="Experimental Data",
    )
    plt.plot(
        soc_line,
        v_line,
        color="#D35400",
        linestyle="--",
        linewidth=2.5,
        dashes=(5, 2),
        label="Nernst Fit",
    )
    title = "OCV Fit" + (f" @ {temp_c}°C" if temp_c is not None else "")
    plt.title(title)
    plt.xlabel("SOC", fontsize=12, color="#333333", labelpad=8)
    plt.ylabel("Voltage (V)", fontsize=12, color="#333333", labelpad=8)
    plt.legend(
        frameon=True,
        fontsize=10,
        loc="best",
        fancybox=True,
        framealpha=1,
        edgecolor="#dddddd",
    )
    plt.grid(True, linestyle="-", color="#e0e0e0")
    plt.tick_params(colors="#333333")
    plt.tight_layout()

    out_name = f"ocv_fit_{temp_c}C.png" if temp_c is not None else f"ocv_fit_{file_path}.png"
    plt.savefig(out_name)
    plt.close()
    print(f"Saved: {out_name}")

plt.figure(figsize=(10, 6), dpi=120)
for r in results:
    label = f"{r['temp_c']}°C" if r["temp_c"] is not None else r["file"]
    plt.plot(r["soc_line"], r["v_line"], linewidth=2, label=label)

plt.title("OCV Fits (overlay)")
plt.xlabel("SOC", fontsize=12, color="#333333", labelpad=8)
plt.ylabel("Voltage (V)", fontsize=12, color="#333333", labelpad=8)
plt.legend(
    frameon=True,
    fontsize=10,
    loc="best",
    fancybox=True,
    framealpha=1,
    edgecolor="#dddddd",
)
plt.grid(True, linestyle="-", color="#e0e0e0")
plt.tick_params(colors="#333333")
plt.tight_layout()
plt.savefig("ocv_fit_all.png")
plt.show()
print("Saved: ocv_fit_all.png")
