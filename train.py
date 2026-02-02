import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

MORANDI = {
    'red':    '#C66B6B',
    'orange': '#D9A66C',
    'purple': '#967BB6',
    'green':  '#759F75',
    'cyan':   '#7798B7',
    'blue':   '#5D8CA8',
    'grey':   '#A9A9A9',
    'dark_grey': '#666666',
    'bg':     '#F5F5F7',
    'grid':   '#E0E0E0'
}

def setup_plot(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor(MORANDI['bg'])
    fig.patch.set_facecolor('white')
    ax.grid(True, linestyle='--', color=MORANDI['grid'], alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    return fig, ax

CSV_FILE = "pixel8_training_data.csv"
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found.")
    exit()

df = pd.read_csv(CSV_FILE)

cols = ['cpu', 's_on', 'br', 'power_w', 'net_kbps', 'voltage_v']
for c in cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

df['br'] = df['br'].clip(lower=0)
df['br_phys'] = (df['br'] / 255.0) ** 2.2
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

def get_A(d):
    A = np.column_stack([
        np.ones(len(d)),
        d['cpu'].values,
        d['s_on'].values,
        d['s_on'].values * d['br_phys'].values,
        d['net_kbps'].values if 'net_kbps' in d else np.zeros(len(d))
    ])
    return A

A_train = get_A(train_df)
y_train = train_df['power_w'].values

lb = np.array([0.05, 1.0, 0.15, 0.5, 0.00001])
ub = np.inf * np.ones(5)

res = lsq_linear(A_train, y_train, bounds=(lb, ub), verbose=0)
p = res.x

print("-" * 30)
print(f"P_base:    {p[0]:.4f}")
print(f"a_cpu:     {p[1]:.4f}")
print(f"a_scr_base:{p[2]:.4f}")
print(f"a_scr_br:  {p[3]:.4f}")
print(f"a_net:     {p[4]:.6f}")
print("-" * 30)

P_mech_train = A_train @ p
residuals = y_train - P_mech_train

features = ['cpu', 's_on', 'br', 'voltage_v', 'net_kbps']
X_cb = train_df[features]

model = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, verbose=0)
model.fit(X_cb, residuals)
model.save_model("catboost_model.cbm")

print("\nSHAP XAI...")

explainer = shap.TreeExplainer(model)
X_test_cb = test_df[features]
shap_values = explainer.shap_values(X_test_cb)

A_test = get_A(test_df)
y_test = test_df['power_w'].values

P_phys_pred = A_test @ p
residual_pred = model.predict(X_test_cb)
P_hybrid_pred = P_phys_pred + residual_pred
P_hybrid_pred = np.maximum(P_hybrid_pred, p[0])

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return mae, rmse, mape, r2

mae1, rmse1, mape1, r21 = calc_metrics(y_test, P_phys_pred)
mae2, rmse2, mape2, r22 = calc_metrics(y_test, P_hybrid_pred)

print(f"{'Metric':<10} | {'Physics Only':<15} | {'Hybrid (Final)':<15}")
print("-" * 45)
print(f"{'MAE':<10} | {mae1:.4f} W        | {mae2:.4f} W")
print(f"{'RMSE':<10} | {rmse1:.4f} W        | {rmse2:.4f} W")
print(f"{'MAPE':<10} | {mape1:.2f}%          | {mape2:.2f}%")
print(f"{'R²':<10} | {r21:.4f}           | {r22:.4f}")

test_df_sorted = test_df.copy().sort_values(by='time')
sorted_indices = test_df_sorted.index
window_size = 300 
if len(test_df_sorted) > window_size:
    plot_df = test_df_sorted.iloc[:window_size]
    P_phys_plot = pd.Series(P_phys_pred, index=test_df.index).loc[sorted_indices].iloc[:window_size]
    P_hybrid_plot = pd.Series(P_hybrid_pred, index=test_df.index).loc[sorted_indices].iloc[:window_size]
else:
    plot_df = test_df_sorted
    P_phys_plot = pd.Series(P_phys_pred, index=test_df.index).loc[sorted_indices]
    P_hybrid_plot = pd.Series(P_hybrid_pred, index=test_df.index).loc[sorted_indices]

fig, ax = setup_plot(f"Model Validation", "Time Sequence", "Power Consumption (W)")

ax.plot(plot_df['time'], plot_df['power_w'], 
        color=MORANDI['grey'], alpha=0.4, linewidth=5, label='Real Measurement')

ax.plot(plot_df['time'], P_phys_plot, 
        color=MORANDI['blue'], linestyle='--', linewidth=2, alpha=0.8, label='Physics-Only Model')

ax.plot(plot_df['time'], P_hybrid_plot, 
        color=MORANDI['red'], linestyle='-', linewidth=2, alpha=0.9, label='Hybrid AI Model')

ax.legend(frameon=True, facecolor='white', edgecolor='#DDDDDD', loc='upper left', fontsize=11)
ax.set_ylim(bottom=0)

plt.savefig("eval_time_series.png", bbox_inches='tight')
print("\nSaved improved plot to eval_time_series.png")
plt.close()