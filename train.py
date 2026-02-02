import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

CSV_FILE = "pixel8_training_data.csv"
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

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='importance', ascending=False)

print(feature_importance)

plt.figure()
shap.summary_plot(shap_values, X_test_cb, show=False)
plt.title("SHAP Summary Plot: Impact on Power Residuals")
plt.tight_layout()
plt.savefig("shap_summary.png")

plt.figure()
shap.summary_plot(shap_values, X_test_cb, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig("shap_bar.png")


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
P_phys_sorted = pd.Series(P_phys_pred, index=test_df.index).loc[sorted_indices]
P_hybrid_sorted = pd.Series(P_hybrid_pred, index=test_df.index).loc[sorted_indices]

plt.figure(figsize=(12, 6))
plt.plot(test_df_sorted['time'], test_df_sorted['power_w'], 'k', alpha=0.4, label='Real')
plt.plot(test_df_sorted['time'], P_phys_sorted, 'g:', label='Physics')
plt.plot(test_df_sorted['time'], P_hybrid_sorted, 'r--', label='Hybrid')
plt.title(f"Model Validation (R²={r22:.3f})")
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.legend()
plt.savefig("eval_time_series.png")
