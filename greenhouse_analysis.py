import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset_sample.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# Rename columns for easier use
df.columns = ['Time', 'Temp', 'Humidity', 'SoilMoisture', 'Light', 'Fan', 'Pump', 'Latency', 'Hour']

print(f"Records: {len(df)}")
print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

# --- DESCRIPTIVE STATISTICS ---
print("\n--- STATISTICS ---")
print(f"Temperature: {df['Temp'].min():.1f} - {df['Temp'].max():.1f} C, Mean: {df['Temp'].mean():.1f}")
print(f"Humidity: {df['Humidity'].min():.1f} - {df['Humidity'].max():.1f}%, Mean: {df['Humidity'].mean():.1f}")
print(f"Soil Moisture: {df['SoilMoisture'].min():.1f} - {df['SoilMoisture'].max():.1f}%")
print(f"Latency: Mean={df['Latency'].mean():.1f}ms, P95={df['Latency'].quantile(0.95):.1f}ms")

# --- M/M/1 QUEUING MODEL ---
print("\n--- M/M/1 QUEUING MODEL ---")
lambda_rate = 1/60  # arrival: 1 reading per minute
mu_rate = 1/(df['Latency'].mean()/1000)  # service rate
rho = lambda_rate / mu_rate  # traffic intensity

print(f"Arrival rate (lambda): {lambda_rate:.4f} per second")
print(f"Service rate (mu): {mu_rate:.2f} per second")
print(f"Traffic intensity (rho): {rho:.6f}")

if rho < 1:
    L = rho / (1 - rho)  # avg queue length
    W = 1 / (mu_rate - lambda_rate)  # avg wait time
    print(f"Avg queue length (L): {L:.6f}")
    print(f"Avg wait time (W): {W*1000:.2f} ms")
    print("Status: STABLE")

# --- LITTLE'S LAW ---
print("\n--- LITTLE'S LAW: L = lambda * W ---")
L_calc = lambda_rate * W
print(f"L from formula: {L:.6f}")
print(f"L from Little's Law: {L_calc:.6f}")
print(f"Verified: {abs(L - L_calc) < 0.0001}")

# --- CORRELATION ---
print("\n--- CORRELATIONS ---")
print(f"Temp vs Humidity: {df['Temp'].corr(df['Humidity']):.3f}")
print(f"Temp vs Fan State: {df['Temp'].corr(df['Fan']):.3f}")
print(f"SoilMoisture vs Pump: {df['SoilMoisture'].corr(df['Pump']):.3f}")

# --- FIGURES ---

# Figure 1: Environmental data
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df['Time'], df['Temp'], 'r-')
axes[0].set_ylabel('Temperature (C)')
axes[0].set_title('Greenhouse Environmental Data')
axes[1].plot(df['Time'], df['Humidity'], 'b-')
axes[1].set_ylabel('Humidity (%)')
axes[2].plot(df['Time'], df['SoilMoisture'], 'g-')
axes[2].set_ylabel('Soil Moisture (%)')
axes[2].set_xlabel('Time')
plt.tight_layout()
plt.savefig('fig1_environment.png', dpi=150)
plt.close()

# Figure 2: Latency distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['Latency'], bins=30, color='steelblue', edgecolor='black')
ax.axvline(df['Latency'].mean(), color='red', linestyle='--', label=f"Mean: {df['Latency'].mean():.1f}ms")
ax.axvline(df['Latency'].quantile(0.95), color='orange', linestyle='--', label=f"P95: {df['Latency'].quantile(0.95):.1f}ms")
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Control Latency Distribution')
ax.legend()
plt.tight_layout()
plt.savefig('fig2_latency.png', dpi=150)
plt.close()

# Figure 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cols = ['Temp', 'Humidity', 'SoilMoisture', 'Light', 'Latency']
corr = df[cols].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45)
ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center')
plt.colorbar(im, label='Correlation')
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('fig3_correlation.png', dpi=150)
plt.close()

# Figure 4: M/M/1 Queue model
fig, ax = plt.subplots(figsize=(8, 5))
rho_vals = np.linspace(0.01, 0.95, 50)
L_vals = rho_vals / (1 - rho_vals)
ax.plot(rho_vals, L_vals, 'b-', linewidth=2)
ax.axvline(rho, color='red', linestyle='--', label=f'Current rho={rho:.4f}')
ax.set_xlabel('Traffic Intensity (rho)')
ax.set_ylabel('Queue Length (L)')
ax.set_title('M/M/1 Model: Queue Length vs Traffic Intensity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_queuing.png', dpi=150)
plt.close()

# Figure 5: Actuator activity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fan_pct = df['Fan'].mean() * 100
pump_pct = df['Pump'].mean() * 100
axes[0].bar(['Fan', 'Pump'], [fan_pct, pump_pct], color=['coral', 'skyblue'])
axes[0].set_ylabel('ON Time (%)')
axes[0].set_title('Actuator Utilization')

hourly_lat = df.groupby('Hour')['Latency'].mean()
axes[1].bar(hourly_lat.index, hourly_lat.values, color='steelblue')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Latency (ms)')
axes[1].set_title('Avg Latency by Hour')
plt.tight_layout()
plt.savefig('fig5_actuators.png', dpi=150)
plt.close()

print("\n--- FIGURES SAVED ---")
print("fig1_environment.png")
print("fig2_latency.png")
print("fig3_correlation.png")
print("fig4_queuing.png")
print("fig5_actuators.png")
