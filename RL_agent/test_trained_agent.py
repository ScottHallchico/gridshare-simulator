import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from microgrid_env import MicrogridEnv

# ============================================================
# 1. SETUP & LOAD DATA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_predictions")
DATA_PATH = os.path.join(BASE_DIR, "..", "engineered_microgrid_data.csv")
AGENT_PATH = os.path.join(BASE_DIR, "trained_agents", "ppo_microgrid_agent.zip")

def load_latest(pattern):
    files = glob.glob(pattern)
    latest = max(files, key=os.path.getmtime)
    return joblib.load(latest)

# Load raw data
data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")

# Load prediction models
demand_pkg = load_latest(os.path.join(MODEL_DIR, "demand_model_*.pkl"))
solar_pkg = load_latest(os.path.join(MODEL_DIR, "solar_model_*.pkl"))

# Pre-calculate arrays (Match the training setup)
print("Generating forecasts for simulation...")
features_d = data[demand_pkg["feature_names"]]
demand_values = np.maximum(demand_pkg["model"].predict(features_d), 0.0)

features_s = data[solar_pkg["feature_names"]]
solar_values = np.maximum(solar_pkg["model"].predict(features_s), 0.0)

# --- NEW DYNAMIC PRICING ---
hours = data.index.hour
price_values = np.zeros(len(data), dtype=np.float32)

# Apply the same rules
price_values[(hours >= 0) & (hours < 6)] = 4.0   # Cheap
price_values[(hours >= 6) & (hours < 17)] = 8.0  # Standard
price_values[(hours >= 17) & (hours < 21)] = 20.0 # Peak
price_values[(hours >= 21)] = 8.0                # Standard

# ============================================================
# 2. LOAD THE TRAINED AGENT
# ============================================================
print(f"Loading agent from: {AGENT_PATH}")
model = PPO.load(AGENT_PATH)

# ============================================================
# 3. RUN A 48-HOUR SIMULATION
# ============================================================
# We use the ARRAY-based environment (The fast one)
env = MicrogridEnv(
    demand_values=demand_values,
    solar_values=solar_values,
    price_values=price_values,
    battery_capacity=5.0
)

obs, _ = env.reset()
soc_history = []
reward_history = []
action_history = []
solar_history = []
demand_history = []

print("\nStarting 48-hour simulation...")
for i in range(48): 
    # Ask the AI what to do
    action, _ = model.predict(obs, deterministic=True)
    
    # Execute action
    obs, reward, done, truncated, _ = env.step(action)
    
    # Capture data for plotting
    current_soc_norm = obs[0] 
    soc_history.append(current_soc_norm * 5.0) # Convert to kWh
    reward_history.append(reward)
    action_history.append(action[0])
    
    # Get the raw values for this hour to compare
    solar_history.append(solar_values[i])
    demand_history.append(demand_values[i])

    print(f"Hour {i:02d} | Solar: {solar_values[i]:.2f} | Action: {action[0]:+.2f} | SOC: {current_soc_norm*100:.0f}%")

print(f"\nTotal Reward: {sum(reward_history):.2f}")

# ============================================================
# 4. PLOT THE RESULTS
# ============================================================
plt.figure(figsize=(14, 8))

# Top Plot: Energy flow
plt.subplot(2, 1, 1)
plt.plot(solar_history, label="Solar Gen (kW)", color="orange", alpha=0.7)
plt.plot(demand_history, label="Home Demand (kW)", color="red", alpha=0.5, linestyle=":")
plt.plot(soc_history, label="Battery SOC (kWh)", color="green", linewidth=2.5)
plt.title("AI Microgrid Management (48 Hours)")
plt.ylabel("Energy (kWh / kW)")
plt.legend()
plt.grid(True, alpha=0.3)

# Bottom Plot: AI Decisions
plt.subplot(2, 1, 2)
plt.bar(range(48), action_history, color=["blue" if x > 0 else "purple" for x in action_history])
plt.axhline(0, color='black', linewidth=0.5)
plt.title("AI Actions (Purple = Charge, Blue = Discharge)")
plt.ylabel("Action Intensity")
plt.xlabel("Hour")
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()