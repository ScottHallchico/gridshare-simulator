import os
import glob
import joblib
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from microgrid_env import MicrogridEnv


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_predictions")
DATA_PATH = os.path.join(BASE_DIR, "..", "engineered_microgrid_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "trained_agents")

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA & MODELS
# ============================================================

def load_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loaded model: {latest}")
    return joblib.load(latest)

# Load raw data
print("Loading data...")
data = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
    index_col="timestamp"
)
print(f"Loaded {len(data)} rows.")

# Load trained forecast models
demand_pkg = load_latest(os.path.join(MODEL_DIR, "demand_model_*.pkl"))
solar_pkg = load_latest(os.path.join(MODEL_DIR, "solar_model_*.pkl"))


# ============================================================
# 3. ⚡ OPTIMIZATION & PRICING SETUP ⚡
# ============================================================
print("\n⏳ Pre-calculating predictions and pricing...")

# --- A. DEMAND FORECAST ---
features_d = data[demand_pkg["feature_names"]]
demand_values = demand_pkg["model"].predict(features_d)
demand_values = np.maximum(demand_values, 0.0) # Clip negative predictions

# --- B. SOLAR FORECAST ---
features_s = data[solar_pkg["feature_names"]]
solar_values = solar_pkg["model"].predict(features_s)
solar_values = np.maximum(solar_values, 0.0)

# --- C. DYNAMIC PRICING (Time-of-Use Logic) ---
# Create an empty array for prices
price_values = np.zeros(len(data), dtype=np.float32)

# Get the hour for every row in the dataset
hours = data.index.hour

# RULE 1: Night (00:00 - 06:00) -> Cheap ($4.00)
# This incentivizes the AI to CHARGE the battery here
price_values[(hours >= 0) & (hours < 6)] = 4.0

# RULE 2: Day (06:00 - 17:00) -> Standard ($8.00)
price_values[(hours >= 6) & (hours < 17)] = 8.0

# RULE 3: Evening Peak (17:00 - 21:00) -> Expensive ($20.00)
# This incentivizes the AI to DISCHARGE (Sell) here
price_values[(hours >= 17) & (hours < 21)] = 20.0

# RULE 4: Late Night (21:00 - 00:00) -> Standard ($8.00)
price_values[(hours >= 21)] = 8.0

print("✅ Arrays ready. Pricing logic applied.")
print("   - Night: $4.00")
print("   - Day:   $8.00")
print("   - Peak:  $20.00")


# ============================================================
# 4. ENV FACTORY
# ============================================================

def make_env():
    # Pass the pre-calculated arrays into the environment
    env = MicrogridEnv(
        demand_values=demand_values,
        solar_values=solar_values,
        price_values=price_values,
        battery_capacity=5.0,
        max_charge_rate=2.0,
        unmet_penalty=2.0,
        battery_penalty=0.01 
    )
    return Monitor(env)


# Vectorized env (4 parallel environments for faster learning)
env = make_vec_env(make_env, n_envs=4)


# ============================================================
# 5. PPO AGENT SETUP
# ============================================================

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,     
    gamma=0.99,             
    n_steps=512,            
    batch_size=64,
    ent_coef=0.01,          
    clip_range=0.2,         
    verbose=1,
    tensorboard_log=os.path.join(SAVE_DIR, "tensorboard")
)


# ============================================================
# 6. TRAIN LOOP
# ============================================================

TOTAL_TIMESTEPS = 1_000_000

print(f"\n🚀 Starting PPO training for {TOTAL_TIMESTEPS} steps...\n")

model.learn(total_timesteps=TOTAL_TIMESTEPS)

print("\n✅ Training complete")


# ============================================================
# 7. SAVE AGENT
# ============================================================

agent_path = os.path.join(SAVE_DIR, "ppo_microgrid_agent")
model.save(agent_path)

print(f"💾 PPO agent saved to: {agent_path}.zip")