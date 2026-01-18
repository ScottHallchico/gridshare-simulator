import glob
import os
import joblib
import pandas as pd
from microgrid_env import MicrogridEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_predictions")
print("MODEL_DIR:", os.path.abspath(MODEL_DIR))
print("PKL files:", glob.glob(os.path.join(MODEL_DIR, "*.pkl")))
# ============================================================
# 1️⃣ Load latest trained forecasting models
# ============================================================

def load_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loaded model: {latest}")
    return joblib.load(latest)

demand_pkg = load_latest(
    os.path.join(MODEL_DIR, "demand_model_*.pkl")
)

solar_pkg = load_latest(
    os.path.join(MODEL_DIR, "solar_model_*.pkl")
)


# ============================================================
# 2️⃣ Load processed dataset
# ============================================================

data = pd.read_csv(
    "engineered_microgrid_data.csv",
    parse_dates=["timestamp"],
    index_col="timestamp"
)



print(f"Loaded data with {len(data)} rows")

# ============================================================
# 3️⃣ Create the RL environment
# ============================================================

env = MicrogridEnv(
    data=data,
    demand_pkg=demand_pkg,
    solar_pkg=solar_pkg,
    battery_capacity=5.0,
    grid_price=8.0
)

# ============================================================
# 4️⃣ Run a sanity-check episode (random actions)
# ============================================================

obs = env.reset()
print("Initial observation:", obs)

total_reward = 0.0

for step in range(24):  # simulate 1 day
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    print(
        f"Step {step:02d} | "
        f"Action {action[0]:+.2f} | "
        f"Reward {reward:+.2f} | "
        f"SOC {obs[0]:.2f}"
    )

    if done:
        break

print("\nTotal reward over episode:", total_reward)
