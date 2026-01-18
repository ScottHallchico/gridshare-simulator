import sys
import os

# --- FIX 1: Add "RL_agent" to the python path ---
# This tells Python: "Look inside the RL_agent folder for files too"
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_agent_path = os.path.join(current_dir, "RL_agent")
sys.path.append(rl_agent_path)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import glob
from stable_baselines3 import PPO

# Now this import will work because we added the folder to the path!
from microgrid_env import MicrogridEnv

app = Flask(__name__)
CORS(app)

# --- FIX 2: Correct the File Paths ---
# Since backend.py is in the main folder, we don't need ".."
BASE_DIR = current_dir
MODEL_DIR = os.path.join(BASE_DIR, "model_predictions")
DATA_PATH = os.path.join(BASE_DIR, "engineered_microgrid_data.csv")

# NOTE: Check if your 'trained_agents' folder is in the main folder or inside RL_agent
# I will assume it is in the main folder for now. 
# If it's inside RL_agent, change this to: os.path.join(BASE_DIR, "RL_agent", "trained_agents", ...)
AGENT_PATH = os.path.join(BASE_DIR, "RL_agent", "trained_agents", "ppo_microgrid_agent")

# Load Data & Models
print(f"Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")

def load_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Could not find files matching: {pattern}")
    return joblib.load(max(files, key=os.path.getmtime))

print("Loading forecast models...")
demand_pkg = load_latest(os.path.join(MODEL_DIR, "demand_model_*.pkl"))
solar_pkg = load_latest(os.path.join(MODEL_DIR, "solar_model_*.pkl"))

print(f"Loading AI Agent from: {AGENT_PATH}")
model = PPO.load(AGENT_PATH)

# Pre-calculate base forecasts
features_d = data[demand_pkg["feature_names"]]
base_demand = np.maximum(demand_pkg["model"].predict(features_d), 0.0)
features_s = data[solar_pkg["feature_names"]]
base_solar = np.maximum(solar_pkg["model"].predict(features_s), 0.0)

# Pricing Logic
hours = data.index.hour
prices = np.zeros(len(data))
prices[(hours >= 0) & (hours < 6)] = 4.0
prices[(hours >= 6) & (hours < 17)] = 8.0
prices[(hours >= 17) & (hours < 21)] = 20.0
prices[(hours >= 21)] = 8.0

@app.route('/simulate', methods=['POST'])
def run_simulation():
    req_data = request.json
    num_houses = int(req_data.get('num_houses', 1))
    battery_size = float(req_data.get('battery_size', 5.0))
    
    sim_steps = 48
    
    total_grid_load = np.zeros(sim_steps)
    total_solar_gen = np.zeros(sim_steps)
    total_battery_soc = np.zeros(sim_steps)

    for i in range(num_houses):
        variation = np.random.uniform(0.8, 1.2)
        house_demand = base_demand * variation
        
        env = MicrogridEnv(
            demand_values=house_demand,
            solar_values=base_solar,
            price_values=prices,
            battery_capacity=battery_size
        )
        obs, _ = env.reset()
        
        for t in range(sim_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
            total_battery_soc[t] += (obs[0] * battery_size) 
            total_solar_gen[t] += base_solar[t]
            
            current_net = house_demand[t] - base_solar[t]
            total_grid_load[t] += max(current_net, 0)

    return jsonify({
        "hours": list(range(sim_steps)),
        "solar": total_solar_gen.tolist(),
        "grid": total_grid_load.tolist(),
        "battery": total_battery_soc.tolist()
    })

if __name__ == '__main__':
    print("⚡ Microgrid API running on http://localhost:5000")
    app.run(port=5000, debug=True)