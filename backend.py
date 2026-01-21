import os
import glob
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO

from RL_agent.microgrid_env import MicrogridEnv

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_predictions")
DATA_PATH = os.path.join(BASE_DIR, "engineered_microgrid_data.csv")
AGENT_PATH = os.path.join(BASE_DIR, "RL_agent", "trained_agents", "ppo_microgrid_agent")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")

def load_latest(pattern):
    files = glob.glob(pattern)
    return joblib.load(max(files, key=os.path.getmtime))

# Forecast models
demand_pkg = load_latest(os.path.join(MODEL_DIR, "demand_model_*.pkl"))
solar_pkg = load_latest(os.path.join(MODEL_DIR, "solar_model_*.pkl"))

# PPO agent
ppo_model = PPO.load(AGENT_PATH)

# --------------------------------------------------
# PRE-COMPUTE FORECASTS
# --------------------------------------------------
features_d = data[demand_pkg["feature_names"]]
base_demand = np.maximum(demand_pkg["model"].predict(features_d), 0.0)

features_s = data[solar_pkg["feature_names"]]
base_solar = np.maximum(solar_pkg["model"].predict(features_s), 0.0)

# Pricing (Time-of-Use)
hours = data.index.hour
prices = np.zeros(len(data))
prices[(hours < 6)] = 4.0
prices[(hours >= 6) & (hours < 17)] = 8.0
prices[(hours >= 17) & (hours < 21)] = 20.0
prices[(hours >= 21)] = 8.0

# --------------------------------------------------
# API
# --------------------------------------------------
@app.route("/simulate", methods=["POST"])
def simulate():
    req = request.json or {}
    battery_size = float(req.get("battery_size", 5.0))
    steps = 48

    demand = base_demand[:steps + 1]
    solar = base_solar[:steps + 1]
    price = prices[:steps + 1]

    env = MicrogridEnv(
        demand_values=demand,
        solar_values=solar,
        price_values=price,
        battery_capacity=battery_size
    )

    obs, _ = env.reset()

    result = {
        "hour": [],
        "solar": [],
        "demand": [],
        "battery": [],
        "grid": [],
        "action": []
    }

    for t in range(steps):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        soc_kwh = obs[0] * battery_size

        net_load = demand[t] - solar[t]
        if action[0] > 0:
            net_load -= action[0] * battery_size
        else:
            net_load += abs(action[0]) * battery_size

        result["hour"].append(t)
        result["solar"].append(float(solar[t]))
        result["demand"].append(float(demand[t]))
        result["battery"].append(float(soc_kwh))
        result["grid"].append(float(max(net_load, 0)))
        result["action"].append(float(action[0]))

    return jsonify(result)

if __name__ == "__main__":
    print("⚡ PPO backend running at http://localhost:5000")
    app.run(port=5000, debug=True)
