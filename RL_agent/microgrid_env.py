import gymnasium
import numpy as np
from gymnasium import spaces

class MicrogridEnv(gymnasium.Env):
    """
    Optimized Microgrid Environment.
    Reads pre-calculated demand/solar values from arrays instead of running models live.
    """

    def __init__(
        self,
        # We now pass ARRAYS, not models or dataframes
        demand_values, 
        solar_values,
        price_values, # Added this for future proofing (or pass fixed 8.0s)
        battery_capacity=5.0,
        max_charge_rate=2.0,
        unmet_penalty=2.0,
        battery_penalty=0.01
    ):
        super().__init__()

        # Store arrays directly (Super fast lookup)
        self.demand_values = demand_values.astype(np.float32)
        self.solar_values = solar_values.astype(np.float32)
        self.price_values = price_values.astype(np.float32)
        
        self.n_steps = len(demand_values) - 1 # Max steps

        # System parameters
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.unmet_penalty = unmet_penalty
        self.battery_penalty = battery_penalty

        # State
        self.t = 0
        self.soc = 0.5 * self.battery_capacity

        # Actions: [-1 (Charge), +1 (Discharge)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [SOC, Demand, Solar, Price, Sin, Cos]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def _get_obs(self):
        # FAST LOOKUP: No Pandas, No Model.predict
        demand = self.demand_values[self.t]
        solar = self.solar_values[self.t]
        price = self.price_values[self.t]
        
        # Calculate time features (assuming t maps to hours cyclically)
        # Note: If you need precise calendar dates, pass sin/cos as arrays too.
        # For now, we approximate based on step count.
        hour = self.t % 24 
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.array([
            self.soc / self.battery_capacity,
            demand,
            solar,
            price,
            hour_sin,
            hour_cos
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5 * self.battery_capacity
        return self._get_obs(), {}
    
    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))

        # 1. Get current data
        demand = self.demand_values[self.t]
        solar = self.solar_values[self.t]
        price = self.price_values[self.t]

        # 2. Physics
        power = action * self.max_charge_rate  # Target power
        
        # Track what actually happens (because battery might be full/empty)
        actual_discharge = 0.0
        actual_charge = 0.0

        if power > 0: # Discharge
            actual_discharge = min(power, self.soc)
            self.soc -= actual_discharge
            
        elif power < 0: # Charge
            actual_charge = min(-power, self.battery_capacity - self.soc)
            self.soc += actual_charge

        # 3. Grid & Reward
        # The Equation: Net Load = (House Needs + Battery Needs) - (Solar + Battery Gives)
        net_load = (demand + actual_charge) - (solar + actual_discharge)
        
        # Grid is used only if Net Load is positive
        grid_import = max(net_load, 0.0)
        
        # Cost Calculation
        reward = (
            - grid_import * price               # Cost of electricity
            - self.unmet_penalty * grid_import  # Penalty for relying on grid
            - self.battery_penalty * abs(power) # Penalty for battery wear
        )
        reward = reward / 50.0 # Normalize

        # 4. Advance Time
        self.t += 1
        terminated = self.t >= self.n_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}