"""Not Working"""

# import pandas as pd
# import yfinance as yf
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from gymnasium import spaces
# import numpy as np

# class StockSelectionEnv(gym.Env):
#     def __init__(self, df):
#         super(StockSelectionEnv, self).__init__()

#         self.df = df
#         self.stocks = df.columns.get_level_values(1).unique().tolist()

#         # Action space: Select a stock from the list (Discrete action)
#         self.action_space = spaces.Discrete(len(self.stocks))

#         # Observation space: Historical stats of all stocks
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(len(self.stocks), 3), dtype=np.float32
#         )

#     def _next_observation(self):
#         """Get stock features like price growth, volatility, Sharpe ratio"""
#         features = []
#         for stock in self.stocks:
#             # Extract the close series for the given stock by slicing the columns along axis=1
#             close_series = self.df.xs(stock, axis=1, level=1)['Close']
#             # Now, close_series should be a 1D Series of closing prices
#             past_returns = close_series.pct_change().fillna(0)

#             # Compute mean and standard deviation (using ddof=0 for numpy-style std)
#             mean_return = past_returns.mean()
#             std_return = past_returns.std(ddof=0) + 1e-6  # add a small epsilon to prevent division by zero
#             sharpe_ratio = mean_return / std_return
#             avg_growth = float(mean_return)
#             volatility = float(past_returns.std(ddof=0))

#             # Force each feature to be a scalar float and add to the list
#             features.append(np.array([float(sharpe_ratio), avg_growth, volatility], dtype=np.float32))

#         return np.array(features, dtype=np.float32)




#     def step(self, action):
#         """Select a stock and return its past performance"""
#         selected_stock = self.stocks[action]

#         # Reward: Choose stocks with **higher Sharpe ratio and positive growth**
#         past_returns = self.df.xs(selected_stock, level=1)['Close'].pct_change().fillna(0)
#         sharpe_ratio = past_returns.mean() / (past_returns.std() + 1e-6)
#         avg_growth = np.mean(past_returns)
#         reward = sharpe_ratio + avg_growth

#         done = False
#         return self._next_observation(), reward, done, {"selected_stock": selected_stock}

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         return self._next_observation(), {}

# if __name__ == "__main__":

#     dow_jones_tickers = [
#             "MMM",  # 3M Company
#             "AXP",  # American Express
#             "AMGN", # Amgen
#             "AAPL", # Apple
#             "BA",   # Boeing
#             "CAT",  # Caterpillar
#             "CVX",  # Chevron
#             "CSCO", # Cisco Systems
#             "KO",   # Coca-Cola
#             "DOW",  # Dow Inc.
#             "GS",   # Goldman Sachs
#             "HD",   # Home Depot
#             "HON",  # Honeywell International
#             "IBM",  # IBM
#             "INTC", # Intel
#             "JNJ",  # Johnson & Johnson
#             "JPM",  # JPMorgan Chase
#             "MCD",  # McDonald's
#             "MRK",  # Merck & Co.
#             "MSFT", # Microsoft
#             "NKE",  # Nike
#             "PG",   # Procter & Gamble
#             "CRM", #salesforce
#             "TRV", #Travelers Companies Inc.
#             "UNH", #UnitedHealth Group
#             "V",   # Visa
#             "VZ",   # Verizon
#             "WBA",  # Walgreens Boots Alliance
#             "WMT",  # Walmart
#             "DIS" #Disney
#         ]
#     df = yf.download(tickers=dow_jones_tickers, period="10y", auto_adjust=True)
#     df["Stock"] = np.tile(dow_jones_tickers, len(df) // len(dow_jones_tickers) + 1)[: len(df)]
#     df = df.set_index(["Stock"], append=True)

#     selector_env = make_vec_env(lambda: StockSelectionEnv(df), n_envs=1)

#     selector_model = PPO("MlpPolicy", selector_env, verbose=1)
#     selector_model.learn(total_timesteps=100)

# selector_model.save("selector_ppo")
# print(df.columns.get_level_values(1).unique().to_list())
# print(df.head())
