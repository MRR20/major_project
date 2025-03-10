import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

MAX_ACCOUNT_BALANCE = 9999999999
MAX_NUM_SHARES = 9999999999
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):

    def __init__(self, df):
        super().__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype=np.float16) # change here to update observation spaces

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = random.uniform(self.df.loc[self.current_step, 'Open'], self.df.loc[self.current_step, 'Close'])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # buy
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # sell
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price


        # update net worth if it is exceeding the max_net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # make cost_basis = 0 if shares_held = 0
        if self.shares_held == 0:
            self.cost_basis = 0

    def _next_observation(self):
        frame = np.array([
            self.df.loc[self.current_step-6 : self.current_step-1, 'Open'].values/MAX_SHARE_PRICE,
            self.df.loc[self.current_step-6 : self.current_step-1, 'High'].values/MAX_SHARE_PRICE,
            self.df.loc[self.current_step-6 : self.current_step-1, 'Low'].values/MAX_SHARE_PRICE,
            self.df.loc[self.current_step-6 : self.current_step-1, 'Close'].values/MAX_SHARE_PRICE,
            self.df.loc[self.current_step-6 : self.current_step-1, 'Volume'].values/MAX_NUM_SHARES,
            self.df.loc[self.current_step-6 : self.current_step-1, 'VWAP'].values/1069.5784397582636, # change
            self.df.loc[self.current_step-6 : self.current_step-1, 'RSI'].values/96.08478275173084 # change
            ])

        obs1 = np.append(frame, [[
            self.balance/MAX_ACCOUNT_BALANCE,
            self.max_net_worth/MAX_ACCOUNT_BALANCE,
            self.shares_held/MAX_NUM_SHARES,
            self.cost_basis/MAX_SHARE_PRICE,
            self.total_shares_sold/MAX_NUM_SHARES,
            self.total_shares_value/(MAX_NUM_SHARES * MAX_SHARE_PRICE),
            ]], axis=0)

        obs = obs1
        return obs

    def reset(self):
        # reset state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values)-6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # render env screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})")
        print(f"Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})")
        print(f"Net worth: {self.net_worth} (Max net worth : {self.max_net_worth})")
        print(f"Profit: {profit}")


if __name__ == "__main__":

    """Change with required dataframe"""
    # df = pd.read_csv("data.csv")
    # df = df.sort_values("Date")
    # df.dropna(inplace=True)
    # df = df.sort_values("Date")
    # df = df.reset_index()
    env = DummyVecEnv([lambda: StockTradingEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
