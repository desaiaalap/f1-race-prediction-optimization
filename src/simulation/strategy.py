# src/simulation/strategy.py

import numpy as np
import pandas as pd
import random
from collections import defaultdict

class RaceStrategyEnv:
    def __init__(self, driver_laps: pd.DataFrame, pit_time_loss=20):
        self.driver_laps = driver_laps
        self.total_laps = driver_laps['LapNumber'].max()
        self.pit_loss = pit_time_loss
        self.tire_wear_penalty = {"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.05}

        self.unique_compounds = driver_laps['Compound'].dropna().unique().tolist()
        self.reset()

    def reset(self):
        self.lap = 1
        self.tire = "SOFT"
        self.stint_lap = 1
        self.total_time = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return (self.lap, self.tire, self.stint_lap)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done

        base_lap_time = self.driver_laps[self.driver_laps['LapNumber'] == self.lap]['LapTime'].dt.total_seconds().mean()
        if np.isnan(base_lap_time):
            base_lap_time = 95  # fallback

        wear_penalty = self.tire_wear_penalty.get(self.tire, 0.1) * self.stint_lap
        lap_time = base_lap_time + wear_penalty

        if action == "pit":
            self.tire = random.choice(self.unique_compounds)
            self.stint_lap = 1
            lap_time += self.pit_loss
        else:
            self.stint_lap += 1

        self.total_time += lap_time
        self.lap += 1

        if self.lap > self.total_laps:
            self.done = True

        reward = -lap_time  # Minimize time

        return self._get_state(), reward, self.done

# Q-learning agent
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: {a: 0 for a in actions})
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

def run_q_learning_strategy(driver_laps, episodes=100):
    env = RaceStrategyEnv(driver_laps)
    agent = QLearningAgent(actions=["pit", "stay"])

    for ep in range(episodes):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break

    # Final strategy execution
    state = env.reset()
    steps = []
    while True:
        action = max(agent.q_table[state], key=agent.q_table[state].get)
        next_state, reward, done = env.step(action)
        steps.append((state, action, reward))
        state = next_state
        if done:
            break

    return steps, env.total_time
