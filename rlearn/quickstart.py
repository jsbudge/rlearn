from tensorforce import Agent, Environment
from tensorforce.core.networks import AutoNetwork
from envs import Yahtzee
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

games = 100
eval_games = 50
max_timesteps = 200

# Pre-defined or custom environment
env = Environment.create(
    environment=Yahtzee, max_episode_timesteps=max_timesteps
)

# Instantiate a Tensorforce agent
agent = Agent.create(agent='a2c', environment=env, batch_size=64, discount=.1, learning_rate=1e-4, memory=max_timesteps)

# Training loop
print('Training...')
for rnd in tqdm(range(games)):
    # Initialize episode
    states = env.reset()
    terminal = False
    rnd_num = 0

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = env.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

# Testing loop
print('Evaluation...')
rw_total = []
win = []
t_gems = []
t_toks = []
t_pts = []
for g in tqdm(range(eval_games)):
    # Initialize episode
    states = env.reset()
    terminal = False
    internals = agent.initial_internals()

    while not terminal:
        # Episode timestep
        actions, internals = agent.act(states=states, internals=internals,
                                          independent=True, deterministic=True)
        states, terminal, reward = env.execute(actions=actions)


env.close()
agent.close()