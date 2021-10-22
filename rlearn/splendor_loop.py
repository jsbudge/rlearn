from tensorforce import Agent, Environment
from tensorforce.core.networks import AutoNetwork
from envs import Maze, Splendor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

n_players = 2
games = 1300
eval_games = 500
max_timesteps = 2000

# Pre-defined or custom environment
env = Environment.create(
    environment=Splendor, max_episode_timesteps=max_timesteps
)

# Instantiate a Tensorforce agent
agents = [Agent.create(agent='a2c', environment=env, batch_size=64, discount=.1, learning_rate=1e-4, memory=max_timesteps),
          Agent.create(agent='ddqn', environment=env, batch_size=64, discount=.1, learning_rate=1e-4, memory=max_timesteps)]

# Training loop
print('Training...')
for rnd in tqdm(range(games)):
    # Initialize episode
    states = env.reset()
    terminal = False
    curr_player = 0
    rnd_num = 0

    while not terminal:
        # Episode timestep
        actions = agents[curr_player].act(states=states)
        states, terminal, reward = env.execute(actions=actions)
        agents[curr_player].observe(terminal=terminal, reward=reward)
        rnd_num += 1 / n_players
        if not terminal:
            curr_player = curr_player + 1 if curr_player + 1 < n_players else 0

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
    curr_player = 0
    internals = [ag.initial_internals() for ag in agents]
    rewards = [0 for _ in range(n_players)]
    g_gems = []
    g_toks = []
    g_pts = []

    while not terminal:
        # Episode timestep
        actions, internals[curr_player] = agents[curr_player].act(states=states, internals=internals[curr_player],
                                          independent=True, deterministic=True)
        states, terminal, reward = env.execute(actions=actions)
        rewards[curr_player] += reward
        if curr_player == 0:
            g_gems.append(sum(states['state'][12 + 4 * 2 + 1, :5]))
            g_toks.append(sum(states['state'][12 + 3 * 2 + 1, :6]))
            g_pts.append(states['state'][12 + 4 * 2 + 1, 5])
        if not terminal:
            curr_player = curr_player + 1 if curr_player + 1 < n_players else 0
    rw_total.append(rewards)
    win.append(curr_player)
    t_gems.append(g_gems)
    t_toks.append(g_toks)
    t_pts.append(g_pts)


env.close()
# Close agents?

plt.figure('Rewards')
gnum = np.arange(eval_games)
for n in range(n_players):
    plt.scatter(gnum[[w == n for w in win]], np.array(rw_total)[[w == n for w in win], n])
    plt.plot(np.array(rw_total)[:, n])

p1wins = np.arange(len(win), dtype=int)[[w == 1 for w in win]]

plt.figure('Pt Summary')
t_pave = np.zeros((np.max([len(t_pts[t]) for t in range(len(win)) if win[t] == 1]),))
t_pdiv = np.zeros((np.max([len(t_pts[t]) for t in range(len(win)) if win[t] == 1]),))
for idx, g in enumerate(t_pts):
    if win[idx] == 1:
        t_pave[:len(g)] = t_pave[:len(g)] + np.array(g)
        t_pdiv[:len(g)] = t_pdiv[:len(g)] + 1
        plt.plot(g, linewidth=.1)
plt.plot(t_pave / t_pdiv, c='b')

plt.figure('Gems Summary')
t_pave = np.zeros((np.max([len(t_gems[t]) for t in range(len(win)) if win[t] == 1]),))
t_pdiv = np.zeros((np.max([len(t_gems[t]) for t in range(len(win)) if win[t] == 1]),))
for idx, g in enumerate(t_gems):
    if win[idx] == 1:
        t_pave[:len(g)] = t_pave[:len(g)] + np.array(g)
        t_pdiv[:len(g)] = t_pdiv[:len(g)] + 1
        plt.plot(g, linewidth=.1)
plt.plot(t_pave / t_pdiv, c='b')

plt.figure('Toks Summary')
t_pave = np.zeros((np.max([len(t_toks[t]) for t in range(len(win)) if win[t] == 1]),))
t_pdiv = np.zeros((np.max([len(t_toks[t]) for t in range(len(win)) if win[t] == 1]),))
for idx, g in enumerate(t_toks):
    if win[idx] == 1:
        t_pave[:len(g)] = t_pave[:len(g)] + np.array(g)
        t_pdiv[:len(g)] = t_pdiv[:len(g)] + 1
        plt.plot(g, linewidth=.1)
plt.plot(t_pave / t_pdiv, c='b')