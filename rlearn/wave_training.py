from tensorforce import Agent, Environment
from tensorforce.core.networks import AutoNetwork
from wave_env import SinglePulseBackground, genPulse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from useful_lib import findPowerOf2, db
from tftb.processing import WignerVilleDistribution

games = 2
eval_games = 1
max_timesteps = 100

# Pre-defined or custom environment
env = Environment.create(
    environment=SinglePulseBackground, max_episode_timesteps=max_timesteps
)

# Instantiate a Tensorforce agent
agent = Agent.create(agent='a2c', environment=env, batch_size=32, discount=.6, learning_rate=1e-4, memory=max_timesteps)

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
plot_state = []
plot_wave = []
for g in tqdm(range(eval_games)):
    # Initialize episode
    states = env.reset()
    terminal = False
    internals = agent.initial_internals()
    timestep = 0

    while not terminal:
        # Episode timestep
        actions, internals = agent.act(states=states, internals=internals,
                                          independent=True, deterministic=True)
        states, terminal, reward = env.execute(actions=actions)
        if timestep % 2 == 0:
            plot_state.append(states)
            plot_wave.append(actions['wave'])
        timestep += 1



env.close()
agent.close()

nr = int((env._environment.env.max_pl * env._environment.plp) * env._environment.fs)
pulse = genPulse(np.linspace(0, 1, 10), plot_wave[1], nr, nr / env._environment.fs,
                 env._environment.fc, env._environment.bw)
rc_pulse = db(np.fft.ifft(np.fft.fft(pulse, findPowerOf2(nr) * 4) * np.fft.fft(pulse, findPowerOf2(nr) * 4).conj().T, findPowerOf2(nr) * 8))
plt.figure('Pulse')
plt.plot(rc_pulse)

wd = WignerVilleDistribution(pulse)
wd.run()
wd.plot(kind='contour', show_tf=True)

log_states = env._environment.log
plt.figure('State')
plt.imshow(plot_state[1])
plt.axis('tight')

fig = plt.figure('Scene')
ax = plt.axes(projection='3d')
ax.plot_wireframe(env._environment.env.eg, env._environment.env.ng, env._environment.env.ug)