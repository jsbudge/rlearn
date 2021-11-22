from tensorforce import Agent, Environment
from wave_env import SinglePulseBackground, genPulse, ambiguity
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera
from tftb.processing import WignerVilleDistribution


def db(x):
    ret = abs(x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2**(np.ceil(np.log2(x))))


games = 5
eval_games = 1
max_timesteps = 400

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

plot_env = env._environment
log_num = 10

nr = int((plot_env.env.max_pl * plot_env.plp) * plot_env.fs)
pulse = genPulse(np.linspace(0, 1, 10), plot_wave[log_num], nr, nr / plot_env.fs,
                 plot_env.fc, plot_env.bw)
fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
plt.figure('Pulse')
plt.plot(rc_pulse)

wd = WignerVilleDistribution(pulse)
wd.run()
wd.plot(kind='contour', show_tf=True)

logs = plot_env.log
cols = ['blue', 'red', 'orange', 'yellow', 'green']
up_cols = ['red', 'blue', 'green', 'orange', 'yellow']
for s in plot_env.env.targets:
    s.reset()
fig, axes = plt.subplots(2)
camera = Camera(fig)
for l in logs[::2]:
    for idx, s in enumerate(plot_env.env.targets):
        pos = []
        amp = []
        pcols = []
        for t in l[2]:
            spow, loc1, loc2 = s(t)
            pos.append([loc1, loc2])
            amp.append(spow + 1)
            pcols.append(cols[idx] if spow == 0 else up_cols[idx])
        pos = np.array(pos)
        if len(pos) > 0:
            axes[1].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)
    axes[1].legend([f'{l[2][0]:.6f}-{l[2][-1]:.6f}'])
    axes[0].imshow(np.fft.fftshift(l[0], axes=1))
    axes[0].axis('tight')
    camera.snap()

animation = camera.animate()
animation.save('test.mp4')

window_pulse = np.fft.ifft(fftpulse * taylor(findPowerOf2(nr)))
amb = ambiguity(pulse, window_pulse, actions['radar'][0] * 2, 150)

plt.figure('Ambiguity')
plt.subplot(3, 1, 1)
plt.imshow(amb[0])
plt.subplot(3, 1, 2)
plt.plot(amb[0][:, 75])
plt.subplot(3, 1, 3)
plt.plot(amb[0][75, :])
