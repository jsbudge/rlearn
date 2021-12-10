from tensorforce import Agent, Environment, Runner
from wave_env import SinglePulseBackground, genPulse, ambiguity, ellipse
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera
from tftb.processing import WignerVilleDistribution


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def db(x):
    ret = abs(x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2**(np.ceil(np.log2(x))))


games = 1
eval_games = 1
max_timesteps = 128
batch_sz = 32

# Pre-defined or custom environment
env = Environment.create(
    environment=SinglePulseBackground, max_episode_timesteps=max_timesteps
)

# Instantiate a Tensorforce agent
agent = Agent.create(agent='a2c', environment=env, batch_size=batch_sz, discount=.99, learning_rate=1e-3,
                     memory=max_timesteps)

runner = Runner(agent=agent, environment=env, max_episode_timesteps=max_timesteps)
runner.run(num_episodes=games)

# Testing loop
print('Evaluation...')
rewards = []
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
        rewards.append(reward)

env.close()
agent.close()

plot_env = env._environment
logs = plot_env.log
log_num = 10

nr = int((plot_env.env.max_pl * plot_env.plp) * plot_env.fs)
pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5])), logs[log_num][5], plot_env.nr, plot_env.nr / plot_env.fs,
                 plot_env.fc, plot_env.bw)
fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
plt.figure('Pulse')
plt.plot(np.fft.fftshift(rc_pulse))
back_noise = np.random.rand(max(nr, 5000)) - .5 + 1j * (np.random.rand(max(nr, 5000)) - .5)
back_noise[:nr] += pulse

try:
    wd = WignerVilleDistribution(back_noise)
    wd.run()
    wd.plot(kind='contour', show_tf=True)
except IndexError:
    print('Pulse too small?')

cols = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'black', 'cyan']
fig, axes = plt.subplots(3)
# Calc Doppler shifts and velocities
camera = Camera(fig)
for l in logs:
    fpos = plot_env.env.pos(l[2])[:, 0]
    dopp_freqs = np.fft.fftshift(np.fft.fftfreq(l[0].shape[1], (l[2][1] - l[2][0]))) / plot_env.fc * c0
    bm_x, bm_y, bm_a, bm_b = plot_env.env.getAntennaBeamLocation(l[2][0], l[3][0], l[4][0])
    main_beam = ellipse(bm_x, -bm_y, bm_a, bm_b, l[3][0])
    axes[1].plot(main_beam[0, :], main_beam[1, :], 'gray')
    axes[1].scatter(fpos[0], fpos[1], marker='*', c='blue')
    for idx, s in enumerate(plot_env.env.targets):
        pos = []
        amp = []
        pcols = []
        for t in l[2]:
            spow, loc1, loc2 = s(t)
            pos.append([loc1, loc2])
            amp.append(spow + 1)
            pcols.append(cols[idx])
        pos = np.array(pos)
        if len(pos) > 0:
            axes[1].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)
            plt_rng = 2 * (np.linalg.norm(plot_env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(plot_env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                      (l[2][-1] - l[2][0])
            axes[1].text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    axes[1].legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])
    axes[0].imshow(np.fft.fftshift(db(l[0]), axes=1),
                   extent=[dopp_freqs[0], dopp_freqs[-1], plot_env.env.gnrange, plot_env.env.gfrange])
    axes[0].axis('tight')
    axes[2].plot(db(
        np.fft.fft(genPulse(np.linspace(0, 1, len(l[5])), l[5], plot_env.nr, plot_env.nr / plot_env.fs,
                 plot_env.fc, plot_env.bw),
                   plot_env.fft_len)), c='blue')
    camera.snap()

animation = camera.animate(interval=10)
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

plt.figure('Rewards')
scores = np.array([l[1] for l in logs])
times = np.array([l[2][0] for l in logs])
for sc_part in range(scores.shape[1] + 1, -1, -1):
    plt.plot(times, np.sum(scores[:, :sc_part], axis=1))
    plt.fill_between(times, np.sum(scores[:, :sc_part], axis=1))
