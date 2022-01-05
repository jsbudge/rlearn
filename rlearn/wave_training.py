from tensorforce import Agent, Environment, Runner
from wave_env import SinglePulseBackground, genPulse, ambiguity, ellipse
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    return int(2 ** (np.ceil(np.log2(x))))


games = 1
eval_games = 1
max_timesteps = 128
batch_sz = 64

# Pre-defined or custom environment
env = Environment.create(
    environment=SinglePulseBackground, max_episode_timesteps=max_timesteps
)

# Instantiate a Tensorforce agent
agent = Agent.create(agent='a2c', environment=env, batch_size=batch_sz, discount=.9, learning_rate=5e-4,
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
back_noise = np.random.rand(max(nr, 5000)) - .5 + 1j * (np.random.rand(max(nr, 5000)) - .5)

plt.figure('RC Pulses')
for ant in range(plot_env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, ant],
                     plot_env.nr, plot_env.nr / plot_env.fs, plot_env.fc, plot_env.bw)
    fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
    rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
    plt.plot(np.fft.fftshift(rc_pulse))
    back_noise[:nr] += pulse

try:
    wd = WignerVilleDistribution(back_noise)
    wd.run()
    wd.plot(kind='contour', show_tf=True)
except IndexError:
    print('Pulse too small?')

cols = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'black', 'cyan']
fig = plt.figure()
gs = gridspec.GridSpec(3, plot_env.n_rx)
axes = []
for ant in range(plot_env.n_rx):
    axes.append(plt.subplot(gs[0, ant]))
axes.append(plt.subplot(gs[1, :]))
axes.append(plt.subplot(gs[2, :]))
# Calc Doppler shifts and velocities
camera = Camera(fig)
for l in logs:
    fpos = plot_env.env.pos(l[2])[:, 0]
    dopp_freqs = np.fft.fftshift(np.fft.fftfreq(l[0].shape[1], (l[2][1] - l[2][0]))) / plot_env.fc * c0
    bm_x, bm_y, bm_a, bm_b = plot_env.env.getAntennaBeamLocation(l[2][0], np.pi/2 + l[3][0], l[4][0])
    main_beam = ellipse(bm_x, -bm_y, bm_a, bm_b, l[3][0])
    axes[plot_env.n_rx].plot(main_beam[0, :], main_beam[1, :], 'gray')
    axes[plot_env.n_rx].scatter(fpos[0], fpos[1], marker='*', c='blue')
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
            axes[plot_env.n_rx].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)
            plt_rng = 2 * (np.linalg.norm(plot_env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(plot_env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                      (l[2][-1] - l[2][0])
            axes[plot_env.n_rx].text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    axes[plot_env.n_rx].legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])
    for ant in range(plot_env.n_rx):
        axes[ant].imshow(np.fft.fftshift(l[0][:, :, ant], axes=1),
                         extent=[dopp_freqs[0], dopp_freqs[-1], plot_env.env.gnrange, plot_env.env.gfrange])
        axes[ant].axis('tight')
        axes[plot_env.n_rx + 1].plot(db(
            np.fft.fft(
                genPulse(np.linspace(0, 1, len(l[5][:, ant])), l[5][:, ant], plot_env.nr, plot_env.nr / plot_env.fs,
                         plot_env.fc, plot_env.bw),
                plot_env.fft_len)), c=cols[ant])
    camera.snap()

animation = camera.animate(interval=500)
# animation.save('test.mp4')

plt.figure('Ambiguity')
cmin = None
cmax = None
for x in range(plot_env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, x],
                     plot_env.nr, plot_env.nr / plot_env.fs, plot_env.fc, plot_env.bw)
    window_pulse = np.fft.ifft(np.fft.fft(pulse, findPowerOf2(nr)) * taylor(findPowerOf2(nr)))
    for y in range(plot_env.n_tx):
        plt.subplot(plot_env.n_tx, plot_env.n_tx, x * plot_env.n_tx + y + 1)
        amb = ambiguity(genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, y],
                                 plot_env.nr, plot_env.nr / plot_env.fs, plot_env.fc, plot_env.bw),
                        window_pulse, actions['radar'][0] * 2, 150, mag=True, normalize=False)
        cmin = np.min(amb[0]) if cmin is None else cmin
        cmax = np.max(amb[0]) if cmax is None else cmax
        plt.imshow(amb[0], clim=[cmin, cmax])
plt.tight_layout()

plt.figure('Rewards')
scores = np.array([l[1] for l in logs])
times = np.array([l[2][0] for l in logs])
for sc_part in range(scores.shape[1] + 1, -1, -1):
    plt.plot(times, np.sum(scores[:, :sc_part], axis=1))
    plt.fill_between(times, np.sum(scores[:, :sc_part], axis=1))

figw, ax = plt.subplots(1)
camw = Camera(figw)
xx, yy = np.meshgrid(np.linspace(0, plot_env.env.eswath, 500), np.linspace(0, plot_env.env.swath, 500))
bgpts = np.array([xx.flatten(), yy.flatten()])
for l in logs:
    gv, gz = plot_env.env.getBG(bgpts.T, l[2][0])
    ax.imshow(gz.reshape(xx.shape), extent=[0, plot_env.env.swath, 0, plot_env.env.eswath])
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
            ax.scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)
            plt_rng = 2 * (np.linalg.norm(plot_env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(plot_env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                      (l[2][-1] - l[2][0])
            ax.text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    ax.legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])
    camw.snap()

animw = camw.animate(interval=500)

plt.figure('VA positions')
ax = plt.subplot(111, projection='3d')
for n in range(plot_env.n_rx):
    ax.scatter(plot_env.virtual_array[0, n * plot_env.n_tx:(n+1) * plot_env.n_tx],
               plot_env.virtual_array[1, n * plot_env.n_tx:(n+1) * plot_env.n_tx],
               plot_env.virtual_array[2, n * plot_env.n_tx:(n+1) * plot_env.n_tx])
    ax.scatter(plot_env.rx_locs[0, n], plot_env.rx_locs[1, n], plot_env.rx_locs[2, n], marker='*')
