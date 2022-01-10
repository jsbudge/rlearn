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


games = 2
eval_games = 1
max_timesteps = 128
batch_sz = 64

# Pre-defined or custom environment
env = SinglePulseBackground(max_timesteps)

# Instantiate a Tensorforce agent
wave_agent = Agent.create(agent='a2c', states=dict(cpi=dict(type='float', shape=(env.nsam, env.cpi_len)),
                                                   currwave=dict(type='float', shape=(100, env.n_tx), min_value=0, max_value=1),
                    currfc=dict(type='float', shape=(env.n_tx,), min_value=8e9, max_value=12e9),
                    currbw=dict(type='float', shape=(env.n_tx,), min_value=10e6, max_value=env.fs / 2 - 5e6)),
                          actions=dict(wave=dict(type='float', shape=(100, env.n_tx), min_value=0, max_value=1),
                    fc=dict(type='float', shape=(env.n_tx,), min_value=8e9, max_value=12e9),
                    bw=dict(type='float', shape=(env.n_tx,), min_value=10e6, max_value=env.fs / 2 - 5e6)
                                       ), max_episode_timesteps=max_timesteps, batch_size=batch_sz, discount=.9, learning_rate=5e-4,
                          memory=max_timesteps)
motion_agent = Agent.create(agent='a2c', states=dict(cpi=dict(type='float', shape=(env.nsam, env.cpi_len)),
                                                     currscan=dict(type='float', shape=(1,), min_value=env.az_lims[0], max_value=env.az_lims[1]),
                                                     currelscan=dict(type='float', shape=(1,), min_value=env.el_lims[0], max_value=env.el_lims[1])),
                            actions=dict(radar=dict(type='float', shape=(1,), min_value=100, max_value=env.maxPRF),
                                         scan=dict(type='float', shape=(1,), min_value=env.az_lims[0], max_value=env.az_lims[1]),
                                         elscan=dict(type='float', shape=(1,), min_value=env.el_lims[0], max_value=env.el_lims[1])),
                            max_episode_timesteps=max_timesteps, batch_size=batch_sz, discount=.9, learning_rate=5e-5,
                            memory=max_timesteps)
#agent = Agent.create(agent='a2c', environment=env, batch_size=batch_sz, discount=.9, learning_rate=5e-4,
#                     memory=max_timesteps)

# Train for 100 episodes
for episode in tqdm(range(games)):

    # Episode using act and observe
    states = env.reset()
    terminal = False
    sum_rewards = 0.0
    num_updates = 0
    while not terminal:
        wave_actions = wave_agent.act(states=dict(cpi=states['cpi'],
                    currwave=states['currwave'], currfc=states['currfc'], currbw=states['currbw']))
        motion_actions = motion_agent.act(states=dict(cpi=states['cpi'], currscan=states['currscan'],
                    currelscan=states['currelscan']))
        actions = {**wave_actions, **motion_actions}
        states, terminal, reward = env.execute(actions=actions)
        num_updates += wave_agent.observe(terminal=terminal, reward=reward[0])
        num_updates += motion_agent.observe(terminal=terminal, reward=reward[1])
        sum_rewards += sum(reward)
    print('Episode {}: \nReturn\t{}\nUpdates:\t{}'.format(episode, sum_rewards, num_updates // 2))

# Testing loop
print('Evaluation...')
rewards = []
for g in tqdm(range(eval_games)):
    # Initialize episode
    states = env.reset()
    terminal = False
    internals = [wave_agent.initial_internals(), motion_agent.initial_internals()]
    timestep = 0

    while not terminal:
        # Episode timestep
        wa, internals[0] = wave_agent.act(states=dict(cpi=states['cpi'],
                    currwave=states['currwave'], currfc=states['currfc'], currbw=states['currbw']), internals=internals[0],
                                       independent=True, deterministic=True)
        ma, internals[1] = motion_agent.act(states=dict(cpi=states['cpi'], currscan=states['currscan'],
                    currelscan=states['currelscan']), internals=internals[1],
                                       independent=True, deterministic=True)
        actions = {**wa, **ma}
        states, terminal, reward = env.execute(actions=actions)
        rewards.append(reward)

env.close()
# wave_agent.close()
# motion_agent.close()

logs = env.log
log_num = 10

nr = int((env.env.max_pl * env.plp) * env.fs)
back_noise = np.random.rand(max(nr, 5000)) - .5 + 1j * (np.random.rand(max(nr, 5000)) - .5)

plt.figure('RC Pulses')
for ant in range(env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, ant],
                     env.nr, env.nr / env.fs, env.fc[ant], env.bw[ant])
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
gs = gridspec.GridSpec(3, env.n_tx)
axes = []
for ant in range(env.n_tx):
    axes.append(plt.subplot(gs[0, ant]))
axes.append(plt.subplot(gs[1, :]))
axes.append(plt.subplot(gs[2, :]))
# Calc Doppler shifts and velocities
camera = Camera(fig)
for l in logs:
    fpos = env.env.pos(l[2])[:, 0]
    dopp_freqs = np.fft.fftshift(np.fft.fftfreq(l[0].shape[1], (l[2][1] - l[2][0]))) / env.fc[0] * c0

    # Draw the beam and platform
    bm_x, bm_y, bm_a, bm_b = env.env.getAntennaBeamLocation(l[2][0], l[3][0], l[4][0])
    main_beam = ellipse(bm_x, bm_y, bm_a, bm_b, l[3][0])
    axes[env.n_tx].plot(main_beam[0, :], main_beam[1, :], 'gray')
    axes[env.n_tx].scatter(fpos[0], fpos[1], marker='*', c='blue')
    beam_dir = np.exp(1j * l[3][0]) * fpos[2] / np.tan(l[4][0])
    axes[env.n_tx].arrow(fpos[0], fpos[1], beam_dir.real, beam_dir.imag)

    # Draw the targets
    for idx, s in enumerate(env.env.targets):
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
            axes[env.n_tx].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)

            # Range to target
            plt_rng = 2 * (np.linalg.norm(env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                      (l[2][-1] - l[2][0])
            axes[env.n_tx].text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    axes[env.n_tx].legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])

    # Draw the RD maps for each antenna
    for ant in range(env.n_tx):
        axes[ant].imshow(np.fft.fftshift(l[0][:, :, ant], axes=1),
                         extent=[dopp_freqs[0], dopp_freqs[-1], env.env.gnrange, env.env.gfrange])
        axes[ant].axis('tight')
        axes[env.n_tx + 1].plot(db(
            np.fft.fft(
                genPulse(np.linspace(0, 1, len(l[5][:, ant])), l[5][:, ant], env.nr, env.nr / env.fs,
                         env.fc[ant], env.bw[ant]),
                env.fft_len)), c=cols[ant])
    camera.snap()

animation = camera.animate(interval=500)
# animation.save('test.mp4')

plt.figure('Ambiguity')
cmin = None
cmax = None
for x in range(env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, x],
                     env.nr, env.nr / env.fs, env.fc[x], env.bw[x])
    window_pulse = np.fft.ifft(np.fft.fft(pulse, findPowerOf2(nr)) * taylor(findPowerOf2(nr)))
    for y in range(env.n_tx):
        plt.subplot(env.n_tx, env.n_tx, x * env.n_tx + y + 1)
        amb = ambiguity(genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, y],
                                 env.nr, env.nr / env.fs, env.fc[y], env.bw[y]),
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
xx, yy = np.meshgrid(np.linspace(0, env.env.eswath, 500), np.linspace(0, env.env.swath, 500))
bgpts = np.array([xx.flatten(), yy.flatten()])
for l in logs:
    gv, gz = env.env.getBG(bgpts.T, l[2][0])
    ax.imshow(gz.reshape(xx.shape), extent=[0, env.env.swath, 0, env.env.eswath])
    for idx, s in enumerate(env.env.targets):
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
            plt_rng = 2 * (np.linalg.norm(env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                      (l[2][-1] - l[2][0])
            ax.text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    ax.legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])
    camw.snap()

animw = camw.animate(interval=500)

plt.figure('VA positions')
ax = plt.subplot(111, projection='3d')
for n in range(env.n_rx):
    ax.scatter(env.virtual_array[0, n * env.n_tx:(n+1) * env.n_tx],
               env.virtual_array[1, n * env.n_tx:(n+1) * env.n_tx],
               env.virtual_array[2, n * env.n_tx:(n+1) * env.n_tx])
    ax.scatter(env.rx_locs[0, n], env.rx_locs[1, n], env.rx_locs[2, n], marker='*')
