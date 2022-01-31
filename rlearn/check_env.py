from tensorforce import Agent, Environment, Runner
from wave_env import SinglePulseBackground, genPulse, ambiguity, ellipse
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from celluloid import Camera
from tftb.processing import WignerVilleDistribution
from cuda_kernels import applyRadiationPatternCPU

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def db(db_x):
    ret = abs(db_x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def findPowerOf2(fpo2x):
    return int(2 ** (np.ceil(np.log2(fpo2x))))


def sliding_window(data, win_size, func=None):
    sz = len(data)
    thresh = np.zeros_like(data)
    for i in range(sz):
        tmp = np.concatenate((data[max(i - win_size, 0):max(i, 0)], data[min(i, sz):min(i + win_size, sz)]))
        thresh[i] = func(tmp)
    return thresh


max_timesteps = 100

# Pre-defined or custom environment
env = SinglePulseBackground(max_timesteps=max_timesteps, cpi_len=64, az_bw=24, el_bw=18, dep_ang=45, boresight_ang=90,
                            altitude=1524, plp=.5, env_samples=500000, fs_decimation=8, az_lim=90, el_lim=20,
                            beamform_type='phased')
linear = np.zeros((100, env.n_tx))
linear[:, 0] = np.linspace(0, 1, 100)
linear[:, 1] = np.linspace(1, 0, 100)


states = env.reset()
for step in tqdm(range(max_timesteps)):
    actions = dict(wave=linear,
                   fc=np.array([9.6e9 for _ in range(env.n_tx)]),
                   bw=np.array([env.fs / 2 for _ in range(env.n_tx)]),
                   power=np.array([1 for _ in range(env.n_tx)]),
                   radar=np.array([500]),
                    scan=np.array([0]),
                    elscan=np.array([0]))
    states, terminal, reward = env.execute(actions=actions)
'''
-------------------------------------------------------------------------------------------
------------------------------ PLOTS ------------------------------------------------------
-------------------------------------------------------------------------------------------
'''

logs = env.log
log_num = 10

nr = int((env.env.max_pl * env.plp) * env.fs)
back_noise = np.random.rand(max(nr, 5000)) - .5 + 1j * (np.random.rand(max(nr, 5000)) - .5)

plt.figure('RC Pulse Width')
for ant in range(env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, ant],
                     env.nr, env.nr / env.fs, env.fc[ant], env.bw[ant])
    fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
    rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
    plt.plot(np.arange(-len(rc_pulse) // 2, len(rc_pulse) // 2)[1:], np.fft.fftshift(rc_pulse)[1:])
    back_noise[:nr] += pulse
plt.ylabel('dB')
plt.xlabel('Lag')
plt.legend(['Tx_{}'.format(n + 1) for n in range(env.n_tx)])

try:
    wd = WignerVilleDistribution(back_noise)
    wd.run()
    wd.plot(kind='contour', show_tf=True)
except IndexError:
    print('Pulse too small?')


'''
--------------- ANIMATED ENVIRONMENT ----------------
'''
cols = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'black', 'cyan']
fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
axes = [plt.subplot(gs[0, 0], projection='polar'), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2], projection='polar'),
        plt.subplot(gs[1, :]), plt.subplot(gs[2, :2]), plt.subplot(gs[2, 2])]
# Calc Doppler shifts and velocities
camera = Camera(fig)
for l in logs:
    fpos = env.env.pos(l[2])[:, 0]
    az_bm = (l[3][0] + env.boresight_ang)
    el_bm = (l[4][0] + env.dep_ang)
    dopp_freqs = np.fft.fftshift(np.fft.fftfreq(l[0].shape[1], (l[2][1] - l[2][0]))) / env.fc[0] * c0

    # Draw the beam direction and platform
    axes[3].scatter(fpos[0], fpos[1], marker='*', c='blue')
    beam_dir = np.exp(1j * az_bm) * fpos[2] / np.tan(el_bm)
    axes[3].arrow(fpos[0], fpos[1], beam_dir.real, beam_dir.imag)
    if len(l[6]) > 0:
        t_dir = np.exp(1j * (l[6][0] + env.boresight_ang)) * fpos[2] / np.tan(el_bm)
        axes[3].arrow(fpos[0], fpos[1], t_dir.real, t_dir.imag)

    # Draw the targets
    t_vec = None
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
            axes[3].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)

            # Range to target
            t_vec = env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])
            plt_vel = 2 * (np.linalg.norm(t_vec) -
                           np.linalg.norm(env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                (l[2][-1] - l[2][0])
            axes[3].text(pos[0, 0], pos[0, 1],
                         f'{-plt_vel:.2f}, '
                         f'{np.linalg.norm(t_vec):.2f}',
                         c='black')
    axes[3].legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])

    # Plot the Range-Doppler beamformed data
    axes[4].imshow(np.fft.fftshift(l[0], axes=1),
                   extent=[dopp_freqs[0], dopp_freqs[-1], env.env.nrange, env.env.frange], origin='lower')
    axes[4].axis('tight')

    # Draw beamformed array pattern
    az_angs = np.linspace(-np.pi / 2, np.pi / 2, 90)
    az_angs[az_angs == 0] = 1e-9
    el_angs = np.linspace(-np.pi / 2, np.pi / 2, 90)
    el_angs[el_angs == 0] = 1e-9
    Y = np.zeros((len(az_angs), len(el_angs)), dtype=np.complex128)
    for az_a in range(len(az_angs)):
        for el_a in range(len(el_angs)):
            u = 2 * np.pi * 9.6e9 / c0 * np.array([np.cos(az_angs[az_a]) * np.sin(el_angs[el_a] + np.pi / 2),
                                                   np.sin(az_angs[az_a]) * np.sin(el_angs[el_a] + np.pi / 2),
                                                   np.cos(el_angs[el_a] + np.pi / 2)])
            du = -np.array([np.cos(az_angs[az_a]) * np.sin(el_angs[el_a]),
                                                   np.sin(az_angs[az_a]) * np.sin(el_angs[el_a]),
                                                   np.cos(el_angs[el_a])])
            va_u = np.exp(1j * env.virtual_array.T.dot(u))
            E = 1
            Y[az_a, el_a] = l[7].dot(va_u) * applyRadiationPatternCPU(*du, 1, *du, 1,
                                                     1e-9, 1e-9,
                                                     2 * np.pi * 9.6e9 / c0)
    Y = db(Y)
    Y = Y - Y.max()
    # fig, axes = plt.subplots(3)
    axes[1].imshow(Y, extent=[az_angs[0] / DTR, az_angs[-1] / DTR, el_angs[0] / DTR, el_angs[-1] / DTR], clim=[-20, 2])
    axes[1].set_ylabel('Azimuth')
    axes[1].set_xlabel('Elevation')

    # Azimuth beam projection at target location
    az_to_target = np.arctan2(-t_vec[1], -t_vec[0]) - env.boresight_ang
    el_to_target = np.arcsin(t_vec[2] / np.linalg.norm(t_vec)) - env.dep_ang
    axes[0].plot(az_angs, Y[abs(az_angs - az_to_target) == abs(az_angs - az_to_target).min(), :].flatten(), c='blue')
    axes[0].scatter(az_to_target, 1, c='red')
    axes[2].plot(el_angs, Y[:, abs(el_angs - el_to_target) == abs(el_angs - el_to_target).min()].flatten(), c='blue')
    axes[2].scatter(el_to_target, 1, c='red')

    # Draw the RD maps for each antenna
    for ant in range(env.n_tx):
        axes[5].plot(db(
            np.fft.fft(
                genPulse(np.linspace(0, 1, len(l[5][:, ant])), l[5][:, ant], env.nr, env.nr / env.fs,
                         env.fc[ant], env.bw[ant]),
                env.fft_len)), c=cols[ant])
    camera.snap()

animation = camera.animate(interval=250)
# animation.save('test.mp4')

plt.figure('Ambiguity')
cmin = None
cmax = None
prf_val = 500.0
for x in range(env.n_tx):
    pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, x],
                     env.nr, env.nr / env.fs, env.fc[x], env.bw[x])
    window_pulse = np.fft.ifft(np.fft.fft(pulse, findPowerOf2(nr)) * taylor(findPowerOf2(nr)))
    for y in range(env.n_tx):
        plt.subplot(env.n_tx, env.n_tx, x * env.n_tx + y + 1)
        prf_plot = actions['radar'][0] * 2 if actions['radar'] is not None else prf_val
        amb = ambiguity(genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, y],
                                 env.nr, env.nr / env.fs, env.fc[y], env.bw[y]),
                        window_pulse, prf_plot, 150, mag=True, normalize=False)
        cmin = np.min(amb[0]) if cmin is None else cmin
        cmax = np.max(amb[0]) if cmax is None else cmax
        if x == y:
            plt.imshow(amb[0])
        else:
            plt.imshow(amb[0], clim=[cmin, cmax])
plt.tight_layout()

plt.figure('Rewards')
scores = np.array([l[1] for l in logs])
times = np.array([l[2][0] for l in logs])
for sc_part in range(scores.shape[1] + 1, -1, -1):
    plt.plot(times, np.sum(scores[:, :sc_part], axis=1))
    plt.fill_between(times, np.sum(scores[:, :sc_part], axis=1))

figw, ax = plt.subplots(2)
camw = Camera(figw)
xx, yy = np.meshgrid(np.linspace(0, env.env.eswath, 250), np.linspace(0, env.env.swath, 250))
bgpts = np.array([xx.flatten(), yy.flatten()])
for l in logs:
    gv, gz = env.env.getBG(bgpts.T, l[2][0])
    p_pos = env.env.pos(l[2][0])
    pn = p_pos / np.linalg.norm(p_pos)
    rd = 2 * np.dot(gv, pn)[:, None] * gv - pn[None, :]
    illum = np.dot(rd, pn)
    illum[illum < 0] = 0
    ax[0].imshow(gz.reshape(xx.shape), extent=[0, env.env.swath, 0, env.env.eswath])
    ax[1].imshow(illum.reshape(xx.shape), extent=[0, env.env.swath, 0, env.env.eswath])
    plt.axis('off')
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
            ax[0].scatter(pos[:, 0], pos[:, 1], s=amp, c=pcols)
            plt_rng = 2 * (np.linalg.norm(env.env.pos(l[2])[:, -1] - np.array([pos[-1, 0], pos[-1, 1], 1])) -
                           np.linalg.norm(env.env.pos(l[2])[:, 0] - np.array([pos[0, 0], pos[0, 1], 1]))) / \
                (l[2][-1] - l[2][0])
            ax[0].text(pos[0, 0], pos[0, 1], f'{-plt_rng:.2f}', c='black')
    ax[0].legend([f'{1 / (l[2][1] - l[2][0]):.2f}Hz: {l[2][-1]:.6f}'])
    ax[0].axis('off')
    camw.snap()

animw = camw.animate(interval=250)
# animw.save('ocean.mp4')

plt.figure('VA positions')
ax = plt.subplot(111, projection='3d')
rot_array = env.el_rot(env.dep_ang, env.az_rot(env.boresight_ang - np.pi / 2, env.rx_locs))
ax.scatter(env.virtual_array[0, :],
           env.virtual_array[1, :],
           env.virtual_array[2, :])
ax.scatter(rot_array[0, :],
           rot_array[1, :],
           rot_array[2, :], marker='*')