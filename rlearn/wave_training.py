import cupy
import keras.models
from tensorforce import Agent, Environment, Runner
from tensorflow.keras.optimizers import Adam, Adadelta
from wave_env import SinglePulseBackground, genPulse, ambiguity, ellipse
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from celluloid import Camera
from tftb.processing import WignerVilleDistribution
from cuda_kernels import applyRadiationPatternCPU, logloss_fp, weighted_bce

# Set the heap memory allotment to 1 GB (this is way more than we need)
cupy.get_default_memory_pool().set_limit(size=1024 ** 3)

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


games = 1
eval_games = 1
max_timesteps = 128
batch_sz = 32
ocean_debug = False
feedback = False
gen_data = False
save_logs = False
load_agent = False
save_agent = False
plot_profiler = True

# Parameters for the environment (and therefore the agents)
cpi_len = 64
az_bw = 45
el_bw = 40
dep_ang = 45
boresight_ang = 90
altitude = np.random.uniform(1000, 1600)
plp = .5
env_samples = 200000
fs_decimation = 8
az_lim = 90
el_lim = 20
beamform_type = 'mmse'

print('Initial memory on GPU:')
print(f'Memory: {cupy.get_default_memory_pool().used_bytes()} / {cupy.get_default_memory_pool().total_bytes()}')
print(f'Pinned Memory: {cupy.get_default_pinned_memory_pool().n_free_blocks()} free blocks')

# We want the learning rate to be *small* so that many different pulses will, on average, train the model correctly
det_model = keras.models.load_model('./id_model')
det_model.compile(optimizer=Adadelta(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
par_model = keras.models.load_model('./par_model')

# Pre-defined or custom environment
env = SinglePulseBackground(max_timesteps=max_timesteps, cpi_len=cpi_len, az_bw=az_bw, el_bw=el_bw, dep_ang=dep_ang,
                            boresight_ang=boresight_ang,
                            altitude=altitude, plp=plp, env_samples=env_samples, fs_decimation=fs_decimation,
                            az_lim=az_lim, el_lim=el_lim,
                            beamform_type=beamform_type, det_model=det_model,
                            par_model=par_model, mdl_feedback=feedback, log=save_logs, gen_train_data=gen_data,
                            randomize_startpoint=False)

# Optimization spec
opt_spec = dict(optimizer='adadelta', learning_rate=1., multi_step=5, subsampling_fraction=64,
                clipping_threshold=1e-2, linesearch_iterations=2)

delta_layer = [dict(type='linear_normalization'), dict(type='deltafier', concatenate=0)]

# Define states for different agents
wave_state = dict(wave_corr=dict(type='float', shape=(env.fft_len, env.n_tx), min_value=-100, max_value=0),
                  currfc=dict(type='float', shape=(env.n_tx,), min_value=8e9,
                              max_value=12e9),
                  currbw=dict(type='float', shape=(env.n_tx,), min_value=10e6,
                              max_value=env.fs / 2 - 5e6),
                  clutter=dict(type='float', shape=(2, env.v_ants, env.v_ants), min_value=-1, max_value=1),
                  target_angs=dict(type='float', shape=(2,), min_value=-np.pi, max_value=np.pi),
                  prf=dict(type='float', shape=(1,), min_value=0, max_value=5000.))

# Define actions for different agents
wave_action = dict(wave=dict(type='float', shape=(100, env.n_tx), min_value=0, max_value=1),
                   fc=dict(type='float', shape=(env.n_tx,), min_value=8e9, max_value=12e9),
                   bw=dict(type='float', shape=(env.n_tx,), min_value=10e6,
                           max_value=env.fs / 2 - 5e6),
                   power=dict(type='float', shape=(env.n_tx,), min_value=10,
                              max_value=150)
                   )

# Instantiate wave agent
print('Initializing agents...')
if not load_agent:
    print('Creating new Agent...')
    wave_agent = Agent.create(agent='a2c', states=wave_state,
                              actions=wave_action,
                              max_episode_timesteps=max_timesteps, batch_size=batch_sz, discount=.99,
                              critic_optimizer=opt_spec, state_preprocessing=delta_layer,
                              memory=max_timesteps, exploration=5e9, entropy_regularization=1e3, variable_noise=.1)
else:
    print('Loading agent from wave_agent')
    wave_agent = Agent.load('./wave_agent')

# Training regimen
print('Beginning training...')
reward_track = np.zeros(games)
for episode in tqdm(range(games)):
    try:
        states = env.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = wave_agent.act(states)
            states, terminal, reward = env.execute(actions=actions)
            num_updates += wave_agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        reward_track[episode] = sum_rewards
    except KeyboardInterrupt:
        reward_track = reward_track[:episode]
        break

# Testing loop
env.mdl_feedback = False  # Reset model feedback to evaluate
actions = None
end_pulses = []
print('Evaluation...')
for g in tqdm(range(eval_games)):

    # Initialize episode
    try:
        states = env.reset()
        terminal = False
        internals = wave_agent.initial_internals()
        timestep = 0

        while not terminal:
            # Episode timestep
            actions, internals = wave_agent.act(states,
                                                internals=internals,
                                                independent=True, deterministic=True)
            states, terminal, reward = env.execute(actions=actions)
    except KeyboardInterrupt:
        break
    end_pulses.append(env.log[-1])

env.close()
# wave_agent.close()
# motion_agent.close()
print('Training and evaluation completed. Running plots.')
print('Final memory on GPU:')
print(f'Memory: {cupy.get_default_memory_pool().used_bytes()} / {cupy.get_default_memory_pool().total_bytes()}')
print(f'Pinned Memory: {cupy.get_default_pinned_memory_pool().n_free_blocks()} free blocks')

'''
-------------------------------------------------------------------------------------------
------------------------------ PLOTS ------------------------------------------------------
-------------------------------------------------------------------------------------------
'''

if plot_profiler:

    logs = env.log
    log_num = min(10, len(logs) - 1)

    nr = int(((env.env.nrange * 2 / c0 - 1 / TAC) * .99 * env.plp) * env.fs)
    back_noise = np.random.normal(0, 1e-8, size=(5000,)) + 1j * np.random.normal(0, 1e-8, size=(5000,))

    if len(reward_track) >= 5:
        plt.figure('Training Reward Track')
        mav = sliding_window(reward_track, 5, func=np.mean)
        plt.plot(reward_track)
        plt.plot(mav, linestyle='dashed')
        plt.legend(['Sum', 'Moving Average'])
    else:
        print('Not enough training episodes for track display.')

    plt.figure('RC Pulse Width Over Time')
    lgns = np.linspace(0, len(logs) - 1, 5).astype(int)
    for t_idx, lgn in enumerate(lgns):
        for ant in range(env.n_tx):
            plt.subplot(1, env.n_tx, ant + 1)
            pulse = genPulse(np.linspace(0, 1, len(logs[lgn][5][:, 0])), logs[lgn][5][:, ant],
                             env.nr, env.nr / env.fs, env.fc[ant], env.bw[ant])
            fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
            rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
            plt.plot(np.arange(-len(rc_pulse) // 2, len(rc_pulse) // 2)[1:], np.fft.fftshift(rc_pulse)[1:])
            plt.title(f'Ant. {ant}')
        plt.ylabel('dB')
        plt.xlabel('Lag')
    plt.legend([f'Step {lgn}' for lgn in lgns])

    plt.figure('Evaluation Ending Pulse Widths')
    for end_log in end_pulses:
        for ant in range(env.n_tx):
            plt.subplot(1, env.n_tx, ant + 1)
            pulse = genPulse(np.linspace(0, 1, len(end_log[5][:, 0])), end_log[5][:, ant],
                             env.nr, env.nr / env.fs, env.fc[ant], env.bw[ant])
            fftpulse = np.fft.fft(pulse, findPowerOf2(nr) * 1)
            rc_pulse = db(np.fft.ifft(fftpulse * (fftpulse * taylor(findPowerOf2(nr))).conj().T, findPowerOf2(nr) * 8))
            plt.plot(np.arange(-len(rc_pulse) // 2, len(rc_pulse) // 2)[1:], np.fft.fftshift(rc_pulse)[1:])
            back_noise[len(back_noise) // 2:len(back_noise) // 2 + nr] += pulse
            plt.title(f'Ant. {ant}')
        plt.ylabel('dB')
        plt.xlabel('Lag')

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
            plt.subplot(gs[1, :]), plt.subplot(gs[2, :])]
    # Calc Doppler shifts and velocities
    camera = Camera(fig)
    for l in logs:
        fpos = env.env.pos(l[2])[:, 0]
        mot_pos = env.el_rot(env.dep_ang, env.az_rot(env.boresight_ang, env.env.vel(l[2][0])))
        pan = np.arctan2(mot_pos[1], mot_pos[0])
        el = np.arcsin(mot_pos[2] / np.linalg.norm(mot_pos))
        dopp_freqs = np.fft.fftshift(np.fft.fftfreq(l[0].shape[1], (l[2][1] - l[2][0]))) / env.fc[0] * c0

        # Draw the beam direction and platform
        axes[3].scatter(fpos[0], fpos[1], marker='*', c='blue')
        beam_dir = np.exp(-1j * pan) * fpos[2] / np.tan(el)
        beamform_dir = np.exp(-1j * (pan + l[3])) * fpos[2] / np.tan(el + l[4])
        axes[3].arrow(fpos[0], fpos[1], beam_dir.real, -beam_dir.imag)
        axes[3].arrow(fpos[0], fpos[1], beamform_dir.real, -beamform_dir.imag)

        # Draw the targets
        t_vec = None
        for idx, s in enumerate(env.targets):
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
                va = env.el_rot(el, env.az_rot(pan, env.virtual_array))
                va_u = np.exp(1j * va.T.dot(u))
                E = 1
                Y[az_a, el_a] = l[7].dot(va_u) * applyRadiationPatternCPU(*du, 1, *du, 1,
                                                                          1e-9, 1e-9,
                                                                          2 * np.pi * 9.6e9 / c0, env.az_fac,
                                                                          env.el_fac)
        Y = db(Y)
        Y = Y - Y.max()
        # fig, axes = plt.subplots(3)
        axes[1].imshow(Y, extent=[az_angs[0] / DTR, az_angs[-1] / DTR, el_angs[0] / DTR, el_angs[-1] / DTR],
                       clim=[-20, 2])
        axes[1].set_ylabel('Azimuth')
        axes[1].set_xlabel('Elevation')

        # Azimuth beam projection at target location
        az_to_target = np.arctan2(-t_vec[1], -t_vec[0]) - env.boresight_ang
        el_to_target = np.arcsin(t_vec[2] / np.linalg.norm(t_vec)) - env.dep_ang
        axes[0].plot(az_angs, Y[abs(az_angs - az_to_target) == abs(az_angs - az_to_target).min(), :].flatten(),
                     c='blue')
        axes[0].scatter(az_to_target, 1, c='red')
        axes[2].plot(el_angs, Y[:, abs(el_angs - el_to_target) == abs(el_angs - el_to_target).min()].flatten(),
                     c='blue')
        axes[2].scatter(el_to_target, 1, c='red')
        camera.snap()
    animation = camera.animate(interval=250)

    wave_fig, wave_ax = plt.subplots(2)
    wavecam = Camera(wave_fig)
    wave_ax[0].set_ylim([-30, 1])
    freqs = np.fft.fftshift(np.fft.fftfreq(env.fft_len, 1 / env.fs))
    for l in logs:
        for ant in range(env.n_tx):
            spect = np.fft.fftshift(db(
                np.fft.fft(
                    genPulse(np.linspace(0, 1, len(l[5][:, ant])), l[5][:, ant], env.nr, env.nr / env.fs,
                             env.fc[ant], env.bw[ant]),
                    env.fft_len)))
            spect = spect - spect.max()
            wave_ax[0].plot(freqs, spect, c=cols[ant])
            wave_ax[1].plot(l[5][:, ant], c=cols[ant])
        wavecam.snap()
    wave_animation = wavecam.animate(interval=250)

    plt.figure('Ambiguity')
    prf_plot = env.PRF * 2 if env.PRF is not None else 500.
    ambigs = []
    for x in range(env.n_tx):
        pulse = genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, x],
                         env.nr, env.nr / env.fs, env.fc[x], env.bw[x])
        window_pulse = np.fft.ifft(np.fft.fft(pulse, findPowerOf2(nr)) * taylor(findPowerOf2(nr)))
        for y in range(env.n_tx):
            ambigs.append(ambiguity(genPulse(np.linspace(0, 1, len(logs[log_num][5][:, 0])), logs[log_num][5][:, y],
                                             env.nr, env.nr / env.fs, env.fc[y], env.bw[y]),
                                    window_pulse, prf_plot, 150, mag=True, normalize=False)[0])
    cmin = min([np.min(amb) for amb in ambigs])
    cmax = max([np.max(amb) for amb in ambigs])
    for x in range(env.n_tx):
        for y in range(env.n_tx):
            amb_sel = x * env.n_tx + y
            plt.subplot(env.n_tx, env.n_tx, amb_sel + 1)
            plt.imshow(ambigs[amb_sel] / cmax, clim=[0, 1.])
            plt.axis('off')
    plt.colorbar()
    plt.tight_layout()

    plt.figure('Rewards')
    wave_scores = np.array([l[1] for l in logs])
    times = np.array([l[2][0] for l in logs])
    plt.title('Wave Agent')
    for sc_part in range(wave_scores.shape[1], 0, -1):
        plt.plot(times, np.sum(wave_scores[:, :sc_part], axis=1))
        plt.fill_between(times, np.sum(wave_scores[:, :sc_part], axis=1))
    plt.legend(['Detection', 'Detected', 'Ambiguity', 'Diversity'])

    # Ocean waves, for pretty picture and debugging
    if ocean_debug:
        figw, ax = plt.subplots(2)
        camw = Camera(figw)
        xx, yy = np.meshgrid(np.linspace(0, env.env.eswath, 250), np.linspace(0, env.env.swath, 250))
        bgpts = np.array([xx.flatten(), yy.flatten()])
        for l in logs:
            gv, gz = env.getBG(bgpts.T, l[2][0])
            p_pos = env.env.pos(l[2][0])
            pn = p_pos / np.linalg.norm(p_pos)
            rd = 2 * np.dot(gv, pn)[:, None] * gv - pn[None, :]
            illum = np.dot(rd, pn)
            illum[illum < 0] = 0
            ax[0].imshow(gz.reshape(xx.shape), extent=[0, env.env.swath, 0, env.env.eswath])
            ax[1].imshow(illum.reshape(xx.shape), extent=[0, env.env.swath, 0, env.env.eswath])
            plt.axis('off')
            for idx, s in enumerate(env.targets):
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

    plt.figure('VA positions')
    ax = plt.subplot(111, projection='3d')
    rot_array = env.el_rot(env.dep_ang, env.az_rot(env.boresight_ang - np.pi / 2, env.rx_locs))
    ax.scatter(env.virtual_array[0, :],
               env.virtual_array[1, :],
               env.virtual_array[2, :])
    ax.scatter(rot_array[0, :],
               rot_array[1, :],
               rot_array[2, :], marker='*')

    plt.show()

    if feedback:
        det_model.save('./id_model')
        par_model.save('./par_model')

    if save_agent:
        wave_agent.save('./wave_agent')
