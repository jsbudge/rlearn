import cupy
import keras.models
from tensorforce import Agent, Environment, Runner
from tensorflow.keras.optimizers import Adam, Adadelta
from wave_env import SinglePulseBackground, genPulse, ambiguity, ellipse, getNoisePower
import numpy as np
from scipy.signal.windows import taylor
from scipy.signal import hilbert, stft
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from celluloid import Camera
from tftb.processing import WignerVilleDistribution, ambiguity
from cuda_kernels import applyRadiationPatternCPU, logloss_fp, weighted_bce
from cambiguity import amb_surf, narrow_band, wide_band

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def findPowerOf2(fpo2x):
    return int(2 ** (np.ceil(np.log2(fpo2x))))


def db(db_x):
    ret = abs(db_x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


eval_games = 1
max_timesteps = 256
batch_sz = 32
ocean_debug = False

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
beamform_type = 'phased'

# We want the learning rate to be *small* so that many different pulses will, on average, train the model correctly
det_model = keras.models.load_model('./id_model')
det_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
par_model = keras.models.load_model('./par_model')

# Pre-defined or custom environment
env = SinglePulseBackground(max_timesteps=max_timesteps, cpi_len=cpi_len, az_bw=az_bw, el_bw=el_bw, dep_ang=dep_ang,
                            boresight_ang=boresight_ang,
                            altitude=altitude, plp=plp, env_samples=env_samples, fs_decimation=fs_decimation,
                            az_lim=az_lim, el_lim=el_lim,
                            beamform_type=beamform_type, det_model=det_model,
                            par_model=par_model, mdl_feedback=False, log=False, gen_train_data=False,
                            randomize_startpoint=False)


wave_agent = Agent.load('./wave_agent')

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
            end_pulses.append(env.log[-1])
    except KeyboardInterrupt:
        break
env.close()

nr = int(((env.env.nrange * 2 / c0 - 1 / TAC) * .99 * env.plp) * env.fs)
sigma_noise = sum(getNoisePower(env.bw, True))
back_noise = np.zeros((len(end_pulses), env.det_sz), dtype=np.complex128)
for idx, end_log in enumerate(end_pulses):
    for ant in range(env.n_tx):
        pulse = genPulse(np.linspace(0, 1, len(end_log[5][:, 0])), end_log[5][:, ant],
                         env.nr, env.nr / env.fs, env.fc[ant], env.bw[ant])
        back_noise[idx, len(back_noise) // 2:len(back_noise) // 2 + nr] += pulse

back_noise += np.random.normal(0, sigma_noise, size=back_noise.shape) + \
                  1j * np.random.normal(0, sigma_noise, size=back_noise.shape)
preds = det_model.predict(back_noise)

waves = ['costas', 'frank', 'p1', 'p3', 'LFM']
plt.figure()
freqs = np.fft.fftshift(np.fft.fftfreq(findPowerOf2(8504)) * 8192)
for fnme in waves:
    fnwave = fnme + '.wave'
    back_noise = np.zeros((128, env.det_sz), dtype=np.complex128)
    with open(fnwave, 'rb') as fid:
        fc = np.fromfile(fid, 'float32', 1, '')[0]
        bw = np.fromfile(fid, 'float32', 1, '')[0]
        real_pulse = np.fromfile(fid, 'float32', -1, '')
    complex_pulse = hilbert(real_pulse)[::2]
    fftpulse = np.fft.fft(complex_pulse, len(freqs))
    back_noise[:, len(back_noise) // 2:len(back_noise) // 2 + len(complex_pulse)] += complex_pulse * 1e-7
    back_noise += np.random.normal(0, sigma_noise, size=back_noise.shape) + \
                  1j * np.random.normal(0, sigma_noise, size=back_noise.shape)
    plt.plot(freqs, np.fft.fftshift(db(np.fft.ifft(fftpulse * fftpulse.conj()))))
    print(fnme + f' det rate: {sum(env.det_model.predict(back_noise) >= .5)[0] / back_noise.shape[0] * 100.:.2f}%')
plt.legend(waves)
plt.xlabel('Lag')
plt.ylabel('Power (dB)')

waves = ['frank', 'LFM']
plt.figure()
freqs = np.fft.fftshift(np.fft.fftfreq(findPowerOf2(8504)) * 8192)
for fnme in waves:
    fnwave = fnme + '.wave'
    back_noise = np.zeros((128, env.det_sz), dtype=np.complex128)
    with open(fnwave, 'rb') as fid:
        fc = np.fromfile(fid, 'float32', 1, '')[0]
        bw = np.fromfile(fid, 'float32', 1, '')[0]
        real_pulse = np.fromfile(fid, 'float32', -1, '')
    complex_pulse = hilbert(real_pulse)[::2]
    fftpulse = np.fft.fft(complex_pulse, len(freqs))
    plt.plot(freqs, np.fft.fftshift(db(np.fft.ifft(fftpulse * fftpulse.conj()))))
for n in [0, 64, 128]:
    fftpulse = np.fft.fft(genPulse(np.linspace(0, 1, len(end_pulses[n][5][:, 0])), end_pulses[n][5][:, 0],
                             env.nr, env.nr / env.fs, env.fc[0], env.bw[0]), len(freqs))
    plt.plot(freqs, np.fft.fftshift(db(np.fft.ifft(fftpulse * fftpulse.conj()))))
add_leg = [f'Generated {n}' for n in [0, 64, 128]]
plt.legend(waves + add_leg)
plt.xlabel('Lag')
plt.ylabel('Power (dB)')

