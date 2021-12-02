from tensorforce import Environment
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import fftconvolve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from tftb.processing import WignerVilleDistribution
from scipy.ndimage import binary_dilation, binary_erosion, label
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from numba import cuda, njit
from tensorflow import keras
import cmath
import math
import cupy as cupy

from celluloid import Camera


def db(x):
    ret = abs(x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2**(np.ceil(np.log2(x))))


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

MAX_ALFA_ACCEL = 0.35185185185185186
MAX_ALFA_SPEED = 21.1111111111111111
THREADS_PER_BLOCK = (16, 16)
EP_LEN_S = 60
WAVEPOINTS = 20


# Container class for radar data and parameters
def genPulse(phase_x, phase_y, nr, t0, fc, bandw):
    phase = fc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * t0 / nr))


class SinglePulseBackground(Environment):
    env = None
    tf = None
    nsam = 0
    nr = 0
    fft_len = 0
    pts = None
    pt_amp = None
    data_block = None
    az_pan = None
    log = None

    def __init__(self):
        super().__init__()
        self.cpi_len = 128
        self.az_bw = 9 * DTR
        self.el_bw = 12 * DTR
        self.dep_ang = 45 * DTR
        self.alt = 1524
        self.plp = .4
        self.fc = 9.6e9
        self.samples = 400000
        self.fs = fs / 4
        self.bw = 240e6
        self.az_pt = np.pi / 2
        self.el_pt = 45 * DTR
        self.az_lims = (np.pi / 4, 3 * np.pi / 4)
        self.el_lims = (30 * DTR, 70 * DTR)
        self.cfar_kernel = np.ones((40, 11))
        self.cfar_kernel[17:24, 3:8] = 0
        self.cfar_kernel = self.cfar_kernel / np.sum(self.cfar_kernel)
        self.reset()
        self.data_block = (self.nsam, self.cpi_len)
        self.MPP = c0 / 2 / self.fs
        self.maxPRF = min(c0 / (self.env.nrange + self.nsam * self.MPP), 500.0)
        self.det_model = keras.models.load_model('./id_model')
        self.ave = 0
        self.std = 0

    def states(self):
        return dict(cpi=dict(type='float', shape=(self.nsam, self.cpi_len)),
                    currscan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    currelscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]),
                    currwave=dict(type='float', shape=(WAVEPOINTS,), min_value=0, max_value=1))

    def actions(self):
        return dict(wave=dict(type='float', shape=(WAVEPOINTS,), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=10, max_value=self.maxPRF),
                    scan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    elscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.arange(1, self.cpi_len + 1) * 1 / actions['radar'][0]
        # We've reached the end of the data, pull out
        done = False if self.tf[-1] < EP_LEN_S else 2
        self.tf[self.tf >= EP_LEN_S] = EP_LEN_S - .01
        motion = (abs(self.az_pt - actions['scan'][0]) + abs(self.el_pt - actions['elscan'][0]))**.1 - .5
        self.az_pt = actions['scan'][0]
        self.el_pt = actions['elscan'][0]
        waveform = np.ones(WAVEPOINTS + 2)
        waveform[0] = 0
        waveform[1:-1] = actions['wave']
        chirp = self.genChirp(waveform, self.bw)
        fft_chirp = np.fft.fft(chirp, self.fft_len)
        win_chirp = np.fft.ifft(fft_chirp * taylor(self.fft_len))
        cpi = self.genCPI(fft_chirp, self.tf, self.az_pt, self.el_pt)
        state = abs(cpi)
        state = (state - np.mean(state)) / np.std(state)
        reward = 0

        # Ambiguity function score
        amb, _, _ = ambiguity(chirp, win_chirp, actions['radar'][0], 150)
        thumb = np.zeros(amb.shape)
        thumb[amb.shape[0] // 2, amb.shape[1] // 2] = 1
        amb_sc = 1 / np.linalg.norm(amb / np.linalg.norm(amb) - thumb)**4
        reward += amb_sc

        # Detectability score
        #net_sz = self.det_model.layers[0].input_shape[0][1]
        #det_chirp = np.random.rand(max(self.nr, net_sz)) + 1j * np.random.rand(max(self.nr, net_sz))
        #det_chirp[:self.nr] += chirp
        #wdd = WignerVilleDistribution(det_chirp).run()[0]
        det_sc = 0 #self.det_model.predict(wdd.reshape((1, *wdd.shape)))[0][1]
        reward += det_sc

        # Movement score
        # Find targets using basic CFAR
        thresh = fftconvolve(state, self.cfar_kernel, mode='same')
        det_targets = state > thresh + 4
        det_targets[:, :3] = 0
        det_targets[:, -3:] = 0
        if np.any(det_targets):
            t_score = state[det_targets].mean() / state.max() + 1 / abs(np.where(det_targets)[0].mean() - det_targets.shape[0] // 2)
        else:
            t_score = motion

        self.log.append([det_targets, [amb_sc, det_sc, t_score],
                         self.tf, actions['scan'], actions['elscan'], waveform])

        full_state = {'cpi': state, 'currscan': [self.az_pt], 'currelscan': [self.el_pt],
                      'currwave': actions['wave']}

        return full_state, done, reward

    def reset(self, num_parallel=None):
        self.tf = np.linspace(0, self.cpi_len / 500.0, self.cpi_len)
        self.env = SimEnv(self.alt, self.az_bw, self.el_bw, self.dep_ang, f_ts=EP_LEN_S)
        self.log = []
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + self.env.max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(self.env.max_pl * self.plp * self.fs)
        self.fft_len = findPowerOf2(self.nsam + self.nr)
        init_wave = np.ones(WAVEPOINTS) * .5
        return {'cpi': np.zeros((self.nsam, self.cpi_len)),
                'currscan': [self.az_pt], 'currelscan': [self.el_pt],
                'currwave': init_wave}

    def genCPI(self, chirp, tf, az_pt, el_pt):
        twin = taylor(self.fft_len)
        win_gpu = cupy.array(np.tile(twin, (self.cpi_len, 1)).T, dtype=np.complex128)
        chirp_gpu = cupy.array(np.tile(chirp, (self.cpi_len, 1)).T, dtype=np.complex128)

        blocks_per_grid = (
            int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])), int(np.ceil(self.samples / THREADS_PER_BLOCK[1])))
        sub_blocks = (int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])),
                      int(np.ceil(len(self.env.targets) / THREADS_PER_BLOCK[1])))
        pos_gpu = cupy.array(np.ascontiguousarray(self.env.pos(tf)), dtype=np.float64)
        az_pan = az_pt * np.ones((self.cpi_len,))
        el_pan = el_pt * np.ones((self.cpi_len,))
        pan_gpu = cupy.array(np.ascontiguousarray(az_pan), dtype=np.float64)
        el_gpu = cupy.array(np.ascontiguousarray(el_pan), dtype=np.float64)
        p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc,
                                     self.alt / np.sin(self.el_pt + self.el_bw / 2) / c0,
                                     self.fs, self.dep_ang, self.env.eswath, self.env.swath]), dtype=np.float64)
        data_r = cupy.zeros(self.data_block, dtype=np.float64)
        data_i = cupy.zeros(self.data_block, dtype=np.float64)
        gx = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        gy = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        sv = []
        for sub in self.env.targets:
            sv.append([sub(t) for t in tf])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)
        times = cupy.array(np.ascontiguousarray(tf), dtype=np.float64)

        genRangeProfile[blocks_per_grid, THREADS_PER_BLOCK](pos_gpu, gx, gy, pan_gpu, el_gpu,
                                                            times, data_r, data_i, p_gpu)
        cupy.cuda.Device().synchronize()

        genSubProfile[sub_blocks, THREADS_PER_BLOCK](pos_gpu, sub_pos, pan_gpu, el_gpu, data_r, data_i, p_gpu)
        cupy.cuda.Device().synchronize()

        data = data_r + 1j * data_i

        ret_data = cupy.fft.fft(
            cupy.fft.ifft(
                (cupy.fft.fft(data, self.fft_len, axis=0) * chirp_gpu * (chirp_gpu * win_gpu).conj()),
                axis=0)[:self.nsam, :], axis=1)
        cupy.cuda.Device().synchronize()
        rd_cpu = ret_data.get() * taylor(self.cpi_len)[None, :]

        del ret_data
        del pan_gpu
        del el_gpu
        del p_gpu
        del data_r
        del data_i
        del times
        del pos_gpu
        del win_gpu
        del chirp_gpu
        del sub_pos
        del gx
        del gy
        cupy.get_default_memory_pool().free_all_blocks()

        return rd_cpu

    def genChirp(self, py, bandwidth):
        return genPulse(np.linspace(0, 1, len(py)), py, self.nr, self.nr / self.fs, 0, bandwidth)


class SimEnv(object):
    swath = 0
    eswath = 0
    nrange = 0
    frange = 0
    pos = None
    spd = 0
    max_pl = 0
    wave = None

    def __init__(self, h_agl, az_bw, el_bw, dep_ang, f_ts=11):
        nrange = h_agl / np.sin(dep_ang + el_bw / 2)
        frange = h_agl / np.sin(dep_ang - el_bw / 2)
        gnrange = h_agl / np.tan(dep_ang + el_bw / 2)
        gfrange = h_agl / np.tan(dep_ang - el_bw / 2)
        gmrange = (gfrange + gnrange) / 2
        blen = np.tan(az_bw / 2) * gmrange
        self.swath = (gfrange - gnrange)
        self.eswath = blen * 2
        self.nrange = nrange
        self.frange = frange
        self.mrange = (frange + nrange) / 2
        self.gnrange = gnrange
        self.gmrange = gmrange
        self.gfrange = gfrange
        self.gbwidth = blen
        self.h_agl = h_agl
        self.targets = []
        self.f_ts = f_ts
        self.genFlightPath()
        for n in range(1):
            self.targets.append(Sub(0, self.eswath,
                                    0, self.swath, f_ts))

    def genFlightPath(self):
        # We start assuming that the bottom left corner of scene is (0, 0, 0) ENU
        # Platform motion (assuming 100 Hz signal)
        npts = int(50 * self.f_ts / 10)
        tt = np.linspace(0, self.f_ts, npts)
        e = gaussian_filter(
            np.linspace(self.eswath / 4, self.eswath - self.eswath / 4, npts) + (np.random.rand(npts) - .5) * 3, 3)
        n = gaussian_filter(-self.gnrange + (np.random.rand(npts) - .5) * 3, 3)
        u = gaussian_filter(self.h_agl + (np.random.rand(npts) - .5) * 3, 3)
        e = np.zeros(npts) + self.eswath / 2
        n = np.zeros(npts) - self.gmrange
        u = self.h_agl + np.zeros(npts)
        re = UnivariateSpline(tt, e, s=.7, k=3)
        rn = UnivariateSpline(tt, n, s=.7, k=3)
        ru = UnivariateSpline(tt, u, s=.7, k=3)
        self.pos = lambda t: np.array([re(t), rn(t), ru(t)])
        self.spd = np.linalg.norm(np.gradient(self.pos(np.linspace(0, self.f_ts, int(100 * self.f_ts / 10))),
                                              axis=1), axis=0).mean()

        # Environment pulse info
        self.max_pl = (self.nrange * 2 / c0 - 1 / TAC) * .99

    def getAntennaBeamLocation(self, t, az_ang, el_ang):
        fpos = self.pos(t)
        eshift = np.cos(az_ang) * self.h_agl / np.tan(el_ang)
        nshift = np.sin(az_ang) * self.h_agl / np.tan(el_ang)
        return fpos[0] + eshift, fpos[1] + nshift, self.gbwidth, (self.gfrange - self.gnrange) / 2


class Sub(object):
    pos = None
    vel = None
    surf = None

    def __init__(self, min_x, max_x, min_y, max_y, f_ts=11):
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)
        self.f_ts = f_ts

        # Plot out next f_ts of movement for reproducability
        self.plotRoute(f_ts, min_x, max_x, min_y, max_y)

    def __call__(self, t):
        return self.surf(t), *self.pos(t)

    def reset(self):
        # Plot out next f_ts of movement for reproducability
        self.plotRoute(self.f_ts, self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1])

    def plotRoute(self, f_ts, min_x, max_x, min_y, max_y):
        loc = np.array([np.random.rand() * (max_x - min_x) + min_x, np.random.rand() * (max_y - min_y) + min_y])
        vels = np.random.rand(2) - .5
        pos = np.zeros((2, f_ts * 100))
        is_surfaced = np.zeros((f_ts * 100))
        surfaced = False
        t0 = .01
        for idx in np.arange(f_ts * 100):
            if idx % 25 == 0:
                if np.random.rand() < .5:
                    surfaced = not surfaced
            acc_dir = (np.random.rand(2) - .5)
            accels = acc_dir / np.linalg.norm(acc_dir) * MAX_ALFA_ACCEL * np.random.rand()
            vels += accels
            if surfaced:
                vels *= .99
            if np.linalg.norm(vels) > MAX_ALFA_SPEED:
                vels = vels / np.linalg.norm(vels) * MAX_ALFA_SPEED
            floc = loc + vels * t0
            if min_x > floc[0] or floc[0] >= max_x:
                vels[0] = -vels[0]
            if min_y > floc[1] or floc[1] >= max_y:
                vels[1] = -vels[1]
            loc += vels * t0
            pos[:, idx] = loc + 0.0
            is_surfaced[idx] = surfaced * 50
        pn = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), pos[0, :])
        pe = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), pos[1, :])
        surf = interp1d(np.linspace(0, f_ts, f_ts * 100), is_surfaced, fill_value=0)
        self.pos = lambda t: np.array([pe(t), pn(t)])
        vn = np.gradient(pos[0, :]) / 100
        ve = np.gradient(pos[1, :]) / 100
        self.vel = lambda t: np.array([ve(t), vn(t)])
        self.surf = surf


@cuda.jit(device=True)
def wavefunction(x, y, t):
    t = 0
    return .25 * cmath.exp(1j * (.70710678 / 100 * x + .70710678 / 100 * y + 2 * np.pi * .1 * t)) + \
        .47 * cmath.exp(1j * (0.9486833 / 40 * x + 0.31622777 / 40 * y + 2 * np.pi * 10 * t))


def wave_cpu(x, y, t):
    return .25 * np.exp(1j * (.70710678 / 100 * x + .70710678 / 100 * y + 2 * np.pi * .1 * t)) + \
        .47 * np.exp(1j * (0.9486833 / 40 * x + 0.31622777 / 40 * y + 2 * np.pi * 10 * t))


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def cpudiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(
    'void(float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :], float64[:, :], float64[:])')
def genRangeProfile(path, gx, gy, pan, el, t, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gx.size:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gx[samp_point] * (params[6] * 2) - params[6] / 2
        ty = gy[samp_point] * (params[7] * 2) - params[7] / 2

        wp = wavefunction(tx, ty, t[tt])

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_x = tx - path[0, tt]
        s_y = ty - path[1, tt]
        s_z = wp.real - path[2, tt]
        rng = math.sqrt(s_x * s_x + s_y * s_y + s_z * s_z)
        rng_bin = (rng * 2 / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            el_tx = math.asin(-s_z / rng)
            az_tx = math.atan2(s_x, s_y)
            eldiff = diff(el_tx, el[tt])
            azdiff = diff(az_tx, pan[tt])
            tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            att = tx_elpat * tx_azpat
            acc_val = wp.imag * att * cmath.exp(-1j * wavenumber * rng * 2) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit('void(float64[:, :], float64[:, :, :], float64[:], float64[:], float64[:, :], float64[:, :], float64[:])')
def genSubProfile(path, subs, pan, el, pd_r, pd_i, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = subs[subnum, tt, 1]
        ty = subs[subnum, tt, 2]
        spow = subs[subnum, tt, 0]
        tz = 20 if spow > 0 else 0

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        for n in range(-3, 3):
            s_x = tx - path[0, tt]
            s_y = ty - path[1, tt]
            s_z = tz - path[2, tt]
            rng = math.sqrt(s_x * s_x + s_y * s_y + s_z * s_z) + c0 / params[4] * n
            rng_bin = (rng * 2 / c0 - 2 * params[3]) * params[4]
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if n_samples > but > 0:
                el_tx = math.asin(-s_z / rng)
                az_tx = math.atan2(s_x, s_y)
                eldiff = diff(el_tx, el[tt])
                azdiff = diff(az_tx, pan[tt])
                tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
                tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
                att = tx_elpat * tx_azpat
                acc_val = spow * att * cmath.exp(-1j * wavenumber * rng * 2) * 1 / (rng * rng)
                cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
                cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


def ellipse(x, y, a, b, ang):
    t = np.linspace(0, 2 * np.pi, 100)
    ell = np.array([a * np.cos(t), b * np.sin(t)])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    fin_ell = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        fin_ell[:, i] = np.dot(rot, ell[:, i])
    return fin_ell + np.array([x, y])[:, None]


@njit
def apply_shift(ray: np.ndarray, freq_shift: np.float64, samp_rate: np.float64) -> np.ndarray:
    # apply frequency shift
    precache = 2j * np.pi * freq_shift / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray


def ambiguity(s1, s2, prf, dopp_bins, mag=True):
    fdopp = np.linspace(-prf / 2, prf / 2, dopp_bins)
    fft_sz = findPowerOf2(len(s1)) * 2
    s1f = np.fft.fft(s1, fft_sz).conj().T
    shift_grid = np.zeros((len(s2), dopp_bins), dtype=np.complex64)
    for n in range(dopp_bins):
        shift_grid[:, n] = apply_shift(s2, fdopp[n], fs)
    s2f = np.fft.fft(shift_grid, n=fft_sz, axis=0)
    A = np.fft.fftshift(np.fft.ifft(s2f * s1f[:, None], axis=0, n=fft_sz * 2),
                        axes=0)[fft_sz - dopp_bins // 2: fft_sz + dopp_bins // 2]
    if mag:
        return abs(A / abs(A).max()) ** 2, fdopp, np.linspace(-len(s1) / 2 / fs, len(s1) / 2 / fs, len(s1))
    else:
        return A / abs(A).max(), fdopp, np.linspace(-dopp_bins / 2 * fs / c0, dopp_bins / 2 * fs / c0, dopp_bins)


if __name__ == '__main__':
    agent = SinglePulseBackground()
    test = SimEnv(agent.alt, agent.az_bw, agent.el_bw, agent.dep_ang)

    pulse = agent.genChirp(np.linspace(0, 1, WAVEPOINTS), agent.bw)

    for cpi in tqdm(range(200)):
        cpi_data, done, reward = agent.execute({'wave': np.linspace(0, 1, WAVEPOINTS), 'radar': [100], 'scan': [np.pi / 2]})
        #print(agent.az_pt / DTR)

    logs = agent.log
    skips = 2
    cols = ['blue', 'red', 'orange', 'yellow', 'green']
    fig, axes = plt.subplots(2)
    axes[1].set_xlim([-agent.env.eswath / 2, agent.env.eswath * 1.5])
    axes[1].set_ylim([-agent.env.gnrange, agent.env.gfrange])
    camera = Camera(fig)
    for log_idx, l in tqdm(enumerate(logs[::skips])):
        fpos = agent.env.pos(l[2][0])
        main_beam = ellipse(*(list(agent.env.getAntennaBeamLocation(l[2][0], l[3][0], l[4][0])) + [l[3][0]]))
        axes[1].plot(main_beam[0, :], main_beam[1, :], 'gray')
        axes[1].scatter(fpos[0], fpos[1], marker='*', c='blue')
        for idx, s in enumerate(agent.env.targets):
            pos = s.pos(l[2][0])
            amp = s.surf(l[2][0]) + 1
            zx = wave_cpu(pos[0], pos[1], l[2][0])
            plt_rng = np.linalg.norm(fpos - np.array([pos[0], pos[1], zx.real]))
            axes[1].scatter(pos[0], pos[1], s=amp, c=cols[idx])
            axes[1].text(pos[0], pos[1], f'{plt_rng:.2f}', c='black')
        axes[1].legend([f'{l[2][0]:.6f}-{l[2][-1]:.6f}'])
        axes[0].imshow(np.fft.fftshift(l[0], axes=1), origin='lower',
                       extent=[0, agent.cpi_len, agent.env.nrange, agent.env.frange])
        axes[0].axis('tight')
        camera.snap()

    animation = camera.animate()

    tt = np.linspace(0, 10, 1000)
    locs = test.pos(tt)

    fig = plt.figure('Plane')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])

    wdd = WignerVilleDistribution(pulse)
    wdd.run()
    wdd.plot(show_tf=True, kind='contour')
    dbw, dt0 = agent.detect(pulse)
    print('Params\t\tBW(MHz)\t\tt0(us)')
    print(f'Detect\t\t{dbw * agent.fs / 1e6:.2f}\t\t{dt0 / agent.fs * 1e6:.2f}')
    print(f'Truth \t\t{agent.bw / 1e6:.2f}\t\t{agent.nr / agent.fs * 1e6:.2f}')
    print(f'Diffs \t\t{abs(agent.bw - dbw * agent.fs) / 1e6:.2f}\t\t{abs(agent.nr - dt0) * 1e6 / agent.fs:.2f}')

    amb = ambiguity(pulse, pulse, 100, 30)

    plt.figure('Ambiguity')
    plt.imshow(amb[0])

    # scores = np.array([l[1] for l in logs])

    # plt.figure('Score breakdown')


    '''fig_wave = plt.figure('Waves')
    gx, gy = np.meshgrid(np.linspace(-agent.env.eswath / 2, agent.env.eswath * 1.5, 1000),
                         np.linspace(-agent.env.swath / 2, agent.env.swath * 1.5, 1000))
    ax = plt.axes()
    cam_wave = Camera(fig_wave)
    for log_idx, l in tqdm(enumerate(logs[::skips])):
        wav_vals = wave_cpu(gx, gy, l[2][0])
        ax.imshow(wav_vals.real)
        cam_wave.snap()

    anim_wave = cam_wave.animate()'''

