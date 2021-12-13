from tensorforce import Environment
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import fftconvolve
from scipy.linalg import convolution_matrix
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
    return int(2 ** (np.ceil(np.log2(x))))


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

MAX_ALFA_ACCEL = 0.35185185185185186
MAX_ALFA_SPEED = 21.1111111111111111
THREADS_PER_BLOCK = (16, 16)
EP_LEN_S = 60
WAVEPOINTS = 100


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
        self.plp = 1
        self.fc = 9.6e9
        self.samples = 200000
        self.fs = fs / 8
        self.bw = 120e6
        self.az_pt = np.pi / 2
        self.el_pt = 45 * DTR
        self.az_lims = (0, np.pi)
        self.el_lims = (30 * DTR, 70 * DTR)

        # Add in extra phase centers
        self.antenna_locs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0)]
        self.antenna_locs = np.array(self.antenna_locs).T
        self.el_rot = lambda el, loc: np.array([[1, 0, 0],
                                [0, np.cos(el), -np.sin(el)],
                                [0, np.sin(el), np.cos(el)]]).dot(loc)
        self.az_rot = lambda az, loc: np.array([[np.cos(az), -np.sin(az), 0],
                                                [np.sin(az), np.cos(az), 0],
                                                [0, 0, 1]]).dot(loc)
        self.n_ants = self.antenna_locs.shape[1]

        # Generate CFAR kernel
        self.cfar_kernel = np.ones((40, 11))
        self.cfar_kernel[17:24, 3:8] = 0
        self.cfar_kernel = self.cfar_kernel / np.sum(self.cfar_kernel)
        self.det_targets = []
        self.reset()

        self.data_block = (self.nsam, self.cpi_len)
        self.MPP = c0 / 2 / self.fs
        self.maxPRF = min(c0 / (self.env.nrange + self.nsam * self.MPP), 500.0)
        try:
            self.det_model = keras.models.load_model('./id_model')
        except OSError:
            self.det_model = None
        self.ave = 0
        self.std = 0

    def states(self):
        return dict(cpi=dict(type='float', shape=(self.nsam, self.cpi_len)),
                    currscan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    currelscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]),
                    currwave=dict(type='float', shape=(WAVEPOINTS, self.n_ants), min_value=0, max_value=1))

    def actions(self):
        return dict(wave=dict(type='float', shape=(WAVEPOINTS, self.n_ants), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=100, max_value=self.maxPRF),
                    scan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    elscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.arange(1, self.cpi_len + 1) * 1 / actions['radar'][0]
        # We've reached the end of the data, pull out
        done = False if self.tf[-1] < EP_LEN_S else 2
        self.tf[self.tf >= EP_LEN_S] = EP_LEN_S - .01
        motion = (abs(self.az_pt - actions['scan'][0]) + abs(self.el_pt - actions['elscan'][0])) ** .1 - .5
        self.az_pt = actions['scan'][0]
        self.el_pt = actions['elscan'][0]
        chirps = np.zeros((self.nr, self.n_ants), dtype=np.complex128)
        for n in range(self.n_ants):
            chirps[:, n] = self.genChirp(actions['wave'][:, n], self.bw)
        fft_chirp = np.fft.fft(chirps, self.fft_len, axis=0)

        # Generate the CPI using chirps; generates a nsam x cpi_len x n_ants block of FFT data
        cpi = self.genCPI(fft_chirp, self.tf, self.az_pt, self.el_pt)
        state = np.zeros((self.nsam, self.cpi_len, self.n_ants))
        for n in range(self.n_ants):
            rda = genRDAMap(cpi[:, :, n], fft_chirp[:, n], self.nsam)
            st = abs(rda)
            if np.std(st) > 0:
                st = (st - np.mean(st)) / np.std(st)
            else:
                pass
            state[:, :, n] = st
        reward = 0
        t_score = 0

        # Ambiguity score
        spike_cov = np.zeros((self.n_ants, self.n_ants))
        for rx in range(self.n_ants):
            for tx in range(self.n_ants):
                rc_chirp = db(np.fft.ifft(fft_chirp[:, rx] * fft_chirp[:, tx].conj().T))
                rc_chirp = rc_chirp / np.linalg.norm(rc_chirp)
                spike_cov[rx, tx] = rc_chirp.max()
        amb_sc = 1 - np.linalg.norm(spike_cov - np.eye(self.n_ants))
        reward += amb_sc

        # PRF shift score
        prf_sc = 0

        # Detectability score
        det_sc = 0
        '''
        if self.det_model is not None:
            net_sz = self.det_model.layers[0].input_shape[0][1]
            det_chirp = (np.random.rand(max(self.nr, net_sz)) - .5 + 1j *
                         (np.random.rand(max(self.nr, net_sz)) - .5)) / 100
            det_chirp[:self.nr] += chirps[:, 0]
            wdd = WignerVilleDistribution(det_chirp).run()[0]
            det_sc = self.det_model.predict(wdd.reshape((1, *wdd.shape)))[0][1]
        else:
            det_sc = 0
        reward += det_sc
        '''

        # Movement score
        # Find targets using basic CFAR
        # First, remove sea spikes using a simple averaging filter
        for ant in range(self.n_ants):
            det_state = fftconvolve(state[:, :, ant], np.ones((5, 5)) / 25.0, mode='same')
            thresh = fftconvolve(det_state, self.cfar_kernel, mode='same')
            det_targets = det_state > thresh + 5
            det_targets[:, :3] = 0
            det_targets[:, -3:] = 0
            t_score = 0
            labels, ntargets = label(det_targets)
            # Run through targets and add any that we might need
            for lab in range(ntargets):
                rngs, vels = np.where(labels == lab)
                rng = c0 / 2 * (rngs.mean() / self.fs + 2 * self.alt / c0 / np.sin(self.el_pt + self.el_bw / 2))
                vel = vels.mean() * actions['radar'][0] / self.fc * c0
                if len(self.det_targets) == 0:
                    self.det_targets.append(Target(rng, vel, self.tf[-1]))
                else:
                    for targ in self.det_targets:
                        if targ(rng, vel, self.tf[-1]):
                            break
        # Disassociate targets
        for targ in self.det_targets:
            if targ.dissac(self.tf[-1]):
                self.det_targets.remove(targ)
        if len(self.det_targets) > 0:
            for targ in self.det_targets:
                t_score += 1 + .5 / abs(targ.rng - c0 / 2 * (2 * self.alt / c0 / np.sin(self.el_pt)))
                prf_sc += 1 / abs(targ.vel)
        t_score += motion
        reward += t_score
        reward += prf_sc

        self.log.append([state, [amb_sc, det_sc, t_score, prf_sc],
                         self.tf, actions['scan'], actions['elscan'], actions['wave'], len(self.det_targets)])

        full_state = {'cpi': state[:, :, 0], 'currscan': [self.az_pt], 'currelscan': [self.el_pt],
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
        init_wave = np.ones((WAVEPOINTS, self.n_ants)) * .5
        return {'cpi': np.zeros((self.nsam, self.cpi_len)),
                'currscan': [self.az_pt], 'currelscan': [self.el_pt],
                'currwave': init_wave}

    def genCPI(self, chirp, tf, az_pt, el_pt):
        blocks_per_grid = (
            int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])), int(np.ceil(self.samples / THREADS_PER_BLOCK[1])))
        sub_blocks = (int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])),
                      int(np.ceil(len(self.env.targets) / THREADS_PER_BLOCK[1])))
        az_pan = az_pt * np.ones((self.cpi_len,))
        el_pan = el_pt * np.ones((self.cpi_len,))
        pan_gpu = cupy.array(np.ascontiguousarray(az_pan), dtype=np.float64)
        el_gpu = cupy.array(np.ascontiguousarray(el_pan), dtype=np.float64)
        p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc,
                                     self.alt / np.sin(self.el_pt + self.el_bw / 2) / c0,
                                     self.fs, self.dep_ang, self.env.eswath, self.env.swath,
                                     np.random.randint(1, 15)]), dtype=np.float64)
        gx = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        gy = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        sv = []
        for sub in self.env.targets:
            sv.append([sub(t) for t in tf])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)
        times = cupy.array(np.ascontiguousarray(tf), dtype=np.float64)
        alocs = self.el_rot(el_pt, self.az_rot(az_pt, self.antenna_locs))
        rd_cpu = np.zeros((self.fft_len, self.cpi_len, self.n_ants), dtype=np.complex128)
        for rx in range(self.n_ants):
            posrx_gpu = cupy.array(np.ascontiguousarray(self.env.pos(tf) + alocs[:, rx][:, None]), dtype=np.float64)
            data_r = cupy.zeros(self.data_block, dtype=np.float64)
            data_i = cupy.zeros(self.data_block, dtype=np.float64)
            for tx in range(self.n_ants):
                chirp_gpu = cupy.array(np.tile(chirp[:, tx], (self.cpi_len, 1)).T, dtype=np.complex128)
                postx_gpu = cupy.array(np.ascontiguousarray(self.env.pos(tf) + alocs[:, tx][:, None]), dtype=np.float64)
                genRangeProfile[blocks_per_grid, THREADS_PER_BLOCK](posrx_gpu, postx_gpu, gx, gy, pan_gpu, el_gpu,
                                                                    times, data_r, data_i, p_gpu)
                cupy.cuda.Device().synchronize()
                genSubProfile[sub_blocks, THREADS_PER_BLOCK](posrx_gpu, postx_gpu, sub_pos, pan_gpu, el_gpu,
                                                             data_r, data_i, p_gpu)
                cupy.cuda.Device().synchronize()
                data = data_r + 1j * data_i
                ret_data = cupy.fft.fft(data, self.fft_len, axis=0) * chirp_gpu
                cupy.cuda.Device().synchronize()
                rd_cpu[:, :, rx] += ret_data.get()

        del ret_data
        del pan_gpu
        del el_gpu
        del p_gpu
        del data_r
        del data_i
        del times
        del posrx_gpu
        del postx_gpu
        del chirp_gpu
        del sub_pos
        del gx
        del gy
        cupy.get_default_memory_pool().free_all_blocks()

        return rd_cpu

    def genChirp(self, py, bandwidth):
        return genPulse(np.linspace(0, 1, len(py)), py, self.nr, self.nr / self.fs, 0, bandwidth)


def genRDAMap(cpi, chirp, nsam):
    twin = taylor(cpi.shape[0])
    win_gpu = cupy.array(np.tile(twin, (cpi.shape[1], 1)).T, dtype=np.complex128)
    chirp_gpu = cupy.array(np.tile(chirp, (cpi.shape[1], 1)).T, dtype=np.complex128)
    cpi_gpu = cupy.array(cpi, dtype=np.complex128)
    rda_gpu = cupy.fft.fft(cupy.fft.ifft(cpi_gpu * (chirp_gpu * win_gpu).conj(), axis=0)[:nsam, :], axis=1)
    cupy.cuda.Device().synchronize()
    rda = rda_gpu.get() * taylor(cpi.shape[1])[None, :]

    del win_gpu
    del chirp_gpu
    del rda_gpu
    del cpi_gpu
    cupy.get_default_memory_pool().free_all_blocks()
    return rda


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
        for n in range(2):
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


class Target(object):
    def __init__(self, rng, vel, t):
        self.rng = rng
        self.vel = vel
        self.track = [[rng, vel, t]]
        self.last_assoc = t

    def __call__(self, rng, vel, t):
        # First, check to see what the expected new position is
        if abs(self.rng + (t - self.last_assoc) * self.vel - rng) < 5:
            if abs(self.vel - vel) < 10:
                # Passes the checks, associate it
                self.rng = rng
                self.vel = vel
                self.track.append([rng, vel, t])
                self.last_assoc = t
                return True
        # Failed!
        return False

    def dissac(self, t):
        # Send a disassociate signal if it has been too long between detections
        if t - self.last_assoc > .1:
            return True
        return False


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
def wavefunction(x, y, t, ws):
    nwaves = int(ws % 8) + 1
    wave = 1j * 0
    for n in range(nwaves):
        wdir = cmath.exp(1j * ws)
        wave += ws / 10.0 * cmath.exp(
            1j * (wdir.real / (ws * 10) * x + wdir.imag / (ws * 10) * y + 2 * np.pi * ws / 4.0 * t))
    return wave
    # return .25 * cmath.exp(1j * (.70710678 / 100 * x + .70710678 / 100 * y + 2 * np.pi * .1 * t)) + \
    #   .47 * cmath.exp(1j * (0.9486833 / 40 * x + 0.31622777 / 40 * y + 2 * np.pi * 10 * t))


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
    'void(float64[:, :], float64[:, :], float64[:], float64[:], float64[:], float64[:], ' +
    'float64[:], float64[:, :], float64[:, :], float64[:])')
def genRangeProfile(pathrx, pathtx, gx, gy, pan, el, t, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gx.size:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gx[samp_point] * (params[6] * 2) - params[6] / 2
        ty = gy[samp_point] * (params[7] * 2) - params[7] / 2

        wp = wavefunction(tx, ty, t[tt], params[8])

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = tx - pathtx[0, tt]
        s_ty = ty - pathtx[1, tt]
        s_tz = wp.real - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        s_rx = tx - pathrx[0, tt]
        s_ry = ty - pathrx[1, tt]
        s_rz = wp.real - pathrx[2, tt]
        rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
        rng = (rngtx + rngrx)
        rng_bin = (rng * 2 / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            el_tx = math.asin(-s_tz / rngtx)
            az_tx = math.atan2(s_tx, s_ty)
            eldiff = diff(el_tx, el[tt])
            azdiff = diff(az_tx, pan[tt])
            tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            el_rx = math.asin(-s_rz / rngrx)
            az_rx = math.atan2(s_rx, s_ry)
            eldiff = diff(el_rx, el[tt])
            azdiff = diff(az_rx, pan[tt])
            rx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            rx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            att = tx_elpat * tx_azpat * rx_elpat * rx_azpat
            acc_val = wp.imag * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit('void(float64[:, :], float64[:, :], float64[:, :, :], float64[:], float64[:], ' +
          'float64[:, :], float64[:, :], float64[:])')
def genSubProfile(pathrx, pathtx, subs, pan, el, pd_r, pd_i, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        sub_x = subs[subnum, tt, 1]
        sub_y = subs[subnum, tt, 2]
        spow = subs[subnum, tt, 0]
        sub_z = 20 if spow > 0 else 0

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        for n in range(-3, 3):
            s_tx = sub_x - pathtx[0, tt]
            s_ty = sub_y - pathtx[1, tt]
            s_tz = sub_z - pathtx[2, tt]
            rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4] * n
            s_rx = sub_x - pathrx[0, tt]
            s_ry = sub_y - pathrx[1, tt]
            s_rz = sub_z - pathrx[2, tt]
            rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4] * n
            rng = (rngtx + rngrx)
            rng_bin = (rng / c0 - 2 * params[3]) * params[4]
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if n_samples > but > 0:
                el_tx = math.asin(-s_tz / rngtx)
                az_tx = math.atan2(s_tx, s_ty)
                eldiff = diff(el_tx, el[tt])
                azdiff = diff(az_tx, pan[tt])
                tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
                tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
                el_rx = math.asin(-s_rz / rngrx)
                az_rx = math.atan2(s_rx, s_ry)
                eldiff = diff(el_rx, el[tt])
                azdiff = diff(az_rx, pan[tt])
                rx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
                rx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
                att = tx_elpat * tx_azpat * rx_elpat * rx_azpat
                acc_val = spow * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
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
        cpi_data, done, reward = agent.execute({'wave': np.linspace(0, 1, WAVEPOINTS),
                                                'radar': [100], 'scan': [np.pi / 2]})
        # print(agent.az_pt / DTR)

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
