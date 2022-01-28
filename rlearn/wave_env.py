import cupyx.scipy.signal
from tensorforce import Environment
import numpy as np
from music import MUSIC
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import gamma as gam_func
from scipy.special import comb as NchooseK
from scipy.linalg import convolution_matrix
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from itertools import combinations_with_replacement, permutations, combinations, product
from tftb.processing import WignerVilleDistribution
from scipy.ndimage import binary_dilation, binary_erosion, label
from DFJeff import MUSICSinglePoint as music
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.metrics import log_loss
from numba import cuda, njit
from tensorflow import keras
import cmath
import math
import cupy as cupy
from cuda_kernels import genSubProfile, genRangeProfile, getDetectionCheck
from numba.core.errors import NumbaPerformanceWarning
import warnings

from celluloid import Camera

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def db(x):
    ret = abs(x)
    ret[ret < 1e-15] = 1e-15
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2 ** (np.ceil(np.log2(x))))


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

MAX_ALFA_ACCEL = 0.35185185185185186
MAX_ALFA_SPEED = 2.577
EP_LEN_S = 30
WAVEPOINTS = 100


def multilateration(rngs, va_pos, x0=None):
    def error(x, c, r):
        return sum([(np.linalg.norm([x[0] - c[i][0], x[1] - c[i][1], 1 - c[i][2]]) - r[i]) ** 2 for i in range(len(c))])

    # get initial guess of point location
    x0 = [0.0, 0.0] if x0 is None else x0
    # optimize distance from signal origin to border of spheres
    return minimize(error, x0, args=(va_pos, rngs), method='Nelder-Mead').x


def gaus_2d(size, sigma, angle=0, height=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    a = np.cos(angle) ** 2 / (2 * sigma[0] ** 2) + np.sin(angle) ** 2 / (2 * sigma[1] ** 2)
    b = -np.sin(2 * angle) / (4 * sigma[0] ** 2) + np.sin(2 * angle) / (4 * sigma[1] ** 2)
    c = np.sin(angle) ** 2 / (2 * sigma[0] ** 2) + np.cos(angle) ** 2 / (2 * sigma[1] ** 2)
    f = height * np.exp(-(a * x ** 2 + 2 * b * x * y + c * y ** 2))
    return f


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
    virtual_array = None
    det_model = None
    par_model = None

    def __init__(self, max_timesteps=128, cpi_len=64, az_bw=24, el_bw=18, dep_ang=45, boresight_ang=90, altitude=1524,
                 plp=.5, env_samples=100000, fs_decimation=8, az_lim=90, el_lim=10, beamform_type='mmse'):
        super().__init__()
        self.cpi_len = cpi_len
        self.az_bw = az_bw * DTR
        self.el_bw = el_bw * DTR
        self.dep_ang = dep_ang * DTR
        self.boresight_ang = boresight_ang * DTR
        self.alt = altitude
        self.plp = plp
        self.samples = env_samples
        self.fs = fs / fs_decimation
        self.az_pt = 0
        self.el_pt = 0
        self.az_lims = (-az_lim * DTR, az_lim * DTR)
        self.el_lims = (-el_lim * DTR, el_lim * DTR)
        self.curr_cpi = None
        self.max_steps = max_timesteps
        self.bf_type = beamform_type
        self.step = 0

        # Antenna array definition
        dr = c0 / 9.6e9
        self.tx_locs = np.array([(0, -dr / 2, 0), (0, dr / 2, 0)]).T
        self.rx_locs = np.array([(-dr, 0, 0), (dr, 0, 0), (0, dr, 0), (0, -dr, 0)]).T
        self.el_rot = lambda el, loc: np.array([[1, 0, 0],
                                                [0, np.cos(el), -np.sin(el)],
                                                [0, np.sin(el), np.cos(el)]]).dot(loc)
        self.az_rot = lambda az, loc: np.array([[np.cos(az), -np.sin(az), 0],
                                                [np.sin(az), np.cos(az), 0],
                                                [0, 0, 1]]).dot(loc)
        self.n_tx = self.tx_locs.shape[1]
        self.n_rx = self.rx_locs.shape[1]

        # Setup center freqs and bandwidths
        self.fc = [9.6e9 for _ in range(self.n_tx)]
        self.bw = [self.fs / 2 - 10e6 for _ in range(self.n_tx)]

        # Setup for virtual array
        self.v_ants = self.n_tx * self.n_rx
        apc = []
        va = np.zeros((3, self.v_ants))
        for n in range(self.n_rx):
            for m in range(self.n_tx):
                va[:, n * self.n_tx + m] = self.rx_locs[:, n] + self.tx_locs[:, m]
                apc.append([n, m])
        self.virtual_array = self.el_rot(self.dep_ang, self.az_rot(self.boresight_ang - np.pi / 2, va))
        self.apc = apc

        self.clipping = [0, 200]
        self.reset()
        self.data_block = (self.nsam, self.cpi_len)
        self.MPP = c0 / 2 / self.fs
        self.maxPRF = min(c0 / (self.env.nrange + self.nsam * self.MPP), 500.0)
        try:
            self.det_model = keras.models.load_model('./id_model')
            self.det_sz = self.det_model.layers[0].input_shape[0][1]
        except OSError:
            self.det_model = None
        try:
            self.par_model = keras.models.load_model('./par_model')
        except OSError:
            self.par_model = None

        # CFAR kernel
        # Trying an inverted gaussian
        cfk = 1 - gaus_2d((self.nsam // 15, self.cpi_len // 15), (1, .3))
        self.cfar_kernel = cfk / np.sum(cfk)

    def max_episode_timesteps(self):
        return self.max_steps

    def states(self):
        return dict(cpi=dict(type='float', shape=(self.nsam, self.cpi_len), min_value=-300, max_value=100),
                    currscan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    currelscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]),
                    currwave=dict(type='float', shape=(WAVEPOINTS, self.n_tx), min_value=0, max_value=1),
                    currfc=dict(type='float', shape=(self.n_tx,), min_value=8e9, max_value=12e9),
                    currbw=dict(type='float', shape=(self.n_tx,), min_value=10e6, max_value=self.fs / 2 - 5e6))

    def actions(self):
        return dict(wave=dict(type='float', shape=(WAVEPOINTS, self.n_tx), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=100, max_value=self.maxPRF),
                    scan=dict(type='float', shape=(1,), min_value=self.az_lims[0], max_value=self.az_lims[1]),
                    elscan=dict(type='float', shape=(1,), min_value=self.el_lims[0], max_value=self.el_lims[1]),
                    fc=dict(type='float', shape=(self.n_tx,), min_value=8e9, max_value=12e9),
                    bw=dict(type='float', shape=(self.n_tx,), min_value=10e6, max_value=self.fs / 2 - 5e6))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.arange(1, self.cpi_len + 1) * 1 / actions['radar'][0]
        # We've reached the end of the data, pull out
        done = False if self.tf[-1] < EP_LEN_S else 2
        self.step += 1
        if self.step >= self.max_episode_timesteps():
            done = 2
        self.tf[self.tf >= EP_LEN_S] = EP_LEN_S - .01
        self.fc = actions['fc']
        self.bw = actions['bw']
        mid_tf = self.tf[len(self.tf) // 2]
        motion = np.linalg.norm([self.az_pt - actions['scan'][0], self.el_pt - actions['elscan'][0]])**2
        self.az_pt = actions['scan'][0]
        self.el_pt = actions['elscan'][0]
        chirps = np.zeros((self.nr, self.n_tx), dtype=np.complex128)
        for n in range(self.n_tx):
            chirps[:, n] = self.genChirp(actions['wave'][:, n], self.bw[n]) * actions['power'][n]
        fft_chirp = np.fft.fft(chirps, self.fft_len, axis=0)

        # Generate the CPI using chirps; generates a nsam x cpi_len x n_ants block of FFT data
        cpi = self.genCPI(fft_chirp)
        curr_cpi = np.zeros((self.nsam, self.cpi_len, self.v_ants), dtype=np.complex128)
        for idx, ap in enumerate(self.apc):
            curr_cpi[:, :, idx] = genRD(cpi[:, :, ap[0]], fft_chirp[:, ap[1]], self.nsam)

        # Ambiguity score
        amb_sc = (1 - np.linalg.norm(np.corrcoef(chirps.T) - np.eye(self.n_tx)))

        # Azimuth and elevation score
        dist_sc = 0

        # Detectability score
        det_sc = motion

        # PRF quality score
        clutter_bw = 2 * self.env.spd * np.sin(self.az_bw / 2) * (max(self.fc) + max(self.bw)) / c0

        # Check pulse detection quality
        detb_sc = 0
        if self.det_model is not None:
            id_data, blen, n_dets = self.genDetBlock(chirps, 20)
            n_close = 1 - log_loss(n_dets, self.det_model.predict(id_data.T).flatten())
            detb_sc += -n_close

        # Find truth sub direction for pulse power
        t_pos = np.array([*self.env.targets[0].pos(mid_tf), 1]) - self.env.pos(mid_tf)
        ea = [np.arctan2(t_pos[1], t_pos[0]) - self.boresight_ang,
              np.arcsin(-t_pos[2] / np.linalg.norm(t_pos)) - self.dep_ang]
        dist_sc += np.linalg.norm([(.0001 / abs(ea[1] - self.az_pt)), (.0001 / abs(ea[0] - self.el_pt))])

        # MIMO beamforming using some CPI stuff
        # if len(ea) == 0:
        ea = np.array([self.az_pt, self.el_pt + np.pi / 2])
        beamform = np.zeros((self.nsam, self.cpi_len), dtype=np.complex128)

        # Direction of beam to synthesize from data
        a = np.exp(1j * 2 * np.pi * np.array([self.fc[n[1]] for n in self.apc]) *
                   self.virtual_array.T.dot(np.array([np.cos(ea[0]) * np.sin(ea[1]),
                                                      np.sin(ea[0]) * np.sin(ea[1]), np.cos(ea[1])])) / c0)

        if self.bf_type == 'mmse':
            # Generate MMSE beamformer and apply to data
            # Focus mostly on clutter so as to avoid nulling out targets
            Rx_thresh = np.max(db(curr_cpi[:, 0, :]), axis=1)
            window = np.where(Rx_thresh == Rx_thresh.max())[0][0]
            min_win = window - 10
            max_win = window + 10
            guard_sz = 3
            while min_win < 0 and min_win < window - guard_sz:
                min_win += 1
            while max_win > curr_cpi.shape[0] and max_win > window + guard_sz:
                max_win -= 1
            Rx_data = np.concatenate(
                (curr_cpi[min_win:window - guard_sz, 0, :], curr_cpi[window + guard_sz:max_win, 0, :]))
            Rx = (np.cov(Rx_data.T) + np.diag([actions['power'][n[1]] ** 2 for n in self.apc]))
            Rx_inv = np.linalg.pinv(Rx)
            U = Rx_inv.dot(a)
        elif self.bf_type == 'phased':
            U = a
        else:
            U = np.ones((self.v_ants,), dtype=np.complex128)
        for tt in range(self.cpi_len):
            beamform[:, tt] = curr_cpi[:, tt, :].dot(U)
        beamform = db(np.fft.fft(beamform, axis=1))

        '''
        kernel = cupy.array(self.cfar_kernel, dtype=np.float64)
        bf = cupy.array(beamform, dtype=np.float64)
        tmp = cupyx.scipy.signal.fftconvolve(bf, kernel, mode='same')
        cupy.cuda.Device().synchronize()
        thresh = tmp.get()

        # Free all the GPU memory
        del bf
        del kernel
        del tmp
        cupy.get_default_memory_pool().free_all_blocks()
        det_st = beamform > thresh + np.std(beamform) * 4
        '''

        # Clipping
        beamform[beamform > self.clipping[1]] = self.clipping[1]
        beamform[beamform < self.clipping[0]] = self.clipping[0]

        # Append everything to the logs, for pretty pictures later
        self.log.append([beamform, [amb_sc, detb_sc, det_sc, dist_sc],
                         self.tf, actions['scan'], actions['elscan'], actions['wave'], ea, U])

        # Whole state space to be split into the various agents
        full_state = {'point_angs': np.array([self.az_pt, self.el_pt]),
                      'currwave': actions['wave'], 'currfc': self.fc, 'currbw': self.bw,
                      'platform_motion': np.array([self.env.pos(mid_tf),
                                                   self.env.vel(mid_tf)]),
                      'target_angs': np.array(ea)}

        return full_state, done, np.array([amb_sc + detb_sc, det_sc + dist_sc])

    def reset(self, num_parallel=None):
        self.step = 0
        self.tf = np.linspace(0, self.cpi_len / 500.0, self.cpi_len)
        self.env = SimEnv(self.alt, self.az_bw, self.el_bw, self.dep_ang, f_ts=EP_LEN_S)
        self.log = []
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + self.env.max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(self.env.max_pl * self.plp * self.fs)
        self.fft_len = findPowerOf2(self.nsam + self.nr)
        init_wave = np.ones((WAVEPOINTS, self.n_tx)) * .5
        init_wave[0:3, :] = .01
        init_wave[-3:, :] = .99
        return {'point_angs': np.array([0.0, 0.0]),
                'currwave': init_wave, 'currfc': [9.6e9 for _ in range(self.n_tx)],
                'currbw': [self.fs / 2 - 10e6 for _ in range(self.n_tx)],
                'platform_motion': np.array([self.env.pos(self.tf[0]),
                                             (self.env.pos(self.tf[
                                                               1]) - self.env.pos(
                                                 self.tf[0])) /
                                             (self.tf[1] - self.tf[0])]),
                'target_angs': np.array([0.0, 0.0])}

    def genCPI(self, chirp):
        mx_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 4
        cpi_threads = int(np.ceil(self.cpi_len / self.samples))
        samp_threads = mx_threads // cpi_threads - 1
        threads_per_block = (16, 16)
        blocks_per_grid = (
            int(np.ceil(self.cpi_len / cpi_threads)), int(np.ceil(self.samples / samp_threads)))
        sub_blocks = (int(np.ceil(mx_threads // self.cpi_len // len(self.env.targets))),
                      len(self.env.targets))
        zo_t = np.zeros((len(self.tf), *self.env.bg[0].shape), dtype=np.complex128)
        # Get pan and tilt angles for fixed array
        mot_pos = self.el_rot(self.dep_ang, self.az_rot(self.boresight_ang, self.env.vel(self.tf)))
        pan = np.arctan2(mot_pos[1, :], mot_pos[0, :])
        el = np.arcsin(mot_pos[2, :] / np.linalg.norm(mot_pos, axis=0))
        pan_gpu = cupy.array(np.ascontiguousarray(pan), dtype=np.float64)
        el_gpu = cupy.array(np.ascontiguousarray(el), dtype=np.float64)
        # Make a big box around the swath
        p_pos = self.env.pos(self.tf[0])
        bx_rng = p_pos[2] / np.tan(el[0])
        bx_srng = np.sqrt(bx_rng**2 + p_pos[2]**2)
        bx_cent = self.az_rot(pan[0], np.array([bx_rng, 0, 0])) + p_pos
        bx_perp = bx_srng * np.tan(self.az_bw / 2)
        gpts = np.zeros((3, self.samples))
        gpts[1, :] = np.random.uniform(-bx_perp * 2, bx_perp * 2, self.samples)
        gpts[0, :] = np.random.uniform((p_pos[2] / np.tan(el[0] + self.el_bw / 2) - bx_rng) * 2,
                                       (p_pos[2] / np.tan(el[0] - self.el_bw / 2) - bx_rng) * 2, self.samples)
        gpts = self.az_rot(pan[0], gpts).T
        gpts[:, 0] += bx_cent[0]
        gpts[:, 1] += bx_cent[1]
        for t in range(len(self.tf)):
            zo_t[t, ...] = self.env.getBGFreqMap(self.tf[t])
        sv = []
        bg_gpu = cupy.array(zo_t, dtype=np.complex128)
        bg_gpu = cupy.fft.ifft2(bg_gpu, axes=(1, 2))
        gpts_gpu = cupy.array(gpts[:, :2], dtype=np.float64)
        for sub in self.env.targets:
            sv.append([sub(t, fullset=True) for t in self.tf])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)
        rd_cpu = np.zeros((self.fft_len, self.cpi_len, self.n_rx), dtype=np.complex128)
        for rx in range(self.n_rx):
            posrx_gpu = cupy.array(np.ascontiguousarray(p_pos + self.rx_locs[:, rx][:, None]),
                                   dtype=np.float64)
            data_r = cupy.zeros(self.data_block, dtype=np.float64)
            data_i = cupy.zeros(self.data_block, dtype=np.float64)
            for tx in range(self.n_tx):
                p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc[tx],
                                             self.alt / np.sin(self.dep_ang + self.el_bw / 2) / c0,
                                             self.fs, self.dep_ang, self.env.bg_ext[0], self.env.bg_ext[1],
                                             self.boresight_ang]),
                                   dtype=np.float64)
                chirp_gpu = cupy.array(np.tile(chirp[:, tx], (self.cpi_len, 1)).T, dtype=np.complex128)
                postx_gpu = cupy.array(np.ascontiguousarray(p_pos + self.tx_locs[:, tx][:, None]),
                                       dtype=np.float64)
                genRangeProfile[blocks_per_grid, threads_per_block](posrx_gpu, postx_gpu, gpts_gpu, pan_gpu, el_gpu,
                                                                    bg_gpu, data_r, data_i, p_gpu)
                cupy.cuda.Device().synchronize()
                genSubProfile[sub_blocks, threads_per_block](posrx_gpu, postx_gpu, sub_pos, pan_gpu, el_gpu,
                                                             data_r, data_i, p_gpu)
                cupy.cuda.Device().synchronize()
                data = data_r + 1j * data_i
                ret_data = cupy.fft.fft(data, self.fft_len, axis=0) * chirp_gpu
                cupy.cuda.Device().synchronize()
                rd_cpu[:, :, rx] += ret_data.get()

        del ret_data
        del p_gpu
        del data_r
        del data_i
        del posrx_gpu
        del postx_gpu
        del chirp_gpu
        del sub_pos
        del gpts_gpu
        del bg_gpu
        del pan_gpu
        del el_gpu
        cupy.get_default_memory_pool().free_all_blocks()

        return rd_cpu

    def genDetBlock(self, chirp, n_dets, block_init=0):
        mx_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 4
        block_len = np.arange(self.cpi_len)[
                        np.logical_and(self.tf[block_init] <= self.tf,
                                       self.tf < self.tf[block_init] + n_dets * self.det_sz / self.fs)].max() + 1
        threads_per_block = (16, 16)  # (cpi_threads, samp_threads)
        thread_ratio = len(self.env.targets) / block_len
        sub_blocks = (int(mx_threads / len(self.env.targets)),
                      int(mx_threads / block_len))
        sv = []
        for sub in self.env.targets:
            sv.append([sub(t, fullset=True) for t in self.tf[block_init:block_len + block_init]])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)

        # Get pan and tilt angles for fixed array
        mot_pos = self.el_rot(self.dep_ang,
                              self.az_rot(self.boresight_ang, self.env.vel(self.tf[block_init:block_len + block_init])))
        pan = np.arctan2(mot_pos[1, :], mot_pos[0, :])
        el = np.arcsin(mot_pos[2, :] / np.linalg.norm(mot_pos, axis=0))
        pan_gpu = cupy.array(np.ascontiguousarray(pan), dtype=np.float64)
        el_gpu = cupy.array(np.ascontiguousarray(el), dtype=np.float64)

        # Calculate how much detection data we'll need
        pt_s = n_dets
        data_r = cupy.zeros((self.det_sz, pt_s), dtype=np.float64)
        data_i = cupy.zeros((self.det_sz, pt_s), dtype=np.float64)
        det_spread = cupy.zeros((n_dets,), dtype=int)
        rd_cpu = np.random.randn(self.det_sz, pt_s) + 1j * np.random.randn(self.det_sz, pt_s)
        fft_len = findPowerOf2(self.det_sz)
        for tx in range(self.n_tx):
            p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc[tx],
                                         self.alt / np.sin(self.dep_ang + self.el_bw / 2) / c0,
                                         self.fs, self.dep_ang, block_init * self.fs / c0, self.det_sz,
                                         self.boresight_ang]),
                               dtype=np.float64)
            chirp_gpu = cupy.array(np.tile(chirp[:, tx], (n_dets, 1)).T, dtype=np.complex128)
            postx_gpu = cupy.array(
                np.ascontiguousarray(
                    self.env.pos(self.tf[block_init:block_len + block_init]) + self.tx_locs[:, tx][:, None]),
                dtype=np.float64)
            getDetectionCheck[sub_blocks, threads_per_block](postx_gpu, sub_pos,
                                                             data_r, data_i, pan_gpu, el_gpu, det_spread, p_gpu)
            cupy.cuda.Device().synchronize()
            data = data_r + 1j * data_i
            ret_data = cupy.fft.ifft(cupy.fft.fft(data, fft_len, axis=0) * cupy.fft.fft(chirp_gpu, fft_len, axis=0),
                                     axis=0)[:self.det_sz, :]
            cupy.cuda.Device().synchronize()
            rd_cpu += ret_data.get()
        n_pulses = det_spread.get()

        del ret_data
        del p_gpu
        del data_r
        del data_i
        del postx_gpu
        del chirp_gpu
        del sub_pos
        del det_spread
        del pan_gpu
        del el_gpu
        cupy.get_default_memory_pool().free_all_blocks()

        return rd_cpu, block_len + block_init, n_pulses

    def genChirp(self, py, bandwidth):
        return genPulse(np.linspace(0, 1, len(py)), py, self.nr, self.nr / self.fs, 0, bandwidth)


def genRDAMap(cpi, chirp, nsam):
    # Gets a the Range-Doppler Map
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


def genRD(cpi, chirp, nsam):
    # Gets a range compressed Range Map
    twin = taylor(cpi.shape[0])
    win_gpu = cupy.array(np.tile(twin, (cpi.shape[1], 1)).T, dtype=np.complex128)
    chirp_gpu = cupy.array(np.tile(chirp, (cpi.shape[1], 1)).T, dtype=np.complex128)
    cpi_gpu = cupy.array(cpi, dtype=np.complex128)
    rda_gpu = cupy.fft.ifft(cpi_gpu * (chirp_gpu * win_gpu).conj(), axis=0)[:nsam, :]
    cupy.cuda.Device().synchronize()
    rda = rda_gpu.get()

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
    vel = None
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
        self.bg_ext = (self.eswath / 2, self.swath / 2)
        self.bg = wavefunction(self.bg_ext, npts=(64, 64))
        for n in range(1):
            self.targets.append(Sub(0, self.swath, 0, self.eswath,
                                    f_ts))

    def genFlightPath(self):
        # We start assuming that the bottom left corner of scene is (0, 0, 0) ENU
        # Platform motion (assuming 100 Hz signal)
        npts = int(50 * self.f_ts / 10)
        tt = np.linspace(0, self.f_ts, npts)
        e = gaussian_filter(
            np.linspace(self.eswath / 4, self.eswath - self.eswath / 4, npts) + (np.random.rand(npts) - .5) * 3, 3)
        n = gaussian_filter(-self.gnrange + (np.random.rand(npts) - .5) * 3, 3)
        u = gaussian_filter(self.h_agl + (np.random.rand(npts) - .5) * 3, 3)
        re = UnivariateSpline(tt, e, s=.7, k=3)
        rn = UnivariateSpline(tt, n, s=.7, k=3)
        ru = UnivariateSpline(tt, u, s=.7, k=3)

        self.pos = lambda t: np.array([re(t), rn(t), ru(t)])
        ttt = np.linspace(0, self.f_ts, npts * 20)
        vels = np.gradient(self.pos(ttt),
                           axis=1) / (ttt[1] - ttt[0])
        ve = UnivariateSpline(ttt, vels[0, :])
        vn = UnivariateSpline(ttt, vels[1, :])
        vu = UnivariateSpline(ttt, vels[2, :])
        self.vel = lambda t: np.array([ve(t), vn(t), vu(t)])
        self.spd = np.linalg.norm(np.gradient(self.pos(np.linspace(0, self.f_ts, int(100 * self.f_ts / 10))),
                                              axis=1), axis=0).mean() * 100 * self.f_ts / 10

        # Environment pulse info
        self.max_pl = (self.nrange * 2 / c0 - 1 / TAC) * .99

    def getAntennaBeamLocation(self, t, az_ang, el_ang):
        fpos = self.pos(t)
        eshift = np.cos(az_ang) * self.h_agl / np.tan(el_ang)
        nshift = np.sin(az_ang) * self.h_agl / np.tan(el_ang)
        return fpos[0] + eshift, fpos[1] + nshift, self.gbwidth, (self.gfrange - self.gnrange) / 2

    def getBGFreqMap(self, t):
        zo = \
            1 / np.sqrt(2) * (
                    self.bg[0] * np.exp(1j * self.bg[2] * t) + self.bg[1].conj() * np.exp(-1j * self.bg[2] * t))
        zo[0, 0] = 0
        return np.fft.fftshift(zo)

    def getBG(self, pts, t):
        zo = \
            1 / np.sqrt(2) * (
                    self.bg[0] * np.exp(1j * self.bg[2] * t) + self.bg[1].conj() * np.exp(-1j * self.bg[2] * t))
        zo[0, 0] = 0
        bg = np.fft.ifft2(np.fft.fftshift(zo)).real
        bg = bg / np.max(bg) * 2
        gx, gy = np.gradient(bg)

        x_i = pts[:, 0] % self.bg_ext[0] / self.bg_ext[0] * bg.shape[0]
        y_i = pts[:, 1] % self.bg_ext[1] / self.bg_ext[1] * bg.shape[1]
        x0 = np.round(x_i).astype(int)
        y0 = np.round(y_i).astype(int)
        x1 = x0 + np.sign(x_i - x0).astype(int)
        y1 = y0 + np.sign(y_i - y0).astype(int)

        # Make sure that the indexing values are valid
        try:
            x1[x1 >= bg.shape[0]] = bg.shape[0] - 1
            x1[x1 < 0] = 0
            x0[x0 >= bg.shape[0]] = bg.shape[0] - 1
            x0[x0 < 0] = 0

            # Similar process with the y values
            y1[y1 >= bg.shape[1]] = bg.shape[1] - 1
            y1[y1 < 0] = 0
            y0[y0 >= bg.shape[1]] = bg.shape[1] - 1
            y0[y0 < 0] = 0
        except TypeError:
            x1 = x1 if x1 < bg.shape[0] else bg.shape[0] - 1
            x0 = x0 if x0 < bg.shape[0] else bg.shape[0] - 1
            y0 = y0 if y0 < bg.shape[1] else bg.shape[1] - 1
            y1 = y1 if y1 < bg.shape[1] else bg.shape[1] - 1

        # Get differences
        xdiff = x_i - x0
        ydiff = y_i - y0
        gx1 = np.zeros((pts.shape[0], 3))
        gx1[:, 0] = 1
        gx1[:, 2] = gx[x1, y1] * xdiff * ydiff + gx[x1, y0] * xdiff * (1 - ydiff) + gx[x0, y1] * \
                    (1 - xdiff) * ydiff + gx[x0, y0] * (1 - xdiff) * (1 - ydiff)
        gx1[:, 0] = 1 / np.sqrt(1 + gx1[:, 2] ** 2)
        gx1[:, 2] = gx1[:, 2] / np.sqrt(1 + gx1[:, 2] ** 2)
        gx2 = np.zeros((pts.shape[0], 3))
        gx2[:, 2] = gy[x1, y1] * xdiff * ydiff + gy[x1, y0] * xdiff * (1 - ydiff) + gy[x0, y1] * \
                    (1 - xdiff) * ydiff + gy[x0, y0] * (1 - xdiff) * (1 - ydiff)
        gx2[:, 1] = 1 / np.sqrt(1 + gx2[:, 2] ** 2)
        gx2[:, 2] = gx2[:, 2] / np.sqrt(1 + gx2[:, 2] ** 2)
        n_dir = np.cross(gx1, gx2)
        hght = bg[x1, y1] * xdiff * ydiff + bg[x1, y0] * xdiff * (1 - ydiff) + bg[x0, y1] * \
               (1 - xdiff) * ydiff + bg[x0, y0] * (1 - xdiff) * (1 - ydiff)
        return n_dir, hght


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

    def __call__(self, t, fullset=False):
        if fullset:
            dd = self.vel(t) - self.vel(t + .01)
            dang = [dd[0] / np.linalg.norm(dd), dd[1] / np.linalg.norm(dd)]
            return self.surf(t), *self.pos(t), *dang
        else:
            return self.surf(t), *self.pos(t)

    def reset(self):
        # Plot out next f_ts of movement for reproducability
        self.plotRoute(self.f_ts, self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1])

    def plotRoute(self, f_ts, min_x, max_x, min_y, max_y):
        loc = np.array([np.random.rand() * (max_x - min_x) + min_x, np.random.rand() * (max_y - min_y) + min_y])
        vels = np.random.rand(2) - .5
        pos = np.zeros((2, f_ts * 100))
        is_surfaced = np.ones((f_ts * 100))
        surfaced = True
        t0 = .01
        for idx in np.arange(f_ts * 100):
            if idx % 25 == 0:
                if np.random.rand() < 0:
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
            is_surfaced[idx] = surfaced * 9.5
        pn = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), pos[0, :])
        pe = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), pos[1, :])
        surf = interp1d(np.linspace(0, f_ts, f_ts * 100), is_surfaced, fill_value=0)
        self.pos = lambda t: np.array([pe(t), pn(t)])
        vn = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), np.gradient(pos[0, :]) / 100)
        ve = UnivariateSpline(np.linspace(0, f_ts, f_ts * 100), np.gradient(pos[1, :]) / 100)
        self.vel = lambda t: np.array([ve(t), vn(t)])
        self.surf = surf


def wavefunction(sz, npts=(64, 64), S=2, u10=10):
    kx = np.arange(-(npts[0] // 2 - 1), npts[0] / 2 + 1) * 2 * np.pi / sz[0]
    ky = np.arange(-(npts[1] // 2 - 1), npts[1] / 2 + 1) * 2 * np.pi / sz[1]
    kkx, kky = np.meshgrid(kx, ky)
    rho = np.random.randn(*kkx.shape)
    sig = np.random.randn(*kkx.shape)
    omega = np.floor(np.sqrt(9.8 * np.sqrt(kkx ** 2 + kky ** 2)) / (2 * np.pi / 10)) * (2 * np.pi / 10)
    zo = 1 / np.sqrt(2) * (rho - 1j * sig) * np.sqrt(var_phi(kkx, kky, S, u10))
    zoc = 1 / np.sqrt(2) * (rho - 1j * sig) * np.sqrt(var_phi(-kkx, -kky, S, u10))
    return zo, zoc, omega


def Sk(k, u10=1):
    # Handles DC case
    k[k == 0] = 1e-9
    g = 9.82
    om_c = .84
    Cd10 = .00144
    ust = np.sqrt(Cd10) * u10
    km = 370
    cm = .23
    lemma = 1.7 if om_c <= 1 else 1.7 + 6 * np.log10(om_c)
    sigma = .08 * (1 + 4 * om_c ** -3)
    alph_p = .006 * om_c ** .55
    alph_m = .01 * (1 + np.log(ust / cm)) if ust <= cm else .01 * (1 + 3 * np.log(ust / cm))
    ko = g / u10 ** 2
    kp = ko * om_c ** 2
    cp = np.sqrt(g / kp)
    cc = np.sqrt((g / kp) * (1 + (k / km) ** 2))
    Lpm = np.exp(-1.25 * (kp / k) ** 2)
    gamma = np.exp(-1 / (2 * sigma ** 2) * (np.sqrt(k / kp) - 1) ** 2)
    Jp = lemma ** gamma
    Fp = Lpm * Jp * np.exp(-.3162 * om_c * (np.sqrt(k / kp) - 1))
    Fm = Lpm * Jp * np.exp(-.25 * (k / km - 1) ** 2)
    Bl = .5 * alph_p * (cp / cc) * Fp
    Bh = .5 * alph_m * (cm / cc) * Fm

    return (Bl + Bh) / k ** 3


def var_phi(kx, ky, S=2, u10=10):
    phi = np.cos(np.arctan2(ky, kx) / 2) ** (2 * S)
    gamma = Sk(np.sqrt(kx ** 2 + ky ** 2), u10) * phi * gam_func(S + 1) / gam_func(S + .5) * np.sqrt(kx ** 2 + ky ** 2)
    gamma[np.logical_and(kx == 0, ky == 0)] = 0
    return gamma


def cpudiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


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


def ambiguity(s1, s2, prf, dopp_bins, mag=True, normalize=True):
    fdopp = np.linspace(-prf / 2, prf / 2, dopp_bins)
    fft_sz = findPowerOf2(len(s1)) * 2
    s1f = np.fft.fft(s1, fft_sz).conj().T
    shift_grid = np.zeros((len(s2), dopp_bins), dtype=np.complex64)
    for n in range(dopp_bins):
        shift_grid[:, n] = apply_shift(s2, fdopp[n], fs)
    s2f = np.fft.fft(shift_grid, n=fft_sz, axis=0)
    A = np.fft.fftshift(np.fft.ifft(s2f * s1f[:, None], axis=0, n=fft_sz * 2),
                        axes=0)[fft_sz - dopp_bins // 2: fft_sz + dopp_bins // 2]
    if normalize:
        A = A / abs(A).max()
    if mag:
        return abs(A) ** 2, fdopp, np.linspace(-len(s1) / 2 / fs, len(s1) / 2 / fs, len(s1))
    else:
        return A, fdopp, np.linspace(-dopp_bins / 2 * fs / c0, dopp_bins / 2 * fs / c0, dopp_bins)
