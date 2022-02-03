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
    bg_ext = None
    bg = None
    targets = None
    PRF = 500.0

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
        self.bg_ext = (128, 128)
        self.PRFrange = (50, 500)
        self.step = 0

        # Hard code in the FFT_Length so we can use it in policy NN
        self.fft_len = 16384

        # Calculate size of horns to get desired beamwidths
        # This is an empirically derived rough estimate of radiation function params
        rad_cons = np.array([4, 3, 2, 1, .01])
        rad_bws = np.array([4.01, 6, 8.6, 18.33, 33])
        self.az_fac = np.interp(az_bw / 2, rad_bws, rad_cons)
        self.el_fac = np.interp(el_bw / 2, rad_bws, rad_cons)

        # Antenna array definition
        dr = c0 / 9.6e9
        self.tx_locs = np.array([(0, -dr, 0), (0, dr, 0)]).T
        self.rx_locs = np.array([(-dr, 0, 0), (dr, 0, 0), (0, 0, 0)]).T
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

        self.clipping = [-100, 200]
        self.reset()
        self.data_block = (self.nsam, self.cpi_len)
        self.MPP = c0 / 2 / self.fs
        # doppPRF = 2 * self.env.spd * np.sin(self.az_bw / 2) * (max(self.fc) + max(self.bw) / 2) / c0

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
        cfk = 1 - gaus_2d((self.nsam // 20, self.cpi_len // 10), (1, .3))
        self.cfar_kernel = cfk / np.sum(cfk)

    def max_episode_timesteps(self):
        return self.max_steps

    def states(self):
        return None

    def actions(self):
        return None

    def execute(self, actions):
        # Calculate times for this CPI based on given PRF
        self.tf = self.tf[-1] + np.arange(1, self.cpi_len + 1) * 1 / actions['radar'][0]

        # Generate positions for the platform
        self.env.genpos(self.tf)
        for t in self.targets:
            t.genpos(self.tf)

        # If we've reached the end of the data, pull out
        done = False
        self.step += 1
        if self.step >= self.max_episode_timesteps():
            done = 2

        # Update radar parameters to given params
        self.fc = actions['fc']
        self.bw = actions['bw']
        self.az_pt = actions['scan'][0]
        self.el_pt = actions['elscan'][0]
        chirps = np.zeros((self.nr, self.n_tx), dtype=np.complex128)
        for n in range(self.n_tx):
            chirps[:, n] = self.genChirp(actions['wave'][:, n], self.bw[n]) * actions['power'][n]
        fft_chirp = np.fft.fft(chirps, self.fft_len, axis=0)
        state_corr = np.fft.fftshift(db(np.fft.ifft(fft_chirp * fft_chirp.conj(), axis=0)), axes=0)
        state_corr = state_corr - np.max(state_corr, axis=0)[None, :]
        state_corr[state_corr < -100] = -100

        # Generate the CPI using chirps; generates a nsam x cpi_len x n_ants block of FFT data
        cpi = self.genCPI(fft_chirp)
        curr_cpi = np.zeros((self.nsam, self.cpi_len, self.v_ants), dtype=np.complex128)
        for idx, ap in enumerate(self.apc):
            curr_cpi[:, :, idx] = genRD(cpi[:, :, ap[0]], fft_chirp[:, ap[1]], self.nsam)

        # Ambiguity score
        amb_sc = (1 - np.linalg.norm(np.corrcoef(chirps.T) - np.eye(self.n_tx)))

        # PRF quality score
        prf_sc = abs(self.PRF - actions['radar'][0]) / 500.0

        # Check pulse detection quality
        detb_sc = 0
        if self.det_model is not None:
            id_data, blen, n_dets = self.genDetBlock(chirps, 20)
            n_close = 1 - log_loss(n_dets, self.det_model.predict(id_data.T).flatten())
            detb_sc += -n_close

        # Truth target distance score
        # Grab truth target location
        t_pos = np.array([*self.targets[0].pos(self.tf[1]), 1]) - self.env.pos(self.tf[1])
        t_vel = np.linalg.norm(
            t_pos - (np.array([*self.targets[0].pos(self.tf[0]), 1]) - self.env.pos(self.tf[0]))) \
                * actions['radar'][0]
        truth_rbin = (2 * np.linalg.norm(t_pos) / c0 - 2 * (
                self.alt / np.sin(self.dep_ang + self.el_bw / 2)) / c0) * self.fs
        ea = [np.arctan2(t_pos[1], t_pos[0]) - self.boresight_ang,
              np.arcsin(-t_pos[2] / np.linalg.norm(t_pos)) - self.dep_ang]
        dist_sc = min(np.linalg.norm([(.001 / cpudiff(ea[1], self.az_pt)), (.001 / cpudiff(ea[0], self.el_pt))]), 1)

        # MIMO beamforming using some CPI stuff
        ea = np.array([self.az_pt, self.el_pt + np.pi / 2])
        beamform = np.zeros((self.nsam, self.cpi_len), dtype=np.complex128)

        # Direction of beam to synthesize from data
        a = np.exp(1j * 2 * np.pi * np.array([self.fc[n[1]] for n in self.apc]) *
                   self.virtual_array.T.dot(np.array([np.cos(ea[0]) * np.sin(ea[1]),
                                                      np.sin(ea[0]) * np.sin(ea[1]), np.cos(ea[1])])) / c0)

        # Get clutter covariance for this CPI
        # This is used for waveform evaluation and MMSE beamforming, if selected
        Rx_thresh = np.max(db(curr_cpi[:, 0, :]), axis=1)
        clutt_ind = np.logical_and(np.median(Rx_thresh) - np.std(Rx_thresh) < Rx_thresh,
                                   Rx_thresh < np.median(Rx_thresh) + np.std(Rx_thresh))
        Rx_data = curr_cpi[clutt_ind, 0, :]
        Rx = (np.cov(Rx_data.T) + np.diag([actions['power'][n[1]] ** 2 for n in self.apc]))
        clutter_cov = np.corrcoef(Rx_data.T)

        if self.bf_type == 'mmse':
            # Generate MMSE beamformer and apply to data
            Rx_inv = np.linalg.pinv(Rx)
            U = Rx_inv.dot(a)
        elif self.bf_type == 'phased':
            # Generated phased array weights in indicated direction
            U = a
        else:
            # This weighting points the beam at 90 degrees
            U = np.ones((self.v_ants,), dtype=np.complex128)
        for tt in range(self.cpi_len):
            beamform[:, tt] = curr_cpi[:, tt, :].dot(U)

        # Put data into Range-Doppler space and window it in Doppler
        beamform = db(np.fft.fft(beamform * taylor(self.cpi_len)[None, :], axis=1))

        # Detection of targets score
        det_sc = 0
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
        det_st = binary_dilation(binary_erosion(beamform > thresh + np.std(beamform)))
        '''
        det_st = beamform > np.median(beamform) + np.std(beamform) * 3
        labels, ntargets = label(det_st)
        if ntargets > 0:
            best_mu = -np.inf
            t_rng = 0
            t_dopp = 0
            for n_targ in range(1, ntargets):
                mu = np.mean(beamform[labels[n_targ]])
                n_rng, n_dopp = np.where(labels == n_targ)
                if mu > best_mu:
                    best_mu = mu
                    t_rng = n_rng.mean()
                    # t_dopp = n_dopp.mean() % (self.cpi_len // 2) * actions['radar'][0] / (2 * self.cpi_len // 2)
            det_sc = min(1 / abs(t_rng - truth_rbin), 1)
        else:
            if 0 < truth_rbin < self.nsam:
                det_sc = -.25

        self.PRF = actions['radar'][0] + 0.0

        # Append everything to the logs, for pretty pictures later
        self.log.append([beamform, [[amb_sc, detb_sc, det_sc], [prf_sc, dist_sc, det_sc]],
                         self.tf, actions['scan'], actions['elscan'], actions['wave'], ea, U])
        # Remove anything longer than the allowed steps, for arbitrary length episodes
        if len(self.log) > self.max_steps:
            self.log.pop(0)

        # Whole state space to be split into the various agents
        full_state = {'point_angs': np.array([self.az_pt, self.el_pt]),
                      'wave_corr': state_corr, 'clutter': np.array([clutter_cov.real, clutter_cov.imag]),
                      'currwave': actions['wave'], 'currfc': self.fc, 'currbw': self.bw,
                      'platform_motion': self.env.vel(self.tf[0]),
                      'target_angs': np.array(ea)}

        return full_state, done, np.array([amb_sc + detb_sc + det_sc, prf_sc + dist_sc + det_sc])

    def reset(self, num_parallel=None):
        self.step = 0
        self.bg = wavefunction(self.bg_ext, npts=(32, 32))
        self.PRF = self.PRFrange[1] / 2
        self.tf = np.linspace(0, self.cpi_len / self.PRF, self.cpi_len)
        self.env = Platform(self.alt, self.az_bw, self.el_bw, self.dep_ang)
        self.env.genpos(self.tf)
        self.log = []

        self.targets = []
        self.targets.append(Sub(0, self.env.swath, 0, self.env.eswath))

        for t in self.targets:
            t.genpos(self.tf)

        # Environment pulse info
        max_pl = (self.env.nrange * 2 / c0 - 1 / TAC) * .99
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(max_pl * self.plp * self.fs)

        # Initial wave is pretty much a single tone at center frequency
        init_wave = np.ones((WAVEPOINTS, self.n_tx)) * .5
        init_wave[0:3, :] = .01
        init_wave[-3:, :] = .99
        return {'point_angs': np.array([0.0, 0.0]), 'wave_corr': np.zeros((self.fft_len, self.n_tx)),
                'clutter': np.array([np.eye(self.v_ants), np.zeros((self.v_ants, self.v_ants))]),
                'currwave': init_wave, 'currfc': [9.6e9 for _ in range(self.n_tx)],
                'currbw': [self.fs / 2 - 10e6 for _ in range(self.n_tx)],
                'platform_motion': self.env.vel(0),
                'target_angs': np.array([0.0, 0.0])}

    def genCPI(self, chirp):
        mx_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
        cpi_threads = int(np.ceil(self.cpi_len / self.samples))
        targ_threads = int(np.ceil(self.cpi_len / len(self.targets)))
        tpb_samples = (cpi_threads, mx_threads // cpi_threads)
        tpb_subs = (targ_threads, mx_threads // targ_threads)
        blocks_per_grid = (
            int(np.ceil(self.cpi_len / tpb_samples[0])), int(np.ceil(self.samples / tpb_samples[1])))
        sub_blocks = (int(np.ceil(self.cpi_len / tpb_subs[0])),
                      int(np.ceil(len(self.targets) / tpb_subs[1])))
        zo_t = np.zeros((len(self.tf), *self.bg[0].shape), dtype=np.complex128)

        # Get pan and tilt angles for fixed array
        mot_pos = self.el_rot(self.dep_ang, self.az_rot(self.boresight_ang, self.env.vel(self.tf)))
        pan = np.arctan2(mot_pos[1, :], mot_pos[0, :])
        el = np.arcsin(mot_pos[2, :] / np.linalg.norm(mot_pos, axis=0))
        pan_gpu = cupy.array(np.ascontiguousarray(pan), dtype=np.float64)
        el_gpu = cupy.array(np.ascontiguousarray(el), dtype=np.float64)
        p_pos = self.env.pos(self.tf)

        # Make a big box around the swath
        bx_rng = p_pos[2, 0] / np.tan(el[0])
        bx_srng = np.sqrt(bx_rng ** 2 + p_pos[2, 0] ** 2)
        bx_cent = self.az_rot(pan[0], np.array([bx_rng, 0, 0])) + p_pos[:, 0]
        bx_perp = bx_srng * np.tan(self.az_bw / 2)

        # Generate ground points using box
        gpts = np.zeros((3, self.samples))
        gpts[1, :] = np.random.uniform(-bx_perp * 2, bx_perp * 2, self.samples)
        gpts[0, :] = np.random.uniform((p_pos[2, 0] / np.tan(el[0] + self.el_bw / 2) - bx_rng) * 2,
                                       (p_pos[2, 0] / np.tan(el[0] - self.el_bw / 2) - bx_rng) * 2, self.samples)
        gpts = self.az_rot(pan[0], gpts).T
        gpts[:, 0] += bx_cent[0]
        gpts[:, 1] += bx_cent[1]
        gpts_gpu = cupy.array(gpts[:, :2], dtype=np.float64)

        # Background generation
        for t in range(len(self.tf)):
            zo_t[t, ...] = self.getBGFreqMap(self.tf[t])

        bg_gpu = cupy.array(zo_t, dtype=np.complex128)
        bg_gpu = cupy.fft.ifft2(bg_gpu, axes=(1, 2))

        # Grab sub positions and add to GPU
        sv = []
        for sub in self.targets:
            sv.append([sub(t) for t in self.tf])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)
        rd_cpu = np.zeros((self.fft_len, self.cpi_len, self.n_rx), dtype=np.complex128)

        # This is waaay easier to run in the system, at the expense of a small bit of fidelity
        rot_txlocs = self.el_rot(el[0], self.az_rot(pan[0] - self.boresight_ang, self.tx_locs))
        rot_rxlocs = self.el_rot(el[0], self.az_rot(pan[0] - self.boresight_ang, self.rx_locs))

        # Data generation loop
        for rx in range(self.n_rx):
            posrx_gpu = cupy.array(np.ascontiguousarray(p_pos + rot_rxlocs[:, rx][:, None]),
                                   dtype=np.float64)
            for tx in range(self.n_tx):
                data_r = cupy.zeros(self.data_block, dtype=np.float64)
                data_i = cupy.zeros(self.data_block, dtype=np.float64)
                p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc[tx],
                                             self.alt / np.sin(self.dep_ang + self.el_bw / 2) / c0,
                                             self.fs, self.dep_ang, self.bg_ext[0], self.bg_ext[1],
                                             self.boresight_ang, self.az_fac, self.el_fac]),
                                   dtype=np.float64)
                chirp_gpu = cupy.array(np.tile(chirp[:, tx], (self.cpi_len, 1)).T, dtype=np.complex128)
                postx_gpu = cupy.array(np.ascontiguousarray(p_pos + rot_txlocs[:, tx][:, None]),
                                       dtype=np.float64)
                genRangeProfile[blocks_per_grid, tpb_samples](posrx_gpu, postx_gpu, gpts_gpu, pan_gpu, el_gpu,
                                                              bg_gpu, data_r, data_i, p_gpu)
                cupy.cuda.Device().synchronize()
                genSubProfile[sub_blocks, tpb_subs](posrx_gpu, postx_gpu, sub_pos, pan_gpu, el_gpu, bg_gpu,
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
        thread_ratio = len(self.targets) / block_len
        sub_blocks = (int(mx_threads / len(self.targets)),
                      int(mx_threads / block_len))
        sv = []
        for sub in self.targets:
            sv.append([sub(t) for t in self.tf[block_init:block_len + block_init]])
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
        det_spread = cupy.zeros((n_dets,), dtype=int)
        rd_cpu = np.random.randn(self.det_sz, pt_s) + 1j * np.random.randn(self.det_sz, pt_s)
        fft_len = findPowerOf2(self.det_sz)
        for tx in range(self.n_tx):
            data_r = cupy.zeros((self.det_sz, pt_s), dtype=np.float64)
            data_i = cupy.zeros((self.det_sz, pt_s), dtype=np.float64)
            p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc[tx],
                                         self.alt / np.sin(self.dep_ang + self.el_bw / 2) / c0,
                                         self.fs, self.dep_ang, block_init * self.fs / c0, self.det_sz,
                                         self.boresight_ang, self.az_fac, self.el_fac]),
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

    def genChirp(self, py, bandwidth):
        return genPulse(np.linspace(0, 1, len(py)), py, self.nr, self.nr / self.fs, 0, bandwidth)


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


class Platform(object):
    swath = 0
    eswath = 0
    nrange = 0
    frange = 0
    spd = 0
    max_pl = 0
    wave = None

    def __init__(self, h_agl, az_bw, el_bw, dep_ang):
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

        self.plog = np.array([[0.0, -self.gmrange, self.h_agl]]).T
        self.vlog = np.array([[1.0, 0.001, 0]]).T
        self.tlog = np.array([-1e-9])
        self.avec = lambda t: np.array([np.random.normal(5, 1, len(t)), np.random.normal(0, 1.2, len(t)), np.zeros(len(t))])

    def genpos(self, t):
        # This function assumes the times given are in the fuuuutuuuure
        dt = t[1] - t[0]
        nvs = self.vlog[:, -1][:, None] + np.cumsum(self.avec(t) * dt, axis=0)
        nps = self.plog[:, -1][:, None] + np.cumsum(nvs, axis=0)

        # Concatenate with old logs
        self.plog = np.concatenate((self.plog, nps), axis=1)
        self.vlog = np.concatenate((self.vlog, nvs), axis=1)
        self.tlog = np.concatenate((self.tlog, t))

        if len(self.tlog) > 512 and self.tlog[-1] - self.tlog[0] > .1:
            # Decimate everything since we don't need such high fidelity for the interpolation
            new_tlog = np.arange(self.tlog[0], self.tlog[-1], .1)
            self.plog = np.array([np.interp(new_tlog, self.tlog, self.plog[0, :]),
                                  np.interp(new_tlog, self.tlog, self.plog[1, :]),
                                  np.interp(new_tlog, self.tlog, self.plog[2, :])])
            self.vlog = np.array([np.interp(new_tlog, self.tlog, self.vlog[0, :]),
                                  np.interp(new_tlog, self.tlog, self.vlog[1, :]),
                                  np.interp(new_tlog, self.tlog, self.vlog[2, :])])
            self.tlog = new_tlog
        self.spd = np.linalg.norm(nvs, axis=0).mean()
        return nps

    def pos(self, t):
        return np.array([np.interp(t, self.tlog, self.plog[0, :]),
                         np.interp(t, self.tlog, self.plog[1, :]),
                         np.interp(t, self.tlog, self.plog[2, :])])

    def vel(self, t):
        return np.array([np.interp(t, self.tlog, self.vlog[0, :]),
                         np.interp(t, self.tlog, self.vlog[1, :]),
                         np.interp(t, self.tlog, self.vlog[2, :])])

    def getAntennaBeamLocation(self, t, az_ang, el_ang):
        fpos = self.pos(t)
        eshift = np.cos(az_ang) * self.h_agl / np.tan(el_ang)
        nshift = np.sin(az_ang) * self.h_agl / np.tan(el_ang)
        return fpos[0] + eshift, fpos[1] + nshift, self.gbwidth, (self.gfrange - self.gnrange) / 2


class Sub(object):
    surf = None

    def __init__(self, min_x, max_x, min_y, max_y):
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)
        self.plog = np.array([[np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]]).T
        self.vlog = np.array([[np.random.rand() - .5, np.random.rand() - .5]]).T
        self.tlog = np.array([-1e-9])
        self.avec = lambda t: np.array(
            [np.random.normal(0, MAX_ALFA_ACCEL / 3, len(t)), np.random.normal(0, MAX_ALFA_ACCEL / 3, len(t))])

        self.surf = lambda t: np.ones_like(t)

    def __call__(self, t):
        return self.surf(t), *self.pos(t)

    def genpos(self, t):
        # This function assumes the times given are in the fuuuutuuuure
        dt = t[1] - t[0]
        av = self.avec(t)
        av_norm = np.linalg.norm(av, axis=0)
        av[:, av_norm > MAX_ALFA_ACCEL] = \
            av[:, av_norm > MAX_ALFA_ACCEL] * MAX_ALFA_ACCEL / av_norm[av_norm > MAX_ALFA_ACCEL]
        nvs = self.vlog[:, -1][:, None] + np.cumsum(self.avec(t) * dt, axis=0)
        nvs_norm = np.linalg.norm(nvs, axis=0)
        nvs[:, nvs_norm > MAX_ALFA_SPEED] = \
            nvs[:, nvs_norm > MAX_ALFA_SPEED] * MAX_ALFA_SPEED / nvs_norm[nvs_norm > MAX_ALFA_SPEED]
        nps = self.plog[:, -1][:, None] + np.cumsum(nvs, axis=0)

        # Concatenate with old logs
        self.plog = np.concatenate((self.plog, nps), axis=1)
        self.vlog = np.concatenate((self.vlog, nvs), axis=1)
        self.tlog = np.concatenate((self.tlog, t))

        if len(self.tlog) > 512 and self.tlog[-1] - self.tlog[0] > .1:
            # Decimate everything since we don't need such high fidelity for the interpolation
            new_tlog = np.arange(self.tlog[0], self.tlog[-1], .1)
            self.plog = np.array([np.interp(new_tlog, self.tlog, self.plog[0, :]),
                                  np.interp(new_tlog, self.tlog, self.plog[1, :])])
            self.vlog = np.array([np.interp(new_tlog, self.tlog, self.vlog[0, :]),
                                  np.interp(new_tlog, self.tlog, self.vlog[1, :])])
            self.tlog = new_tlog
        return nps

    def pos(self, t):
        return np.array([np.interp(t, self.tlog, self.plog[0, :]),
                         np.interp(t, self.tlog, self.plog[1, :])])

    def vel(self, t):
        return np.array([np.interp(t, self.tlog, self.vlog[0, :]),
                         np.interp(t, self.tlog, self.vlog[1, :])])


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
