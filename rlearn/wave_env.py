from tensorforce import Agent, Environment
from radar import SimPlatform, SimChannel, Antenna
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import convolve2d
from scipy.ndimage import rotate, gaussian_filter, label
from tftb.processing import WignerVilleDistribution
from scipy.ndimage.filters import median_filter
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from numba import vectorize, float64, int32, complex128, jit

from rawparser import loadASHFile, loadASIFile
from useful_lib import findPowerOf2, db

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


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

    def __init__(self):
        super().__init__()
        self.cpi_len = 64
        self.az_bw = 6 * DTR
        self.el_bw = 8 * DTR
        self.dep_ang = 55 * DTR
        self.plp = .25
        self.fc = 9e9
        self.samples = 2000
        self.fs = fs / 4
        self.bw = 225e6
        self.tf = np.linspace(0, self.cpi_len / 150.0, self.cpi_len)
        self.env = SimEnv(1000, self.az_bw, self.el_bw, self.dep_ang)
        self.log = []
        self.reset()

    def states(self):
        return dict(type='float', shape=(self.nsam, self.cpi_len), min_value=-300)

    def actions(self):
        return dict(wave=dict(type='float', shape=(10,), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=100, max_value=5000))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.linspace(0, 1 / actions['radar'][0], self.cpi_len)
        chirp = self.genChirp(actions['wave'], self.bw)
        state = self.genCPI(chirp)
        reward = 0

        done = False

        # Sidelobe score
        fftchirp = np.fft.fft(chirp, self.fft_len * 2)
        rc_chirp = db(np.fft.ifft(fftchirp * fftchirp.conj().T))
        rc_chirp = (rc_chirp - rc_chirp.mean()) / rc_chirp.std()
        sll = np.max(rc_chirp)
        reward += sll

        # Detectability score
        # dbw, dt0 = self.detect(chirp)
        # det_sc = 1 / abs(dbw - (actions['wave'].max() - actions['wave'].min()) / self.fs) + 1 / (abs(dt0 - self.nr) + .01)
        # reward += det_sc

        #self.log.append([db(state), [sll, det_sc, ntargets]])

        return db(state), done, reward

    def reset(self, num_parallel=None):
        self.runBackground()
        return np.zeros((self.nsam, self.cpi_len))

    def runBackground(self):
        ntargs = len(self.env.targets)
        asamples = np.zeros((self.samples + ntargs,))
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + self.env.max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(self.env.max_pl * self.plp * self.fs)
        self.fft_len = findPowerOf2(self.nsam + self.nr)
        esamples = np.random.rand(self.samples) * self.env.eswath
        nsamples = np.random.rand(self.samples) * self.env.swath
        usamples, asamples[ntargs:] = self.env(esamples, nsamples)
        pts = np.ones((3, self.samples + ntargs, self.cpi_len))
        for p in range(self.cpi_len):
            for idx, s in enumerate(self.env.targets):
                asamples[idx], (pts[0, idx, p], pts[1, idx, p]) = s(self.tf[p])
                pts[2, idx, p] = 2
        pts[0, ntargs:, :] *= esamples[:, None]
        pts[1, ntargs:, :] *= nsamples[:, None]
        pts[2, ntargs:, :] *= usamples[:, None]
        self.pts = pts
        self.pt_amp = np.repeat(asamples.reshape((-1, 1)), self.cpi_len, axis=1)

    def genChirp(self, py, bandwidth):
        return genPulse(np.linspace(0, 1, len(py)), py, self.nr, self.nr / self.fs, 0, bandwidth)

    def genCPI(self, chirp):
        self.runBackground()
        fft_len = findPowerOf2(self.nsam + self.nr)
        MPP = c0 / 2 / self.fs
        range_bins = self.env.nrange + np.arange(self.nsam) * MPP
        pvecs = self.env.pos(self.tf)[:, None, :] - self.pts
        prng = np.linalg.norm(pvecs, axis=0)
        pbin = getBin(prng, self.env.nrange, self.fs)
        rp_vals = getRangePoint(pvecs[0, ...], pvecs[1, ...], pvecs[2, ...], prng, self.pt_amp, self.fc,
                                self.dep_ang, self.az_bw, self.el_bw)
        rp = np.zeros((len(range_bins), self.cpi_len), dtype=np.complex128)
        rp = genRangeProfile(rp, rp_vals, pbin, self.nsam, self.cpi_len)
        rp = np.fft.fft(rp, n=fft_len, axis=0)
        ffchirp = np.fft.fft(chirp, n=fft_len)
        return np.fft.fft(np.fft.ifft(rp * ffchirp[:, None] * ffchirp.conj().T[:, None], axis=0)[:self.nsam, :], axis=1)

    def detect(self, signal):
        wd = WignerVilleDistribution(signal)
        wd.run()
        tfr = wd.tfr
        bmask = tfr > np.mean(tfr) + np.std(tfr)
        bmask = binary_erosion(binary_dilation(bmask, iterations=1), iterations=1)
        bandwidth = sum(wd.freqs[np.sum(bmask, axis=1) > 0]) / sum(wd.freqs) / 2
        tfs = wd.ts[np.sum(bmask, axis=0) > 0]
        t0 = (tfs.max() - tfs.min())
        return bandwidth, t0


class SimEnv(object):
    swath = 0
    eswath = 0
    nrange = 0
    frange = 0
    pos = None
    max_pl = 0
    ng = None
    eg = None
    ug = None
    pts = None

    def __init__(self, h_agl, az_bw, el_bw, dep_ang):
        nrange = h_agl / np.sin(dep_ang + el_bw / 2)
        frange = h_agl / np.sin(dep_ang - el_bw / 2)
        gnrange = h_agl / np.tan(dep_ang + el_bw / 2)
        gfrange = h_agl / np.tan(dep_ang - el_bw / 2)
        gmrange = (gfrange + gnrange) / 2
        blen = np.tan(az_bw / 2) * gmrange
        self.swath = (gfrange - gnrange) * 2
        self.eswath = blen * 4
        self.nrange = nrange
        self.frange = frange
        self.gnrange = gnrange
        self.h_agl = h_agl
        self.targets = []
        self.genBackground()
        for n in range(np.random.randint(1, 5)):
            self.targets.append(Sub(0, self.eswath, 0, self.swath))

    def genBackground(self):
        # We start assuming that the bottom left corner of scene is (0, 0, 0) ENU
        # Platform motion (assuming 100 Hz signal, so this is 10 sec)
        npts = 50
        tt = np.linspace(0, 10, npts)
        e = gaussian_filter(np.linspace(self.eswath / 4, self.eswath - self.eswath / 4, npts) + (np.random.rand(npts) - .5) * 3, 3)
        n = gaussian_filter(-self.gnrange + (np.random.rand(npts) - .5) * 3, 3)
        u = gaussian_filter(self.h_agl + (np.random.rand(npts) - .5) * 3, 3)
        re = UnivariateSpline(tt, e, s=.7, k=3)
        rn = UnivariateSpline(tt, n, s=.7, k=3)
        ru = UnivariateSpline(tt, u, s=.7, k=3)
        self.pos = lambda t: np.array([re(t), rn(t), ru(t)])

        # Environment pulse info
        self.max_pl = (self.nrange * 2 / c0 - 1 / TAC) * .99

        self.ng, self.eg = np.meshgrid(np.arange(0, self.swath + 1), np.arange(0, self.eswath + 1))

        # Since this is water, theoretically, we can randomly assign heights
        self.ug = gaussian_filter(np.random.rand(*self.eg.shape), 3)

        # Randomly assign amplitudes for power
        self.pts = gaussian_filter(np.random.rand(*self.eg.shape), 3)

    def __call__(self, e, n):
        x0 = np.round(e).astype(int)
        y0 = np.round(n).astype(int)
        xdiff = e - x0
        ydiff = n - y0
        x1 = x0 + np.sign(xdiff).astype(int)
        y1 = y0 + np.sign(ydiff).astype(int)
        amp = self.pts[x1, y1] * xdiff * ydiff + self.pts[x1, y0] * xdiff * (1 - ydiff) + self.pts[x0, y1] * \
              (1 - xdiff) * ydiff + self.pts[x0, y0] * (1 - xdiff) * (1 - ydiff)
        hght = self.ug[x1, y1] * xdiff * ydiff + self.ug[x1, y0] * xdiff * (1 - ydiff) + self.ug[x0, y1] * \
               (1 - xdiff) * ydiff + self.ug[x0, y0] * (1 - xdiff) * (1 - ydiff)
        return hght, amp


class Sub(object):

    def __init__(self, min_x, max_x, min_y, max_y, f_ts=10):
        # Generate sub location and time (random)
        self.loc = [np.random.rand() * (max_x - min_x) + min_x, np.random.rand() * (max_y - min_y) + min_y]
        self.vel = [(np.random.rand() - .5) * 8, (np.random.rand() - .5) * 8]
        self.fs = 2e9
        surface_t = np.random.rand() * f_ts
        self.t_s = (surface_t, surface_t + np.random.rand() * min(3, f_ts - surface_t))
        self.tk = 0
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)

    def __call__(self, t):
        pow = 20 if self.t_s[0] <= t < self.t_s[1] else 0
        # Update the location
        self.loc = [self.loc[0] + self.vel[0] * (t - self.tk), self.loc[1] + self.vel[1] * (t - self.tk)]
        # Ping pong inside of swath
        if self.xbounds[1] > self.loc[0] < self.xbounds[0]:
            self.vel[0] = -self.vel[0]
        if self.ybounds[1] > self.loc[1] < self.ybounds[0]:
            self.vel[1] = -self.vel[1]
        self.tk = t
        return pow, self.loc


@vectorize([int32(float64, float64, float64)])
def getBin(x, rng, _fs):
    n = (x * 2 / c0 - 2 * rng / c0) * _fs
    return int(n) if n - int(n) < .5 else int(n+1)


@vectorize([complex128(float64, float64, float64, float64, float64, float64, float64, float64, float64)])
def getRangePoint(x, y, z, prng, pt_amp, fc, da, az_bw, el_bw):
    el = np.arcsin(z / prng) - da
    az = np.arctan2(y, x) + np.pi / 2
    bp_att = abs(np.sin(np.pi / az_bw * az) / (np.pi / az_bw * az)) * \
             abs(np.sin(np.pi / el_bw * el) / (np.pi / el_bw * el))
    return pt_amp * bp_att * np.exp(-1j * 2 * np.pi * fc / c0 * prng * 2) / (prng * 2)


@jit
def genRangeProfile(rp, rp_vals, pbin, nsam, cpi_len):
    for rbin in range(nsam):
        for pulse in range(cpi_len):
            rp[rbin, pulse] += sum(rp_vals[pbin[:, pulse] == rbin, pulse])
    return rp


if __name__ == '__main__':
    test = SimEnv(1000, 5 * DTR, 8 * DTR, 55 * DTR)

    agent = SinglePulseBackground()
    pulse = agent.genChirp(np.linspace(0, 1, 10), agent.bw)
    pulse = pulse# + np.random.rand(len(pulse)) * .5 + 2j * np.random.rand(len(pulse)) * np.pi

    for cpi in range(10):
        cpi_data, done, reward = agent.execute({'wave': np.linspace(0, 1, 10), 'radar': [1000]})

        plt.figure('Generated CPI {}'.format(cpi))
        plt.imshow(cpi_data)
        plt.axis('tight')

    esamples = np.random.rand(4000) * test.eswath
    nsamples = np.random.rand(4000) * test.swath
    usamples, asamples = test(esamples, nsamples)

    plt.figure('Initial')
    plt.subplot(1, 2, 1)
    plt.imshow(test.pts)
    plt.subplot(1, 2, 2)
    plt.scatter(esamples, nsamples, c=asamples)

    tt = np.linspace(0, 10, 1000)
    locs = test.pos(tt)

    fig = plt.figure('Plane')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])

    fig = plt.figure('Scene')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])
    ax.plot_wireframe(test.eg, test.ng, test.ug)

    wdd = WignerVilleDistribution(pulse)
    wdd.run()
    wdd.plot(show_tf=True, kind='contour')
    dbw, dt0 = agent.detect(pulse)
    print('Params\t\tBW(MHz)\t\tt0(us)')
    print(f'Detect\t\t{dbw * agent.fs / 1e6:.2f}\t\t{dt0 / agent.fs * 1e6:.2f}')
    print(f'Truth \t\t{agent.bw / 1e6:.2f}\t\t{agent.nr / agent.fs * 1e6:.2f}')
    print(f'Diffs \t\t{abs(agent.bw - dbw * agent.fs) / 1e6:.2f}\t\t{abs(agent.nr - dt0) * 1e6 / agent.fs:.2f}')

    fig = plt.figure('Subs')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])
    ax.plot_wireframe(test.eg, test.ng, test.ug)
    for sub in test.targets:
        spos = np.array([sub(t)[1] for t in np.linspace(0, 10, 1000)])
        spow = [sub(t)[0] for t in np.linspace(0, 10, 1000)]
        ax.plot(spos[:, 0], spos[:, 1], spow)
