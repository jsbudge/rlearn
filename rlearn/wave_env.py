from tensorforce import Agent, Environment
from radar import SimPlatform, SimChannel, Antenna
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import convolve2d, find_peaks
from scipy.ndimage import rotate, gaussian_filter, label
from tftb.processing import WignerVilleDistribution
from scipy.ndimage.filters import median_filter
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from numba import vectorize, float64, int32, complex128, jit
from celluloid import Camera

from rawparser import loadASHFile, loadASIFile
from useful_lib import findPowerOf2, db

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

MAX_ALFA_ACCEL = 0.35185185185185186
MAX_ALFA_SPEED = 21.1111111111111111


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
        self.az_bw = 3 * DTR
        self.el_bw = 12 * DTR
        self.dep_ang = 45 * DTR
        self.alt = 1524
        self.plp = .4
        self.fc = 9.6e9
        self.samples = 5000
        self.fs = fs / 8
        self.bw = 200e6
        self.scanRate = 10 * DTR
        self.scanDir = 1
        self.az_pt = 0
        self.reset()

    def states(self):
        return dict(type='float', shape=(self.nsam, self.cpi_len))

    def actions(self):
        return dict(wave=dict(type='float', shape=(10,), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=10, max_value=500))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.linspace(0, 1 / actions['radar'][0], self.cpi_len)
        chirp = self.genChirp(actions['wave'], self.bw)
        motScan = self.scanRate * (self.tf[-1] - self.tf[0]) * self.scanDir
        if np.pi / 4 <= motScan < np.pi * 3 / 4:
            self.az_pt += motScan
        else:
            self.scanDir *= -1
            self.az_pt += motScan * -1
        state = self.genCPI(chirp)
        reward = 0

        # We've reached the end of the data, pull out
        done = False if self.tf[-1] < 10 else True

        # Sidelobe score
        fftchirp = np.fft.fft(chirp, self.fft_len)
        rc_chirp = db(np.fft.ifft(fftchirp * (fftchirp * taylor(self.fft_len)).conj().T, self.fft_len * 8))
        rc_grad = np.gradient(rc_chirp)
        rc_chirp = (rc_chirp - rc_chirp.mean()) / rc_chirp.std()
        for ml_width in range(1, 500):
            if np.sign(rc_grad[ml_width]) != np.sign(rc_grad[ml_width - 1]):
                break
        for slidx in range(ml_width + 1, ml_width + 500):
            if np.sign(rc_grad[slidx]) != np.sign(rc_grad[slidx - 1]):
                break
        sll = 1 - rc_chirp[slidx]
        mll = 1 - (ml_width * c0 / (self.fs * 8))
        reward += sll + mll

        # Detectability score
        # dbw, dt0 = self.detect(chirp)
        # det_sc = abs(dt0 - self.nr) / self.nr + abs(dbw - self.bw / self.fs) * self.fs / self.bw
        # reward += det_sc

        self.log.append([db(state), [sll, 1 - mll], self.tf])

        return db(state), done, reward

    def reset(self, num_parallel=None):
        self.tf = np.linspace(0, self.cpi_len / 500.0, self.cpi_len)
        self.env = SimEnv(self.alt, self.az_bw, self.el_bw, self.dep_ang)
        self.log = []
        self.runBackground()
        return np.zeros((self.nsam, self.cpi_len))

    def runBackground(self):
        ntargs = len(self.env.targets)
        asamples = np.zeros((self.samples + ntargs, self.cpi_len))
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + self.env.max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(self.env.max_pl * self.plp * self.fs)
        self.fft_len = findPowerOf2(self.nsam + self.nr)
        esamples = np.random.rand(self.samples) * self.env.eswath
        nsamples = np.random.rand(self.samples) * self.env.swath
        usamples, asamples[ntargs:] = self.env(esamples, nsamples, self.tf)
        pts = np.ones((3, self.samples + ntargs, self.cpi_len))
        for p in range(self.cpi_len):
            for idx, s in enumerate(self.env.targets):
                asamples[idx], (pts[0, idx, p], pts[1, idx, p]) = s(self.tf[p])
                pts[2, idx, p] = 2
        pts[0, ntargs:, :] *= esamples[:, None]
        pts[1, ntargs:, :] *= nsamples[:, None]
        pts[2, ntargs:, :] = usamples
        self.pts = pts
        self.pt_amp = asamples

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
                                self.dep_ang, self.az_bw, self.el_bw, self.az_pt)
        rp = np.zeros((len(range_bins), self.cpi_len), dtype=np.complex128)
        rp = genRangeProfile(rp, rp_vals, pbin, self.nsam, self.cpi_len)
        rp = np.fft.fft(rp, n=fft_len, axis=0)
        ffchirp = np.fft.fft(chirp, n=fft_len)
        return np.fft.fft(np.fft.ifft(rp * ffchirp[:, None] * (ffchirp * taylor(self.fft_len)).conj().T[:, None], axis=0)[:self.nsam, :], axis=1)

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
        self.spd = np.linalg.norm(np.gradient(self.pos(np.linspace(0, 10, 100)), axis=1), axis=0).mean()

        # Environment pulse info
        self.max_pl = (self.nrange * 2 / c0 - 1 / TAC) * .99
        wa = np.random.rand(2)
        wd = np.random.rand(4) * .1
        wf = np.random.rand(2) * 2000

        self.wave = lambda x, y, t: wa[0] * np.exp(1j * (wd[0] * x + wd[1] * y + 2 * np.pi * wf[0] * t)) + \
                                    wa[1] * np.exp(1j * (wd[2] * x + wd[3] * y + 2 * np.pi * wf[1] * t))

    def __call__(self, x, y, t):
        hght = np.zeros((*x.shape, len(t)))
        amp = np.zeros_like(hght)
        for n in range(len(t)):
            wv = self.wave(x, y, t[n])
            hght[..., n] = wv.real
            amp[..., n] = wv.imag
        amp[amp < 0] = 0
        return hght, amp


class Sub(object):

    def __init__(self, min_x, max_x, min_y, max_y, f_ts=10):
        # Generate sub location and time (random)
        self.loc = np.array([np.random.rand() * (max_x - min_x) + min_x, np.random.rand() * (max_y - min_y) + min_y])
        self.pos = [self.loc + 0.0]
        self.surf = [0]
        self.vels = np.random.rand(2) - .5
        self.fs = 2e9
        self.t_s = 0
        self.surfaced = False
        self.tk = 0
        self.xbounds = (min_x, max_x)
        self.ybounds = (min_y, max_y)

    def __call__(self, t):
        pow = 0
        t0 = (t - self.tk)
        self.t_s += t0
        # Check to see if it surfaces
        if self.surfaced:
            pow = 20
            if self.t_s > .25:
                if np.random.rand() < .05:
                    self.surfaced = False
                self.t_s = 0
        else:
            if self.t_s > .25:
                if np.random.rand() < .05:
                    self.surfaced = True
                self.t_s = 0
        acc_dir = (np.random.rand(2) - .5)
        accels = acc_dir / np.linalg.norm(acc_dir) * MAX_ALFA_ACCEL * np.random.rand()
        self.vels += accels
        if self.surfaced:
            self.vels *= .99
        if np.linalg.norm(self.vels) > MAX_ALFA_SPEED:
            self.vels = self.vels / np.linalg.norm(self.vels) * MAX_ALFA_SPEED
        floc = self.loc + self.vels * t0
        if self.xbounds[0] > floc[0] or floc[0] >= self.xbounds[1]:
            self.vels[0] = -self.vels[0]
        if self.ybounds[0] > floc[1] or floc[1] >= self.ybounds[1]:
            self.vels[1] = -self.vels[1]
        self.loc += self.vels * t0
        self.pos.append(self.loc + 0.0)
        self.surf.append(pow)
        self.tk = t
        return pow, self.loc

    def reset(self):
        self.loc = self.pos[0]
        self.pos = [self.loc + 0.0]
        self.surf = [0]
        self.vels = np.random.rand(2) - .5
        self.tk = 0
        self.t_s = 0
        self.surfaced = False


@vectorize([int32(float64, float64, float64)])
def getBin(x, rng, _fs):
    n = (x * 2 / c0 - 2 * rng / c0) * _fs
    return int(n) if n - int(n) < .5 else int(n+1)


@vectorize([complex128(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)])
def getRangePoint(x, y, z, prng, pt_amp, fc, da, az_bw, el_bw, az_pt):
    el = np.arcsin(z / prng) - da
    az = np.arctan2(y, x) + az_pt
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
    agent = SinglePulseBackground()
    test = SimEnv(agent.alt, agent.az_bw, agent.el_bw, agent.dep_ang)

    pulse = agent.genChirp(np.linspace(0, 1, 10), agent.bw)

    for cpi in tqdm(range(100)):
        cpi_data, done, reward = agent.execute({'wave': np.linspace(0, 1, 10), 'radar': [100]})
        #print(agent.az_pt / DTR)

    logs = agent.log
    cols = ['blue', 'red', 'orange', 'yellow', 'green']
    fig, axes = plt.subplots(2)
    camera = Camera(fig)
    for log_idx, l in tqdm(enumerate(logs)):
        for idx, s in enumerate(agent.env.targets):
            pos = np.array(s.pos[log_idx * agent.cpi_len:log_idx * agent.cpi_len + agent.cpi_len])
            amp = np.array(s.surf[log_idx * agent.cpi_len:log_idx * agent.cpi_len + agent.cpi_len]) + 1
            if len(s.pos) > 0:
                axes[1].scatter(pos[:, 0], pos[:, 1], s=amp, c=cols[idx])
        axes[1].legend([f'{l[2][0]:.6f}-{l[2][-1]:.6f}'])
        axes[0].imshow(np.fft.fftshift(l[0], axes=1))
        axes[0].axis('tight')
        camera.snap()

    animation = camera.animate()

    esamples = np.random.rand(1000) * test.eswath
    nsamples = np.random.rand(1000) * test.swath
    _, asamples = test(esamples, nsamples, [0.1])
    eg, ng = np.meshgrid(np.linspace(0, test.eswath, 1000), np.linspace(0, test.swath, 1000))
    ug, ag = test(eg, ng, [0.1])

    plt.figure('Initial')
    plt.subplot(1, 2, 1)
    plt.imshow(ag)
    plt.subplot(1, 2, 2)
    plt.scatter(esamples, nsamples, c=asamples)

    fig = plt.figure('Waves')
    ax = plt.axes()
    cam1 = Camera(fig)
    for t in np.linspace(0, .1, 10):
        uga, aga = test(eg, ng, [t])
        ax.imshow(aga[:, :, 0])
        ax.axis('tight')
        cam1.snap()
    anim1 = cam1.animate()

    tt = np.linspace(0, 10, 1000)
    locs = test.pos(tt)

    fig = plt.figure('Plane')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])

    fig = plt.figure('Scene')
    ax = plt.axes(projection='3d')
    ax.plot(locs[0, :], locs[1, :], locs[2, :])
    ax.plot_wireframe(eg, ng, ug[:, :, 0])

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
    ax.plot_wireframe(eg, ng, ug[:, :, 0], rstride=int(eg.max() - eg.min()), cstride=int(ng.max() - ng.min()))
    for sub in agent.env.targets:
        spos = np.array(sub.pos)
        spow = np.array(sub.surf)
        ax.plot(spos[:, 0], spos[:, 1], spow)

    test_sub = Sub(0, test.eswath, 0, test.swath)
    for t in np.linspace(0, 10, 1000):
        test_sub(t)
