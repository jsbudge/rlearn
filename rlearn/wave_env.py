from tensorforce import Environment
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.ndimage import gaussian_filter
from tftb.processing import WignerVilleDistribution
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numba import cuda
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
        self.cpi_len = 64
        self.az_bw = 12 * DTR
        self.el_bw = 12 * DTR
        self.dep_ang = 45 * DTR
        self.alt = 1524
        self.plp = .4
        self.fc = 9.6e9
        self.samples = 200000
        self.fs = fs / 4
        self.bw = 200e6
        self.scanRate = 45 * DTR
        self.scanDir = 1
        self.az_pt = np.pi / 2
        self.reset()
        self.data_block = (self.nsam, self.cpi_len)
        MPP = c0 / 2 / self.fs
        self.range_bins = cupy.array(self.env.nrange + np.arange(self.nsam) * MPP, dtype=np.float64)

    def states(self):
        return dict(type='float', shape=(self.nsam, self.cpi_len))

    def actions(self):
        return dict(wave=dict(type='float', shape=(10,), min_value=0, max_value=1),
                    radar=dict(type='float', shape=(1,), min_value=10, max_value=500))

    def execute(self, actions):
        self.tf = self.tf[-1] + np.linspace(0, 1 / actions['radar'][0], self.cpi_len)
        chirp = np.fft.fft(self.genChirp(actions['wave'], self.bw), self.fft_len)
        cpi = self.genCPI(chirp)
        state = db(cpi)
        reward = 0

        # We've reached the end of the data, pull out
        done = False if self.tf[-1] < 10 else True

        # Sidelobe score
        ml_width = 0
        slidx = 0
        rc_chirp = db(np.fft.ifft(chirp * (chirp * taylor(self.fft_len)).conj().T, self.fft_len * 8))
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

        self.log.append([db(state), [sll, 1 - mll], self.tf, self.az_pan + 0.0])

        return db(state), done, reward

    def reset(self, num_parallel=None):
        self.tf = np.linspace(0, self.cpi_len / 500.0, self.cpi_len)
        self.env = SimEnv(self.alt, self.az_bw, self.el_bw, self.dep_ang)
        self.log = []
        self.nsam = int((np.ceil((2 * self.env.frange / c0 + self.env.max_pl * self.plp) * TAC) -
                         np.floor(2 * self.env.nrange / c0 * TAC)) * self.fs / TAC)
        self.nr = int(self.env.max_pl * self.plp * self.fs)
        self.fft_len = findPowerOf2(self.nsam + self.nr)
        return np.zeros((self.nsam, self.cpi_len))

    def genCPI(self, chirp):
        twin = taylor(self.fft_len)
        win_gpu = cupy.array(np.tile(twin, (self.cpi_len, 1)).T, dtype=np.complex128)
        chirp_gpu = cupy.array(np.tile(chirp, (self.cpi_len, 1)).T, dtype=np.complex128)

        blocks_per_grid = (
            int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])), int(np.ceil(self.samples / THREADS_PER_BLOCK[1])))
        sub_blocks = (int(np.ceil(self.cpi_len / THREADS_PER_BLOCK[0])),
                      int(np.ceil(len(self.env.targets) / THREADS_PER_BLOCK[1])))
        pos_gpu = cupy.array(np.ascontiguousarray(self.env.pos(self.tf)), dtype=np.float64)
        az_pan = self.az_pt + (self.tf - self.tf[0]) * self.scanRate * self.scanDir
        if np.any(az_pan < np.pi / 4):
            az_pan[az_pan < np.pi / 4] = np.pi / 4 + (self.tf - self.tf[0])[az_pan < np.pi / 4] * self.scanRate
            self.scanDir = 1
        elif np.any(az_pan > np.pi * 3 / 4):
            az_pan[az_pan > np.pi * 3 / 4] = np.pi * 3 / 4 - \
                                             (self.tf - self.tf[0])[az_pan > np.pi * 3 / 4] * self.scanRate
            self.scanDir = -1
        self.az_pan = az_pan
        pan_gpu = cupy.array(np.ascontiguousarray(az_pan), dtype=np.float64)
        p_gpu = cupy.array(np.array([np.pi / self.el_bw, np.pi / self.az_bw, c0 / self.fc, self.env.nrange / c0,
                                     self.fs, self.dep_ang, self.env.eswath, self.env.swath]), dtype=np.float64)
        data_r = cupy.zeros(self.data_block, dtype=np.float64)
        data_i = cupy.zeros(self.data_block, dtype=np.float64)
        gx = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        gy = cupy.array(np.random.rand(self.samples), dtype=np.float64)
        sv = []
        for sub in self.env.targets:
            sv.append([sub(t) for t in self.tf])
        sub_pos = cupy.array(np.ascontiguousarray(sv), dtype=np.float64)
        times = cupy.array(np.ascontiguousarray(self.tf), dtype=np.float64)

        genRangeProfile[blocks_per_grid, THREADS_PER_BLOCK](pos_gpu, gx, gy, pan_gpu,
                                                            times, data_r, data_i, p_gpu)
        cupy.cuda.Device().synchronize()

        genSubProfile[sub_blocks, THREADS_PER_BLOCK](pos_gpu, sub_pos, pan_gpu, data_r, data_i, p_gpu)
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
        self.swath = (gfrange - gnrange)
        self.eswath = blen * 2
        self.nrange = nrange
        self.frange = frange
        self.gnrange = gnrange
        self.gmrange = gmrange
        self.gfrange = gfrange
        self.gbwidth = blen
        self.h_agl = h_agl
        self.targets = []
        self.genFlightPath()
        for n in range(np.random.randint(1, 5)):
            self.targets.append(Sub(0, self.eswath,
                                    0, self.swath))

    def genFlightPath(self):
        # We start assuming that the bottom left corner of scene is (0, 0, 0) ENU
        # Platform motion (assuming 100 Hz signal, so this is 10 sec)
        npts = 50
        tt = np.linspace(0, 10, npts)
        e = gaussian_filter(
            np.linspace(self.eswath / 4, self.eswath - self.eswath / 4, npts) + (np.random.rand(npts) - .5) * 3, 3)
        n = gaussian_filter(-self.gnrange + (np.random.rand(npts) - .5) * 3, 3)
        u = gaussian_filter(self.h_agl + (np.random.rand(npts) - .5) * 3, 3)
        re = UnivariateSpline(tt, e, s=.7, k=3)
        rn = UnivariateSpline(tt, n, s=.7, k=3)
        ru = UnivariateSpline(tt, u, s=.7, k=3)
        self.pos = lambda t: np.array([re(t), rn(t), ru(t)])
        self.spd = np.linalg.norm(np.gradient(self.pos(np.linspace(0, 10, 100)), axis=1), axis=0).mean()

        # Environment pulse info
        self.max_pl = (self.nrange * 2 / c0 - 1 / TAC) * .99

    def getAntennaBeamLocation(self, t, ang):
        fpos = self.pos(t)
        eshift = np.cos(ang) * self.gmrange
        nshift = np.sin(ang) * self.gmrange
        return fpos[0] + eshift, fpos[1] + nshift, self.gbwidth, (self.gfrange - self.gnrange) / 2


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
            pow = 50
            if self.t_s > .25:
                #if np.random.rand() < .05:
                self.surfaced = False
                self.t_s = 0
        else:
            if self.t_s > .25:
                #if np.random.rand() < .05:
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
        return pow, *self.loc

    def reset(self):
        self.loc = self.pos[0]
        self.pos = [self.loc + 0.0]
        self.surf = [0]
        self.vels = np.random.rand(2) - .5
        self.tk = 0
        self.t_s = 0
        self.surfaced = False


@cuda.jit(device=True)
def wavefunction(x, y, t):
    return .25 * cmath.exp(1j * (.70710678 / 100 * x + .70710678 / 100 * y + 2 * np.pi * .1 * t)) + \
        .47 * cmath.exp(1j * (0.9486833 / 40 * x + 0.31622777 / 40 * y + 2 * np.pi * 10 * t))


def wave_cpu(x, y, t):
    return .25 * np.exp(1j * (.70710678 / 100 * x + .70710678 / 100 * y + 2 * np.pi * .1 * t)) + \
        .47 * np.exp(1j * (0.9486833 / 40 * x + 0.31622777 / 40 * y + 2 * np.pi * 10 * t))


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(
    'void(float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:, :], float64[:, :], float64[:])')
def genRangeProfile(path, gx, gy, pan, t, pd_r, pd_i, params):
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
            eldiff = diff(el_tx, params[5])
            azdiff = diff(az_tx, pan[tt])
            tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            att = tx_elpat * tx_azpat
            acc_val = wp.imag * att * cmath.exp(-1j * wavenumber * rng * 2) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit('void(float64[:, :], float64[:, :, :], float64[:], float64[:, :], float64[:, :], float64[:])')
def genSubProfile(path, subs, pan, pd_r, pd_i, params):
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
        s_x = tx - path[0, tt]
        s_y = ty - path[1, tt]
        s_z = tz - path[2, tt]
        rng = math.sqrt(s_x * s_x + s_y * s_y + s_z * s_z)
        rng_bin = (rng * 2 / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            el_tx = math.asin(-s_z / rng)
            az_tx = math.atan2(s_x, s_y)
            eldiff = diff(el_tx, params[5])
            azdiff = diff(az_tx, pan[tt])
            tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            att = tx_elpat * tx_azpat
            acc_val = spow * att * cmath.exp(-1j * wavenumber * rng * 2) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)

            # Add as a small gaussian
            cuda.atomic.add(pd_r, (but - 1, np.uint64(tt)), acc_val.real * .6)
            cuda.atomic.add(pd_i, (but - 1, np.uint64(tt)), acc_val.imag * .6)
            cuda.atomic.add(pd_r, (but + 1, np.uint64(tt)), acc_val.real * .6)
            cuda.atomic.add(pd_i, (but + 1, np.uint64(tt)), acc_val.imag * .6)
            cuda.atomic.add(pd_r, (but - 2, np.uint64(tt)), acc_val.real * .2)
            cuda.atomic.add(pd_i, (but - 2, np.uint64(tt)), acc_val.imag * .2)
            cuda.atomic.add(pd_r, (but + 2, np.uint64(tt)), acc_val.real * .2)
            cuda.atomic.add(pd_i, (but + 2, np.uint64(tt)), acc_val.imag * .2)


def ellipse(x, y, a, b, ang):
    t = np.linspace(0, 2 * np.pi, 100)
    ell = np.array([a * np.cos(t), b * np.sin(t)])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    fin_ell = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        fin_ell[:, i] = np.dot(rot, ell[:, i])
    return fin_ell + np.array([x, y])[:, None]


if __name__ == '__main__':
    agent = SinglePulseBackground()
    test = SimEnv(agent.alt, agent.az_bw, agent.el_bw, agent.dep_ang)

    pulse = agent.genChirp(np.linspace(0, 1, 10), agent.bw)

    for cpi in tqdm(range(200)):
        cpi_data, done, reward = agent.execute({'wave': np.linspace(0, 1, 10), 'radar': [100]})
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
        main_beam = ellipse(*(list(agent.env.getAntennaBeamLocation(l[2][0], l[3][0])) + [l[3][0]]))
        axes[1].plot(main_beam[0, :], main_beam[1, :], 'gray')
        axes[1].scatter(fpos[0], fpos[1], marker='*', c='blue')
        for idx, s in enumerate(agent.env.targets):
            pos = np.array(s.pos[log_idx * agent.cpi_len * skips])
            amp = np.array(s.surf[log_idx * agent.cpi_len * skips]) + 1
            if len(s.pos) > 0:
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

