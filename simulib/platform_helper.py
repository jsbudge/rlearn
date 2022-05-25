import numpy as np
from scipy.interpolate import CubicSpline
from SDRParsing import SDRParse
from simulation_functions import llh2enu

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


class Platform(object):
    _pos = None
    _vel = None
    _att = None
    _heading = None

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None):
        self._gpst = t
        # Build the position spline
        ee = CubicSpline(t, e)
        nn = CubicSpline(t, n)
        uu = CubicSpline(t, u)
        self._pos = lambda lam_t: np.array([ee(lam_t), nn(lam_t), uu(lam_t)])

        # Build a velocity spline
        ve = CubicSpline(t, np.gradient(e))
        vn = CubicSpline(t, np.gradient(n))
        vu = CubicSpline(t, np.gradient(u))
        self._vel = lambda lam_t: np.array([ve(lam_t), vn(lam_t), vu(lam_t)])

        # attitude spline
        rr = CubicSpline(t, r)
        pp = CubicSpline(t, p)
        yy = CubicSpline(t, y)
        self._att = lambda lam_t: np.array([rr(lam_t), pp(lam_t), yy(lam_t)])

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[0], self._vel(lam_t)[1])

    @property
    def pos(self):
        return self._pos

    @property
    def heading(self):
        return self._heading

    @property
    def att(self):
        return self._att

    @property
    def gpst(self):
        return self._gpst


class RadarPlatform(Platform):

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, ant_offsets=None, dep_angle=45.,
                 squint_angle=0., az_bw=10., el_bw=10., fs=2e9):
        super().__init__(e, n, u, r, p, y, t)
        self.dep_ang = dep_angle * DTR
        self.squint_ang = squint_angle * DTR
        self.az_half_bw = az_bw * DTR / 2
        self.el_half_bw = el_bw * DTR / 2
        self.ant_locs = ant_offsets
        self.n_ants = ant_offsets.shape[0]
        self.fs = fs

    def calcRanges(self, height):
        nrange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang + self.el_half_bw - np.pi / 2)
        frange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang - self.el_half_bw - np.pi / 2)
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False):
        nrange, _ = self.calcRanges(height)
        plength_s = (nrange * 2 / c0 - 1 / TAC) * pulse_length_percent
        return int(plength_s * self.fs) if use_tac else plength_s

    def calcNumSamples(self, height, plp=1.):
        nrange, frange = self.calcRanges(height)
        pl_s = self.calcPulseLength(height, plp)
        return int((np.ceil((2 * frange / c0 + pl_s) * TAC) - np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)

    def calcRangeBins(self, height, upsample=1, plp=1.):
        nrange, frange = self.calcRanges(height)
        pl_s = self.calcPulseLength(height, plp)
        nsam = int((np.ceil((2 * frange / c0 + pl_s) * TAC) -
                    np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)
        MPP = c0 / self.fs / upsample / 2
        return nrange + np.arange(nsam * upsample) * MPP + c0 / self.fs


class SDRPlatform(RadarPlatform):
    _sdr = None

    def __init__(self, sdr_file, origin, ant_offsets=None, fs=2e9):
        sdr = SDRParse(sdr_file)
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        if ant_offsets is None:
            ant_offsets = np.array([np.array([ant.x, ant.y, ant.z]) for ant in sdr.port])
        super().__init__(e=e, n=n, u=u, r=sdr.gps_data['r'] + np.pi / 2, p=sdr.gps_data['p'], y=sdr.gps_data['y'],
                         t=sdr.gps_data.index.values, ant_offsets=ant_offsets, dep_angle=sdr.ant[0].dep_ang / DTR,
                         squint_angle=sdr.ant[0].squint / DTR, az_bw=sdr.ant[0].az_bw / DTR,
                         el_bw=sdr.ant[0].el_bw / DTR, fs=fs)
        self._sdr = sdr
