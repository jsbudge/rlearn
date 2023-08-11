import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import medfilt
from SDRParsing import SDRParse, AntennaPort
from simulation_functions import llh2enu, findPowerOf2, loadPostCorrectionsGPSData, loadRawData, loadGimbalData, \
    loadMatchedFilter, loadReferenceChirp

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
INS_REFRESH_HZ = 100


"""
Platform class

Base class used to hold information on radar platform, such as velocity, position, attitude, and gimbal corrections.
This is expected to provide positional information for only one pair of Tx and Rx antennas.
"""


class Platform(object):
    _pos = None
    _vel = None
    _att = None
    _heading = None

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, gimbal=None, gimbal_offset=None,
                 gimbal_rotations=None, tx_offset=None, rx_offset=None, gps_data=None):
        """
        Init function.
        :param e: array. Eastings in meters used to generate position function.
        :param n: array. Northings in meters used to generate position function.
        :param u: array. Altitude in meters used to generate position function.
        :param r: array. Roll values in degrees.
        :param p: array. Pitch values in degrees.
        :param y: array. Yaw values in degrees.
        :param t: array. GPS times, in seconds, that correspond to the above position/attitude arrays.
        :param gimbal: N x 2 array. Pan and Tilt values from the gimbal.
        :param gimbal_offset: 3-tuple. Gimbal offset from body in meters, XYZ.
        :param gimbal_rotations: 3-tuple. Initial gimbal rotation from the inertial frame in degrees.
        :param tx_offset: 3-tuple. Offset from the gimbal in XYZ, meters, for the Tx antenna.
        :param rx_offset: 3-tuple. Offset from the gimbal in XYZ, meters, for the Rx antenna.
        :param gps_data: DataFrame. This is a dataframe of GPS data, taken from the GPS debug data in APS. Optional.
        """
        self._gpst = t
        self._txant = tx_offset
        self._rxant = rx_offset
        self._gimbal = gimbal
        self._gimbal_offset = gimbal_offset
        self._gimbal_offset_mat = None
        self.gimbal_rotations = gimbal_rotations
        new_t = gps_data['sec'] if gps_data is not None else t

        # attitude spline
        rr = CubicSpline(t, r)
        pp = CubicSpline(t, p)
        yy = CubicSpline(t, y) if gps_data is None else CubicSpline(new_t, gps_data['az'] + 2 * np.pi)
        self._att = lambda lam_t: np.array([rr(lam_t), pp(lam_t), yy(lam_t)])

        # Take into account the gimbal if necessary
        if gimbal is not None:
            self._gimbal_offset_mat = getRotationOffsetMatrix(*gimbal_rotations)
            # Matrix to rotate from body to inertial frame for each INS point
            tx_offset = tx_offset if tx_offset is not None else np.array([0., 0., 0.])
            rx_offset = rx_offset if rx_offset is not None else np.array([0., 0., 0.])
            self._gimbal_pan = CubicSpline(t, gimbal[:, 0])
            self._gimbal_tilt = CubicSpline(t, gimbal[:, 1])

            # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
            if gps_data is not None:
                te = gps_data['te']
                tn = gps_data['tn']
                tu = gps_data['tu']
                re = gps_data['re']
                rn = gps_data['rn']
                ru = gps_data['ru']
            else:
                tx_corrs = np.array([getPhaseCenterInertialCorrection(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                                 p[n], r[n], tx_offset, gimbal_offset)
                                for n in range(gimbal.shape[0])])
                rx_corrs = np.array(
                    [getPhaseCenterInertialCorrection(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                      p[n], r[n], rx_offset, gimbal_offset)
                     for n in range(gimbal.shape[0])])
                te = e + tx_corrs[:, 0]
                tn = n + tx_corrs[:, 1]
                tu = u + tx_corrs[:, 2]
                re = e + rx_corrs[:, 0]
                rn = n + rx_corrs[:, 1]
                ru = u + rx_corrs[:, 2]

            # Rotate antenna into inertial frame in the same way as above
            bai = np.array([getBoresightVector(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n], p[n], r[n])
                            for n in range(gimbal.shape[0])])
            bai = bai.reshape((bai.shape[0], bai.shape[1])).T

            # Calculate antenna azimuth/elevation for beampattern
            # gphi = y - np.pi / 2 if gphi is None else gphi
            # gtheta = np.zeros(len(t)) + 20 * DTR if gtheta is None else gtheta
            if gps_data is not None:
                gtheta = np.interp(new_t, t, np.arcsin(-bai[2, :]))
                gphi = np.interp(new_t, t, np.arctan2(bai[0, :], bai[1, :]))
            else:
                gtheta = np.arcsin(-bai[2, :])
                gphi = np.arctan2(bai[0, :], bai[1, :])
        else:
            te = re = e
            tn = rn = n
            tu = ru = u
            gphi = y - np.pi / 2
            gtheta = np.zeros(len(t)) + 20 * DTR

        # Build the position spline
        ee = CubicSpline(t, e)
        nn = CubicSpline(t, n)
        uu = CubicSpline(t, u)
        self._pos = lambda lam_t: np.array([ee(lam_t), nn(lam_t), uu(lam_t)])
        tte = CubicSpline(new_t, te)
        ttn = CubicSpline(new_t, tn)
        ttu = CubicSpline(new_t, tu)
        self._txpos = lambda lam_t: np.array([tte(lam_t), ttn(lam_t), ttu(lam_t)])
        rre = CubicSpline(new_t, re)
        rrn = CubicSpline(new_t, rn)
        rru = CubicSpline(new_t, ru)
        self._rxpos = lambda lam_t: np.array([rre(lam_t), rrn(lam_t), rru(lam_t)])

        # Build a velocity spline
        ve = CubicSpline(t, medfilt(np.gradient(e), 15) * INS_REFRESH_HZ)
        vn = CubicSpline(t, medfilt(np.gradient(n), 15) * INS_REFRESH_HZ)
        vu = CubicSpline(t, medfilt(np.gradient(u), 15) * INS_REFRESH_HZ)
        self._vel = lambda lam_t: np.array([ve(lam_t), vn(lam_t), vu(lam_t)])

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[0], self._vel(lam_t)[1])

        # Beampattern stuff
        self.pan = CubicSpline(new_t, gphi)
        self.tilt = CubicSpline(new_t, gtheta)

    def boresight(self, t):
        # Get a boresight angle from the pan/tilt values
        r, p, y = self._att(t)
        return getBoresightVector(self._gimbal_offset_mat, self._gimbal_pan(t), self._gimbal_tilt(t), y, p, r)

    @property
    def pos(self):
        # Return the position lambda
        return self._pos

    @property
    def heading(self):
        # Return the heading lambda
        return self._heading

    @property
    def att(self):
        # Return the attitude lambda
        return self._att

    @property
    def gpst(self):
        # Return the array of GPS times used to create lambdas
        return self._gpst

    @property
    def rxpos(self):
        # Return Rx position lambda
        return self._rxpos

    @property
    def txpos(self):
        # Return Tx position lambda
        return self._txpos

    @property
    def vel(self):
        # Return velocity lambda
        return self._vel

    @property
    def gimbal_offset(self):
        return self._gimbal_offset


"""
RadarPlatform
Inherits from base Platform class. This is more specialized for radar, and includes some radar parameters such as
beamwidths, sampling frequencies, and angles. Intended for use in simulation work, where there isn't a SAR file to
provide these things.
"""


class RadarPlatform(Platform):

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, tx_offset=None, rx_offset=None,
                 gimbal=None, gimbal_offset=None, gimbal_rotations=None, dep_angle=45.,
                 squint_angle=0., az_bw=10., el_bw=10., fs=2e9, gps_data=None, tx_num=0, rx_num=0, wavenumber=0):
        """
        Init function.
        :param e: array. Eastings in meters used to generate position function.
        :param n: array. Northings in meters used to generate position function.
        :param u: array. Altitude in meters used to generate position function.
        :param r: array. Roll values in degrees.
        :param p: array. Pitch values in degrees.
        :param y: array. Yaw values in degrees.
        :param t: array. GPS times, in seconds, that correspond to the above position/attitude arrays.
        :param tx_offset: 3-tuple. Offset from the gimbal in XYZ, meters, for the Tx antenna.
        :param rx_offset: 3-tuple. Offset from the gimbal in XYZ, meters, for the Rx antenna.
        :param gimbal: N x 2 array. Pan and Tilt values from the gimbal.
        :param gimbal_offset: 3-tuple. Gimbal offset from body in meters, XYZ.
        :param gimbal_rotations: 3-tuple. Initial gimbal rotation from the inertial frame in degrees.
        :param dep_angle: float. Depression angle of antenna in degrees.
        :param squint_angle: float. Squint angle of antenna in degrees.
        :param az_bw: float. Beamwidth of antenna in azimuth, in degrees.
        :param el_bw: float. Beamwidth of antenna in elevation, in degrees.
        :param fs: float. Sampling frequency in Hz.
        :param gps_data: DataFrame. This is a dataframe of GPS data, taken from the GPS debug data in APS. Optional.
        """
        super().__init__(e, n, u, r, p, y, t, gimbal, np.array(gimbal_offset), np.array(gimbal_rotations),
                         tx_offset, rx_offset, gps_data)
        self.dep_ang = dep_angle * DTR
        self.squint_ang = squint_angle * DTR
        self.az_half_bw = az_bw * DTR / 2
        self.el_half_bw = el_bw * DTR / 2
        self.fs = fs
        self.near_range_angle = self.dep_ang + self.el_half_bw
        self.far_range_angle = self.dep_ang - self.el_half_bw
        self.rx_num = rx_num
        self.tx_num = tx_num
        self.wavenumber = wavenumber
        self.pulse = None
        self.mfilt = None
        self.fc = None
        self.bwidth = None

    def calcRanges(self, height, exp_factor=1):
        """
        Calculates out the expected near and far slant ranges of the antenna.
        :param exp_factor: float. Expansion factor for range.
        :param height: float. Height of antenna off the ground in meters.
        :return: tuple of near and far slant ranges in meters.
        """
        nrange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang + self.el_half_bw * exp_factor)
        frange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang - self.el_half_bw * exp_factor)
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False, nrange=None):
        """
        Calculates the pulse length given a height off the ground and a pulse percent.
        :param nrange: float. Near range, for override, if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param pulse_length_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :param use_tac: bool. If True, returns the pulse length in TAC. If False, returns in seconds.
        :return: Expected pulse length in either TAC or seconds.
        """
        if nrange is None:
            nrange, _ = self.calcRanges(height)
        plength_s = (nrange * 2 / c0) * pulse_length_percent
        return int(np.ceil(plength_s * self.fs)) if use_tac else plength_s

    def calcNumSamples(self, height, plp=1., ranges=None):
        """
        Calculate the number of samples in a pulse.
        :param ranges: 2-tuple. (Near range, Far range) for range override if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param plp: float, <1. Percentage of maximum pulse length to use in radar.
        :return: Number of samples in a returned pulse.
        """
        if ranges is None:
            nrange, frange = self.calcRanges(height)
        else:
            nrange = ranges[0]
            frange = ranges[1]
        pl_s = self.calcPulseLength(height, plp)
        return int((np.ceil((2 * frange / c0 + pl_s) * TAC) - np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)

    def calcRangeBins(self, height, upsample=1, plp=1., ranges=None):
        """
        Calculates the range bins for a given pulse.
        :param ranges: 2-tuple. (Near range, Far range) for range override if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param upsample: int. Upsample factor.
        :param plp: float, <1. Percentage of maximum pulse length to use in radar.
        :return: array of range bins.
        """
        if ranges is None:
            nrange, frange = self.calcRanges(height)
        else:
            nrange = ranges[0]
            frange = ranges[1]
        pl_s = self.calcPulseLength(height, plp, nrange=None if ranges is None else ranges[0])
        nsam = int((np.ceil((2 * frange / c0 + pl_s) * TAC) -
                    np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)
        MPP = c0 / self.fs / upsample / 2
        return nrange + np.arange(nsam * upsample) * MPP + c0 / self.fs


"""
SDRPlatform
More specialized version of RadarPlatform that gets all of the values it needs from a SAR file or instance of
SDRParse.
"""


class SDRPlatform(RadarPlatform):
    _sdr = None

    def __init__(self, sdr_file, origin=None, tx_offset=None, rx_offset=None, fs=None, channel=0, gps_debug=None,
                 gimbal_debug=None, gimbal_offset=None, gps_replace=None, use_ecef=True):
        """
        Init function.
        :param sdr_file: SDRParse object or str. This is path to the SAR file used as a basis for other calculations,
            or the SDRParse object of an already parsed file.
        :param origin: 3-tuple. Point used as the origin for ENU reference frame, in (lat, lon, alt).
        :param tx_offset: 3-tuple. Offset of Tx antenna from body frame in meters.
        :param rx_offset: 3-tuple. Offset of Rx antenna from body frame in meters.
        :param fs: float. Sampling frequency in Hz.
        :param channel: int. Channel of data for this object to represent in the SAR file.
        :param gps_debug: str. Path to a file of GPS debug data from APS for this collect. Optional.
        :param gimbal_debug: str. Path to a file of Gimbal debug data from APS for this collect. Optional.
        """
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])
        if gps_debug is not None:
            gps_data = gps_debug
            t = gps_data['sec']
            if use_ecef:
                gps_data['te'], gps_data['tn'], gps_data['tu'] = llh2enu(gps_data['tx_lat'], gps_data['tx_lon'],
                                                                         gps_data['tx_alt'], origin)
                gps_data['re'], gps_data['rn'], gps_data['ru'] = llh2enu(gps_data['rx_lat'], gps_data['rx_lon'],
                                                                         gps_data['rx_alt'], origin)
            else:
                gps_data['tn'] = gps_data['tx_lat'] * gps_data['latConv'] - origin[0] * gps_data['latConv']
                gps_data['te'] = gps_data['tx_lon'] * gps_data['lonConv'] - origin[1] * gps_data['lonConv']
                gps_data['tu'] = gps_data['tx_alt'] - origin[2]

                gps_data['rn'] = gps_data['rx_lat'] * gps_data['latConv'] - origin[0] * gps_data['latConv']
                gps_data['re'] = gps_data['rx_lon'] * gps_data['lonConv'] - origin[1] * gps_data['lonConv']
                gps_data['ru'] = gps_data['rx_alt'] - origin[2]

        else:
            gps_data = None
        if gps_replace is not None:
            if use_ecef:
                e, n, u = llh2enu(gps_replace['lat'], gps_replace['lon'], gps_replace['alt'], origin)
            else:
                e = gps_replace['lat'] * gps_data['latConv'] - origin[0] * gps_data['latConv']
                n = gps_replace['lon'] * gps_data['lonConv'] - origin[1] * gps_data['lonConv']
                u = gps_replace['alt'] - origin[2]
            r = np.interp(t, gps_replace['gps_ms'], gps_replace['r'])
            p = np.interp(t, gps_replace['gps_ms'], gps_replace['p'])
            y = np.interp(t, gps_replace['gps_ms'], np.angle(gps_replace['azimuthY'] + 1j * gps_replace['azimuthX']))
            e = np.interp(t, gps_replace['gps_ms'], e)
            n = np.interp(t, gps_replace['gps_ms'], n)
            u = np.interp(t, gps_replace['gps_ms'], u)
        else:
            e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
            r = sdr.gps_data['r'].values
            p = sdr.gps_data['p'].values
            y = sdr.gps_data['y'].values
        if gimbal_debug is None:
            try:
                pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                                sdr.gimbal['pan'].values.astype(np.float64))
                tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                                sdr.gimbal['tilt'].values.astype(np.float64))
            except TypeError:
                pan = np.zeros_like(sdr.gps_data['systime'].values)
                tilt = np.zeros_like(sdr.gps_data['systime'].values)
            pan = np.interp(t, sdr.gps_data.index.values, pan)
            tilt = np.interp(t, sdr.gps_data.index.values, tilt)
        else:
            gim = loadGimbalData(gimbal_debug)
            if len(gim['systime']) == 0:
                # This is for the weird interpolated to each channel data
                times = sdr[channel].pulse_time[:len(gim['pan'])]
            else:
                times = np.interp(gim['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
            pan = np.interp(t, times, gim['pan'])
            tilt = np.interp(t, times, gim['tilt'])
        goff = np.array([sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset]) if gimbal_offset is None else gimbal_offset
        grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])
        try:
            channel_dep = (sdr.xml['Channel_0']['Near_Range_D'] + sdr.xml['Channel_0']['Far_Range_D']) / 2
        except KeyError:
            channel_dep = sdr.ant[0].dep_ang / DTR
        if sdr[channel].is_receive_only:
            tx_num = np.where([isinstance(n, AntennaPort) for n in sdr.port])[0][0]
        else:
            tx_num = sdr[channel].trans_num
            tx_offset = np.array([sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num
        rx_offset = np.array([sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]) if rx_offset is None else rx_offset
        super().__init__(e=e, n=n, u=u, r=r, p=p, y=y, t=t, tx_offset=tx_offset, rx_offset=rx_offset,
                         gimbal=np.array([pan, tilt]).T, gimbal_offset=goff, gimbal_rotations=grot,
                         dep_angle=channel_dep, squint_angle=sdr.ant[sdr.port[tx_num].assoc_ant].squint / DTR,
                         az_bw=sdr.ant[sdr.port[tx_num].assoc_ant].az_bw / DTR,
                         el_bw=sdr.ant[sdr.port[tx_num].assoc_ant].el_bw / DTR, fs=fs, gps_data=gps_data, tx_num=tx_num,
                         rx_num=rx_num)
        self._sdr = sdr
        self.origin = origin
        self._channel = channel

    def calcRanges(self, fdelay, partial_pulse_percent=1.):
        """
        Calculate near and far ranges for this collect using the SAR file.
        :param fdelay: float. FDelay desired in TAC.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: tuple of near and far ranges in meters.
        """
        nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        return nrange, frange

    def calcPulseLength(self, height=0, pulse_length_percent=1., use_tac=False):
        """
        Calculate the pulse length for this collect.
        :param height: Not used. Here for compatibility with parent classes.
        :param pulse_length_percent: Not used. Here for compatibility with parent classes.
        :param use_tac: bool. If True, returns pulse length in TAC, otherwise in seconds.
        :return: Pulse length in TAC, otherwise in seconds.
        """
        return self._sdr[self._channel].pulse_length_N if use_tac else self._sdr[self._channel].pulse_length_S

    def calcNumSamples(self, height=0, plp=1.):
        """
        Get number of samples in a pulse.
        :param height: Not used. Here for compatibility with parent classes.
        :param plp: Not used. Here for compatibility with parent classes.
        :return: Number of samples in a pulse.
        """
        return self._sdr[self._channel].nsam

    def calcRangeBins(self, fdelay, upsample=1, partial_pulse_percent=1.):
        """
        Calculate range bins for a pulse/collect.
        :param fdelay: float. FDelay desired for this pulse in TAC.
        :param upsample: int. Upsample factor.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: array of range bins in meters.
        """
        nrange, frange = self.calcRanges(fdelay, partial_pulse_percent=partial_pulse_percent)
        MPP = c0 / self.fs / upsample
        return (nrange * 2 + np.arange(self.calcNumSamples() * upsample) * MPP) / 2

    def calcRadVelRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the radial velocity resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Radial velocity resolution in meters per second.
        """
        return dopplerBroadeningFactor * c0 * self._sdr[self._channel].prf / \
                     (self._sdr[self._channel].fc * cpi_len)

    def calcDopRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the Doppler resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Doppler resolution in Hz.
        """
        return dopplerBroadeningFactor * self._sdr[self._channel].prf / cpi_len

    def calcWrapVel(self):
        """
        Calculate the wrap velocity for this collect.
        :return: Wrap velocity in meters per second.
        """
        return self._sdr[self._channel].prf * (c0 / self._sdr[self._channel].fc) / 4.0


def bodyToInertial(yaw, pitch, roll, x, y, z):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # compute the inertial to body rotation matrix
    rotItoB = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])
    return rotItoB.T.dot(np.array([x, y, z]))


def inertialToBody(yaw, pitch, roll, x, y, z):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # compute the inertial to body rotation matrix
    rotItoB = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])
    return np.linalg.pinv(rotItoB).T.dot(np.array([x, y, z]))


def gimbalToBody(rotBtoMG, pan, tilt, x, y, z):
    cp = np.cos(pan)
    sp = np.sin(pan)
    ct = np.cos(tilt)
    st = np.sin(tilt)

    rotMGtoGP = np.array([
        [cp, -sp, 0],
        [sp * ct, cp * ct, st],
        [-sp * st, -cp * st, ct]])

    # compute the gimbal mounted to gimbal pointing rotation matrix
    rotBtoGP = rotMGtoGP.dot(rotBtoMG)
    return rotBtoGP.T.dot(np.array([x, y, z]))


def getRotationOffsetMatrix(roll0, pitch0, yaw0):
    cps0 = np.cos(yaw0)
    sps0 = np.sin(yaw0)
    cph0 = np.cos(roll0)
    sph0 = np.sin(roll0)
    cth0 = np.cos(pitch0)
    sth0 = np.sin(pitch0)

    Delta1 = cph0 * cps0 + sph0 * sth0 * sps0
    Delta2 = -cph0 * sps0 + sph0 * sth0 * cps0
    Delta3 = -sph0 * cth0
    Delta4 = cth0 * sps0
    Delta5 = cth0 * cps0
    Delta6 = sth0
    Delta7 = sph0 * cps0 - cph0 * sth0 * sps0
    Delta8 = -sph0 * sps0 - cph0 * sth0 * cps0
    Delta9 = cph0 * cth0

    return np.array([[-Delta1, -Delta2, -Delta3], [Delta4, Delta5, Delta6], [-Delta7, -Delta8, -Delta9]])


def getBoresightVector(ROffset, pan, tilt, yaw, pitch, roll):
    """Returns the a 3x1 numpy array with the normalized boresight vector in 
    the inertial frame"""
    # set the boresight pointing vector in the pointed gimbal frame
    delta_gp = np.array([[0], [0], [1.0]])
    # rotate these into the body frame
    delta_b = gimbalToBody(
        ROffset, pan, tilt, *delta_gp)
    # return the boresight in the inertial frame
    return bodyToInertial(yaw, pitch, roll, *delta_b)


def getPhaseCenterInertialCorrection(rotBtoMG, pan, tilt, yaw, pitch, roll, ant_offset, gimbal_offset):
    """Returns the 3x1 numpy array with the inertial translational
    correction to be applied to the INS position value to have the position
    of the antenna phase center."""

    # rotate the antenna offsets (provided in the gimbal frame) into the
    #   the body frame of the INS
    antDelta_b = gimbalToBody(rotBtoMG, pan, tilt, *ant_offset)
    # add the antenna offset in the body frame to the offsets measured from
    #   the INS center to the gimbal axes center of rotation
    totDelta_b = antDelta_b + gimbal_offset
    # return the inertial correction
    return bodyToInertial(yaw, pitch, roll, totDelta_b[0], totDelta_b[1], totDelta_b[2])

