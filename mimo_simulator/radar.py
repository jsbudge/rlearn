
import pylab as plab
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.signal.windows import taylor
from scipy.signal import hilbert
from scipy.spatial.transform import Rotation as rot
import numpy as np
import numba

import rawparser as rp
import simlib as sl
from useful_lib import findAllFilenames, findPowerOf2, db

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

'''
Radar Class

defines a radar based on an XML file.
'''


# Container class for radar data and parameters
def genPulse(phase_x, phase_y, nr, t0, fc, bandw):
    phase = fc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * t0 / nr))


class Radar(object):

    """
    init method for Radar object
    :param fnme: str of .sar file used to source all the other data
    :param llr: (double, double, double) of lat/lon/height reference point used to do all coordinate frame conversions
    :param offsets: (double, double, double) of antenna offsets, in XYZ in antenna frame, if used in MIMO configuration
    :param use_mean_pantilt: bool flag that disables use of gimbal pan/tilt values from debug file
    :param use_xml_flightpath: bool flag that disables use of GPS flight data from debug file
    :param presum: int that sets what the presum should be. If None, uses the value from XML file
    """
    def __init__(self, fnme, llr, offsets=None, use_mean_pantilt=False, use_xml_flightpath=False, presum=None):
        files = findAllFilenames(fnme)
        mean_heading = 0
        xml_params = rp.loadXMLFile(files['xml'])
        offsets = np.array([0, 0, 0]) if offsets is None else offsets
        presum = int(xml_params['Presum_Factor']) if presum is None else presum
        nframes, nsam, att, systime = rp.getRawSDRParams(files['RawData'])
        if presum != 1:
            systime = np.interp(np.linspace(0, nframes, nframes * presum), np.arange(nframes), systime)
            att = np.repeat(att, presum)
        nr = int(xml_params['Pulse_Length_S'] * fs)
        params = {**xml_params, 'Nr': nr, 'lambda': c0 / xml_params['Center_Frequency_Hz']}
        fft_len = findPowerOf2(nsam + params['Nr'])
        if type(files['ash']) != dict:
            fdelay = float(files['ash'].split('/')[-1].split('.')[0].split('_')[-1]) / 10
        else:
            fdelay = float(files['ash'][[f for f in files['ash']][0]].split('/')[-1].split('.')[0].split('_')[-1]) / 10

        if use_xml_flightpath:
            # Geometry calculations to get slant ranges and range bins
            h_rov = params['Flight_Line_Altitude_M'] + llr[2]
            h_agl = params['Flight_Line_Altitude_M']
            near_slant_range = ((params['Receive_On_TAC'] - params['Transmit_On_TAC'] - fdelay + 8) / TAC) * c0 / 2
            MPP = c0 / 2 / fs
            range_bins = near_slant_range + np.arange(nsam) * MPP
            far_slant_range = range_bins[-1]
            near_ground_range = np.sqrt(near_slant_range ** 2 - h_agl ** 2)
            far_ground_range = np.sqrt(far_slant_range ** 2 - h_agl ** 2)
            mid_ground_range = (far_ground_range + near_ground_range) / 2

            # Generate spline functions to get us the position of the rover at any system time
            se, sn, su = sl.llh2enu(params['Start_Latitude_D'], params['Start_Longitude_D'], h_rov, llr)
            ee, en, eu = sl.llh2enu(params['Stop_Latitude_D'], params['Stop_Longitude_D'], h_rov, llr)
            re = lambda t: (ee - se) * t * TAC / systime[-1] + se + offsets[0]
            rn = lambda t: (en - sn) * t * TAC / systime[-1] + sn + offsets[1]
            ru = lambda t: h_agl * np.ones((len(t),)) + offsets[2] \
                if type(t) != np.float64 else h_agl + offsets[2]
            self.pos = lambda t: np.array([re(t), rn(t), ru(t)])
            apsdata = None
            gps_times = np.linspace(systime[0], systime[-1], int((systime[-1] - systime[0]) / TAC / .01)) / TAC

            mean_heading = params['Initial_Course_Angle_R'] - np.pi / 2
            rad_pan = lambda xp: mean_heading * np.ones((len(xp),)) if type(xp) != np.float64 else mean_heading
            rad_tilt = lambda xt: params['Depression_Angle_D'] * DTR * np.ones((len(xt),)) \
                if type(xt) != np.float64 else params['Depression_Angle_D'] * DTR

        else:
            data = rp.loadGPSData(files['GPSDataPostJumpCorrection'])
            apsdata = pd.DataFrame(data=data).set_index(['gps_ms'])
            apsdata['time'] = apsdata['systime'] / TAC
            gps_times = apsdata['time'].values

            # Get position of rover in ECEF and ENU based on reference point
            x, y, z = sl.llh2ecef(apsdata['lat'].values,
                                  apsdata['lon'].values,
                                  apsdata['alt'].values - sl.undulationEGM96(apsdata['lat'].values,
                                                                             apsdata['lon'].values))
            apsdata['ex'] = x
            apsdata['ey'] = y
            apsdata['ez'] = z
            apsdata['y'] = np.angle(apsdata['vn'] + 1j * apsdata['ve'])
            e, n, u = sl.ecef2enu(x, y, z, llr)
            if 'Gimbal' in files:
                # Account for gimbal in the position of the antenna
                gimbal = pd.DataFrame(rp.loadGimbalData(files['Gimbal']))
                for col in ['pan', 'tilt']:
                    spline = CubicSpline(gimbal['systime'], gimbal[col])
                    apsdata['gimbal_' + col] = spline(apsdata['systime'])

                # Matrix to rotate from body to inertial frame for each INS point
                Rbi = [rot.from_rotvec(row[['p', 'r', 'y']].values) for idx, row in
                       apsdata.iterrows()]

                # Account for gimbal frame mounting rotations
                Rgb2g = rot.from_rotvec(np.array([0, params['Roll_D'] * DTR, 0]))
                Rb2gblg = rot.from_rotvec(np.array([params['Pitch_D'] * DTR, 0, 0]))
                Rblgb = rot.from_rotvec(np.array([0, 0, params['Yaw_D'] * DTR]))
                # This is because the gimbal is mounted upside down
                Rmgg = rot.from_rotvec([0, -np.pi, 0])
                Rgb = Rmgg * Rgb2g * Rb2gblg * Rblgb
                gimbaloffsets = np.array([params['Gimbal_X_Offset_M'],
                                          params['Gimbal_Y_Offset_M'],
                                          params['Gimbal_Z_Offset_M']])
                ant_offsets = np.array([params['Antenna_X_Offset_M'] + offsets[0],
                                        params['Antenna_Y_Offset_M'] + offsets[1],
                                        params['Antenna_Z_Offset_M'] + offsets[2]])

                # Convert gimbal angles to rotations
                Rmg = [rot.from_rotvec([row['gimbal_tilt'], 0, row['gimbal_pan']])
                       for idx, row in apsdata.iterrows()]

                # Apply rotations through antenna frame, gimbal frame, and add to gimbal offsets
                gamma_b_gpc = [(Rgb * n).inv().apply(ant_offsets).flatten() + gimbaloffsets for n in Rmg]

                # Rotate gimbal/antenna offsets into inertial frame
                rotated_offsets = np.array([Rbi[i].inv().apply(gamma_b_gpc[i]).flatten()
                                            for i in range(apsdata.shape[0])])

                # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
                apsdata['e'] = e + rotated_offsets[:, 1]
                apsdata['n'] = n + rotated_offsets[:, 0]
                apsdata['u'] = u - rotated_offsets[:, 2]

                # Rotate antenna into inertial frame in the same way as above
                boresight = np.array([0, 0, 1])
                bai = np.array([(Rbi[n].inv() * (Rgb * Rmg[n]).inv()).apply(boresight).flatten()
                                for n in range(apsdata.shape[0])])

                # Calculate antenna azimuth/elevation for beampattern
                gtheta = np.arcsin(-bai[:, 2])
                gphi = np.arctan2(-bai[:, 1], bai[:, 0])
                if use_mean_pantilt:
                    mean_heading = gphi.mean()
                    rad_pan = lambda xp: mean_heading * np.ones((len(xp),)) if type(xp) != np.float64 else mean_heading
                    rad_tilt = lambda xt: params['Depression_Angle_D'] * DTR * np.ones((len(xt),)) \
                        if type(xt) != np.float64 else params['Depression_Angle_D'] * DTR
                else:
                    rad_pan = CubicSpline(gps_times, gphi)
                    rad_tilt = CubicSpline(gps_times, gtheta)
            else:
                apsdata['e'] = e + offsets[0]
                apsdata['n'] = n + offsets[1]
                apsdata['u'] = u + offsets[2]
                if use_mean_pantilt:
                    mean_heading = apsdata['y'].mean() - np.pi / 2
                    rad_pan = lambda xp: mean_heading * np.ones((len(xp),)) if type(xp) != np.float64 else mean_heading
                    rad_tilt = lambda xt: params['Depression_Angle_D'] * DTR * np.ones((len(xt),)) \
                        if type(xt) != np.float64 else params['Depression_Angle_D'] * DTR
                else:
                    rad_pan = CubicSpline(gps_times, apsdata['y'].values - np.pi / 2)
                    rad_tilt = CubicSpline(gps_times, apsdata['r'].values)

            # Geometry calculations to get slant ranges and range bins
            h_rov = apsdata['alt'].values[0]
            h_agl = (h_rov - llr[2])
            near_slant_range = ((params['Receive_On_TAC'] - params['Transmit_On_TAC'] - fdelay + 8) / TAC) * c0 / 2
            MPP = c0 / 2 / fs
            range_bins = near_slant_range + np.arange(nsam) * MPP
            far_slant_range = range_bins[-1]
            near_ground_range = np.sqrt(near_slant_range ** 2 - h_agl ** 2)
            far_ground_range = np.sqrt(far_slant_range ** 2 - h_agl ** 2)
            mid_ground_range = (far_ground_range + near_ground_range) / 2

            # Generate spline functions to get us the position of the rover at any system time
            re = CubicSpline(gps_times, apsdata['e'].values)
            rn = CubicSpline(gps_times, apsdata['n'].values)
            ru = CubicSpline(gps_times, apsdata['u'].values)
            self.pos = lambda t: np.array([re(t), rn(t), ru(t)])

        self.locdata = apsdata
        self.params = params
        self.bandwidth = params['Bandwidth_Hz']
        self.fc = params['Center_Frequency_Hz']
        self.el_bw = params['Elevation_Beamwidth_D'] * DTR
        self.az_bw = params['Azimuth_Beamwidth_D'] * DTR
        self.wavelength = params['lambda']
        self.prf = params['Doppler_PRF_Hz']
        self.tp = nsam / fs
        self.t0 = params['Pulse_Length_S']
        self.offsets = offsets
        self.range_bins = range_bins
        self.far_slant_range = far_slant_range
        self.near_ground_range = near_ground_range
        self.near_slant_range = near_slant_range
        self.far_ground_range = far_ground_range
        self.mid_ground_range = mid_ground_range
        self.nsam = nsam
        self.nframes = nframes
        self.att = att
        self.systimes = systime
        self.times = systime / TAC
        self.gps_times = gps_times
        self.upsample = 1
        self.upsample_nsam = nsam
        self.pan = rad_pan
        self.tilt = rad_tilt
        self.r_angle = apsdata['y'].mean() - np.pi / 2 if apsdata is not None else mean_heading - np.pi / 2
        self.fft_len = fft_len
        self.fdelay = fdelay
        self.presum = presum
        self.is_presummed = True if presum != 1 else False
        self.velocity = params['Velocity_Knots'] * .514444

        # A few member variables for convenience
        self.nr = params['Nr']
        self.chirp = None
        self.fft_chirp = None
        self.mf = None

    def upsampleData(self, upsample):
        """
        Resamples range bins if upsampling.
        :param upsample: Upsample factor.
        :return: Updated range bins using upsample.
        """
        self.upsample = upsample
        MPP = c0 / fs / upsample
        self.upsample_nsam = self.nsam * upsample
        range_bins = self.near_slant_range + np.arange(self.upsample_nsam) * MPP / 2
        self.range_bins = range_bins
        self.fft_len = self.fft_len * upsample
        self.fft_chirp = np.fft.fft(self.chirp, self.fft_len)
        self.genMF(upsample)
        return range_bins

    def getAllPos(self):
        """
        Gets all positions of the radar at all pulse times.
        :return: numpy array of radar positions at pulse times.
        """
        return self.pos(self.times)

    def genChirp(self, px=None, py=None):
        """
        Generates an arbitrary phase chirp based on given params. Integrates over the
            phase function defined by px and py to create the complex chirp.
        :param px: numpy array of time parameters. Should be from zero to one and same size as py.
        :param py: numpy array of phase parameters. Should be from zero to one and same size as px.
        :return: chirp function.
        """
        if py is None:
            px = np.linspace(0, 1, 10)
            py = np.linspace(0, 1, 10)
        chirp = genPulse(px, py, self.nr, self.params['Pulse_Length_S'], 0, self.bandwidth)
        r_chirp = np.zeros((self.fft_len,), dtype=np.complex128)
        r_chirp[:len(chirp)] = chirp
        self.chirp = r_chirp
        self.fft_chirp = np.fft.fft(self.chirp, self.fft_len)
        self.genMF()
        return self.chirp

    def genMF(self, upsample=1):
        """
        Generates a matched filter for the saved chirp, windowing it and zeroing out the
            stopband.
        :param upsample: int. Upsample factor.
        :return: Nothing.
        """
        upsample_nr = self.nr * upsample
        mf_fft = np.fft.fft(self.chirp, self.fft_len).conj().T
        tay = taylor(upsample_nr)
        mf_fft[:upsample_nr // 2] = mf_fft[:upsample_nr // 2] * tay[-upsample_nr // 2:]
        mf_fft[-upsample_nr // 2:] = mf_fft[-upsample_nr // 2:] * tay[:upsample_nr // 2]
        mf_fft[upsample_nr // 2:-upsample_nr // 2] = 0
        self.mf = mf_fft

    def loadChirp(self, chirp):
        """
        Loads a chirp from a numpy array.
        :param chirp: Numpy array. This is chirp data.
        :return: Nothing. Sets a lot of member variables.
        """
        r_chirp = np.zeros((self.fft_len,), dtype=np.complex128)
        r_chirp[:len(chirp)] = chirp
        self.chirp = r_chirp
        self.fft_chirp = np.fft.fft(self.chirp, self.fft_len)
        self.genMF()
        self.t0 = len(self.chirp) / fs
        self.nr = len(self.chirp)

    def loadMF(self, mf):
        """
        Loads a matched filter if you want something other than the generated one.
        :param mf: Numpy array. Matched filter data to use.
        :return: Nothing. Sets the saved matched filter.
        """
        self.mf = mf

    def loadChirpFromFile(self, fnme):
        """
        Loads a waveform from APS files.
        :param fnme: Str. Path to APS waveform file.
        :return: Nothing. Sets a lot of member variables.
        """
        with open(fnme, 'rb') as fid:
            num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
            real_pulse = np.fromfile(fid, 'float32', num_frames, '')
        complex_pulse = hilbert(real_pulse)
        complex_pulse = np.fft.ifft(np.fft.fft(complex_pulse, findPowerOf2(len(real_pulse))), n=num_frames // 2)
        self.chirp = complex_pulse
        self.fft_chirp = np.fft.fft(self.chirp, self.fft_len)
        self.mf = np.fft.fft(self.chirp, self.fft_len).conj().T


class RadarArray(object):
    def __init__(self, radars=None, names=None):
        """
        Container class for Radar objects. Useful for MIMO arrays.
        :param radars: list of Radar objects.
        :param names: list of str. If you want to name your radars.
        """
        if radars is None:
            radars = []
            names = []
        elif names is None:
            names = [n for n in range(len(radars))]
        self.radars = radars
        self.names = names
        self.range_bins = None
        self.systimes = self.times = None
        self.is_presummed = False
        self.upsample_nsam = None
        self.num = 0
        self.pos = None
        self.pan = None
        self.tilt = None

    def append(self, li, name=None):
        """
        Adds a radar to the internal list.
        :param li: Radar object. The object to add to the list.
        :param name: str. Name of radar object, should it be desired.
        :return: Nothing. Adds it to a member variable.
        """
        if len(self.radars) == 0:
            self.range_bins = li.range_bins
            self.systimes = li.systimes
            self.upsample_nsam = li.upsample_nsam
            self.times = li.times
            self.is_presummed = li.is_presummed
            self.pan = li.pan
            self.tilt = li.tilt
        self.radars.append(li)
        if name is None:
            name = self.num
        self.names.append(name)
        self.num += 1

    def calcFactors(self):
        self.pos = lambda t: np.mean([r.pos(t) for r in self.radars], axis=0)

    def __iter__(self):
        for r in self.radars:
            yield r

    def split(self):
        for r in range(len(self.radars)):
            yield self.radars[r], [self.radars[d] for d in range(len(self.radars)) if d != r]


@numba.njit
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
    A = np.fft.fftshift(np.fft.ifft(s2f * s1f[:, None], axis=0),
                        axes=0)[fft_sz // 2 - dopp_bins // 2: fft_sz // 2 + dopp_bins // 2]
    if mag:
        return abs(A / abs(A).max()) ** 2, fdopp, np.linspace(-len(s1) / 2 / fs, len(s1) / 2 / fs, len(s1))
    else:
        return A / abs(A).max(), fdopp, np.linspace(-dopp_bins / 2 / fs, dopp_bins / 2 / fs, dopp_bins)
