
import pylab as plab
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.signal.windows import taylor
from scipy.spatial.transform import Rotation as rot
import numpy as np

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

    def __init__(self, fnme, llr, use_mean_pantilt=False, use_xml_flightpath=False, presum=None):
        files = findAllFilenames(fnme)
        xml_params = rp.loadXMLFile(files['xml'])
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
            re = lambda t: (ee - se) * t * TAC / systime[-1] + se  # CubicSpline(filt_times, trip_e)
            rn = lambda t: (en - sn) * t * TAC / systime[-1] + sn  # CubicSpline(filt_times, trip_n)
            ru = lambda t: h_agl * np.ones((len(t),)) \
                if type(t) != np.float64 else h_agl
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
                                  apsdata['alt'].values - sl.undulationEGM96(apsdata['lat'].values, apsdata['lon'].values))
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
                ant_offsets = np.array([params['Antenna_X_Offset_M'],
                                        params['Antenna_Y_Offset_M'],
                                        params['Antenna_Z_Offset_M']])

                # Convert gimbal angles to rotations
                Rmg = [rot.from_rotvec([row['gimbal_tilt'], 0, row['gimbal_pan']])
                       for idx, row in apsdata.iterrows()]

                # Apply rotations through antenna frame, gimbal frame, and add to gimbal offsets
                gamma_b_gpc = [(Rgb * n).inv().apply(ant_offsets).flatten() + gimbaloffsets for n in Rmg]

                # Rotate gimbal/antenna offsets into inertial frame
                rotated_offsets = np.array([Rbi[i].inv().apply(gamma_b_gpc[i]).flatten() for i in range(apsdata.shape[0])])

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
                apsdata['e'] = e
                apsdata['n'] = n
                apsdata['u'] = u
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
        # self.supported_pulses = min(np.ceil(gps_times[-1] / (params['PRI_TAC'] / TAC)).astype(int),
        #                             nframes)
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

    def resampleRangeBins(self, upsample):
        MPP = c0 / fs / upsample
        self.upsample_nsam = self.nsam * upsample
        range_bins = self.near_slant_range + np.arange(self.upsample_nsam) * MPP / 2
        self.range_bins = range_bins
        return range_bins

    def getAllPos(self):
        return self.pos(self.times)

    def chirp(self, px=None, py=None):
        if py is None:
            px = np.linspace(0, 1, 10)
            py = np.linspace(0, 1, 10)
        return genPulse(px, py, self.nr, self.params['Pulse_Length_S'], 0, self.bandwidth)
