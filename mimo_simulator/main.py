"""
Main script for MIMO simulation

Should take the various objects and run them
in the Environment to create
simulated data in the APS debug output
or SAR file style.
"""
import math
import sys

import cupy as cupy
import numpy as np
from numpy import ndarray
from numpy.fft import fft, ifft, fftshift
from scipy.io import loadmat
from scipy.stats import trapezoid
from scipy.stats.mstats import gmean
from scipy.signal import stft
from scipy.ndimage import sobel, gaussian_filter
from tqdm import tqdm
import imageio
from itertools import combinations

from cuda_kernels import genRangeProfile, backproject
from environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from pathlib import Path

from radar import ambiguity, genPulse, XMLPlatform, SimPlatform, XMLChannel, \
    SimChannel, Antenna, getTVSignal, getWidebandSignal
from rawparser import loadReferenceChirp, loadMatchedFilter, loadASHFile, getRawDataGen, getRawSDRParams
from simlib import getElevation, enu2llh
from useful_lib import findAllFilenames, factors, db, findPowerOf2, gaus
from SARParse import DebugWriter, SDRParse, SDRWriter


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def genPoints(n_pts, e, offsets, method='uniform'):
    np.random.seed(666)
    _gx = 0
    _gy = 0
    if n_pts == 1:
        _gx = 0 + offsets[0]
        _gy = 0 + offsets[1]
        _gz, _gv = e(_gx, _gy)
        _gv = 1e9
    else:
        if method == 'uniform':
            _gx = np.random.rand(n_pts) * e.shape[0] - e.shape[0] / 2 + offsets[0]
            _gy = np.random.rand(n_pts) * e.shape[1] - e.shape[1] / 2 + offsets[1]
            _gz, _gv = e(_gx, _gy)
            while np.any(_gv == 0):
                bad_len = sum(_gv == 0)
                _gx[_gv == 0] = np.random.rand(bad_len) * e.shape[0] - e.shape[0] / 2 + offsets[0]
                _gy[_gv == 0] = np.random.rand(bad_len) * e.shape[1] - e.shape[1] / 2 + offsets[1]
                _gz[_gv == 0], _gv[_gv == 0] = e(_gx[_gv == 0], _gy[_gv == 0])
        elif method == 'gauss':
            _gx = np.random.normal(0, e.shape[0] / 2, n_pts) + offsets[0]
            while np.any(abs(_gx) > e.data_shape[0]):
                _gx[abs(_gx) > e.data_shape[0]] = np.random.normal(0, e.shape[0] / 2, sum(abs(_gx) > e.data_shape[0])) \
                                                  + offsets[0]
            _gy = np.random.normal(0, e.shape[1] / 2, n_pts) + offsets[1]
            while np.any(abs(_gy) > e.data_shape[1]):
                _gy[abs(_gy) > e.data_shape[1]] = np.random.normal(0, e.shape[1] / 2, sum(abs(_gy) > e.data_shape[1])) \
                                                  + offsets[1]
        elif method == 'trap':
            _gx = trapezoid.rvs(0.2, .8, size=n_pts) * (e.shape[0] + e.shape[0] * .4) - \
                  (e.shape[0] + e.shape[0] * .4) / 2 + offsets[0]
            _gy = trapezoid.rvs(0.2, .8, size=n_pts) * (e.shape[1] + e.shape[1] * .4) - \
                (e.shape[1] + e.shape[1] * .4) / 2 + offsets[1]
        elif method == 'contrast':
            gx_tmp = np.random.rand(n_pts) * e.shape[0] - e.shape[0] / 2 + offsets[0]
            gy_tmp = np.random.rand(n_pts) * e.shape[1] - e.shape[1] / 2 + offsets[1]
            tsd = gaussian_filter(sobel(e.data / np.max(e.data)), sigma=10)
            grads = np.gradient(tsd)

            _gx = gx_tmp + env(gx_tmp, gy_tmp, override_data=grads[0])[1]  # + np.random.rand(n_pts) - .5
            _gy = gy_tmp + env(gx_tmp, gy_tmp, override_data=grads[1])[1]  # + np.random.rand(n_pts) - .5

        _gz, _gv = e(_gx, _gy)
    return _gx, _gy, _gz, _gv


''' CONSTANTS '''
c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
MAX_SAMPLE_SIZE = 1e6
INT_16_MAX = 2 ** 15
plt.close('all')

''' FILENAMES '''
fnme = '/data5/SAR_DATA/2021/09222021/SAR_09222021_163338.sar'
output_dir = '/home/jeff/repo/mimo_simulator/'
cust_bg = '/home/jeff/Downloads/josh.png'

''' CUSTOM OPTIONS '''
threads_per_block = (16, 16)
upsample = 1
chunk_sz = 256
n_samples = 2000000
presum = 1
noise_level = 0.1
poly_interp = 1
channel = 0
att_value = 5
n_rfi_pts = 2
rfi_perc = .35
wb_rfi_pts = 1
wrfi_perc = .15
subgrid_size = (255, 255)
bpj_size = (150, 150)
rand_method = 'contrast'
scp = (40.098785, -111.659957)

''' FLAGS '''
do_backproject = True
introduce_rfi = False
write_to_file = False
use_background_image = True
use_config_radar = False
display_figures = True

# Load in files
files = findAllFilenames(fnme, debug_dir='/data5/SAR_Freq_Data')
scp_el = (scp[0], scp[1], getElevation(scp))

# Initialize all the variables
match_filt = None
gpx_gpu = gpy_gpu = gpz_gpu = gpv_gpu = None
rfix_gpu = rfiy_gpu = rfiz_gpu = rfip_gpu = None
rfi_sigs = rfi_ts = None
n_splits = rcd_gpu = rpf_gpu = None
check_tt = check_pnum = sdr_w = None
gx = gy = gz = gv = None
rng_ax = bpj_ax = dopp_ax = slice_ax = pulse_ax = None
att_rng = 10 ** (np.arange(1, 32) / 20)

# Make the user aware of everything that's going on
if introduce_rfi:
    print('Introducing {} RFI points.'.format(n_rfi_pts))
if do_backproject:
    print('Backprojection enabled.')
else:
    print('Backprojection disabled.')
if write_to_file:
    print('Results will be saved to file.')
else:
    print('No files will be written.')

print('Loading environment...')
if use_background_image:
    print('Using custom background image.')
    im = imageio.imread(cust_bg)
    crop_im = np.flipud(np.sum(im, axis=2))
    env = Environment(llr=scp_el, bg_data=crop_im, subgrid_size=subgrid_size, dec_fac=1)
else:
    print('Using .asi image.')
    if type(files['ash']) == dict:
        kkk = list(files['ash'])[0]
        ash_fnme = files['ash'][kkk]
        asi_fnme = files['asi'][kkk]
    else:
        ash_fnme = files['ash']
        asi_fnme = files['asi']
    env = Environment(llr=scp_el, ashfile=ash_fnme, asifile=asi_fnme, subgrid_size=subgrid_size, dec_fac=8)
shift_gpu = cupy.array(np.ascontiguousarray([[0, 0]]), dtype=np.float64)
scp_offsets = env.getOffset(*scp, False)

print('Loading radar list...')
# List out the radar chirp and relative position from center of array for each antenna/channel
# Custom waveforms:
# (np.linspace(0, 1, 100), np.array([0, 0, 0]))
# Load in from Strider:
# ('/home/jeff/repo/mimo_simulator/waveforms/p4_400_300.wave', np.array([10, 1, 15]))
sdr_f = SDRParse(files['sar'])
params = {'Flight_Line_Altitude_M': 1524.00,
          'Gimbal_Settings': {'Gimbal_Depression_Angle_D': 45.00, 'Gimbal_X_Offset_M': -.3461,
                                'Gimbal_Y_Offset_M': 1.3966, 'Gimbal_Z_Offset_M': -1.2522, 'Roll_D': 0.00,
                                'Pitch_D': 0.00, 'Yaw_D': -90.00},
          'Antenna_Settings': {'Antenna_0':
                                   {'Antenna_X_Offset_M': -.2241, 'Antenna_Y_Offset_M': 0.0,
                                    'Antenna_Z_Offset_M': .1344, 'Azimuth_Beamwidth_D': 27.72,
                                    'Elevation_Beamwidth_D': 30.00, 'Doppler_Beamwidth_D': 40.00,
                                    'Antenna_Depression_Angle_D': 45.00},
                               'Antenna_1':
                                   {'Antenna_X_Offset_M': -.2241, 'Antenna_Y_Offset_M': 1.0,
                                    'Antenna_Z_Offset_M': .1344, 'Azimuth_Beamwidth_D': 27.72,
                                    'Elevation_Beamwidth_D': 30.00, 'Doppler_Beamwidth_D': 40.00,
                                    'Antenna_Depression_Angle_D': 45.00}
                               }}
gimbal = sdr_f.gimbal
gimbal['pan'] = gimbal['pan'] + np.linspace(-1, 1, gimbal.shape[0])**2 / 100
# ra = SimPlatform(env.scp, sdr_f.gps_data, gimbal, params)
ra = XMLPlatform(fnme, env.scp, sarfile=sdr_f)
ra_pos = [(np.linspace(0, 1, 100), XMLChannel(sdr_f.xml['Channel_0'], sdr_f[0], env.scp, ra.alt, 0, presum=1)),
          (np.linspace(1, 0, 100), SimChannel(env.scp, ra, 1500e6, 10e9, 50.0, 40.0, 15.00, trans_port=1, rec_port=1))]
for idx, sr in enumerate(ra_pos):
    ra.addChannel(sr[1])
    try:
        ra[idx].genChirp(px=np.linspace(0, 1, 100), py=sr[0])
        # ra[idx].loadChirp(sdr_f[idx].ref_chirp)
    except ValueError:
        ra[idx].loadChirpFromFile(sr[0])

# Make sure to upsample in case our backprojection is wanted
ra.upsample(upsample)

# Get all the stuff associated with the overall array
n_frames = ra[0].nframes
rc_data = [None for n in ra]
range_prof = [None for n in ra]
pulses = [None for n in ra]

# Load memory for CUDA processing
mempool = cupy.get_default_memory_pool()

# Get grid for debug testing
bx, by = np.meshgrid(np.linspace(-bpj_size[0] // 2, bpj_size[0] // 2, bpj_size[0] * 4),
                     np.linspace(-bpj_size[1] // 2, bpj_size[1] // 2, bpj_size[1] * 4))
gs = bx.shape
bx += scp_offsets[0]
by += scp_offsets[1]
bz, _ = env(bx, by)
bx_gpu = cupy.array(bx, dtype=np.float64)
by_gpu = cupy.array(by, dtype=np.float64)
bz_gpu = cupy.array(bz, dtype=np.float64)
blocks_per_grid_bpj = (
    int(np.ceil(bx.shape[0] / threads_per_block[0])), int(np.ceil(bx.shape[1] / threads_per_block[1])))

# Write header stuff to .dat file
init_time = 0
if write_to_file:
    # Get name of file
    write_fnme = fnme.split('/')[-1].split('.')[0] + 'SIM'
    sdr_w = SDRWriter(fpath=output_dir, fnme=write_fnme)
    sdr_w.addCommonSettings(file=sdr_f.xml['Common_Channel_Settings'])
    for ch in sdr_f.xml['Common_Channel_Settings']['Antenna_Settings']:
        ant = sdr_f.xml['Common_Channel_Settings']['Antenna_Settings'][ch]
        sdr_w.addAntenna((ant['Antenna_X_Offset_M'], ant['Antenna_Y_Offset_M'], ant['Antenna_Z_Offset_M']),
                         ant['Azimuth_Beamwidth_D'], ant['Elevation_Beamwidth_D'], ant['Doppler_Beamwidth_D'],
                         ant['Antenna_Depression_Angle_D'])
    sdr_w.addChannel(ra[0].fc, ra[0].bandwidth, ra[0].tp, ra[0].prf, ra[0].near_angle, ra[0].far_angle, ra[0].plp, 1,
                     trans_port=ra[0].tx_ant, rec_port=ra[0].rx_ant)
    fl = sdr_f.xml['Flight_Line']
    sdr_w.addFlightLineInfo((fl['Start_Latitude_D'], fl['Start_Longitude_D']),
                            (fl['Stop_Latitude_D'], fl['Stop_Longitude_D']))
    gim = sdr_f.xml['Common_Channel_Settings']['Gimbal_Settings']
    sdr_w.addGimbal(gim['Pan_Limits_D'], gim['Tilt_Limits_D'], gim['Gimbal_Depression_Angle_D'],
                    (gim['Gimbal_X_Offset_M'], gim['Gimbal_Y_Offset_M'], gim['Gimbal_Z_Offset_M']),
                    (gim['Roll_D'], gim['Pitch_D'], gim['Yaw_D']), gim['Initial_Course_Angle_R'])
    sdr_w.writeXML()
    sdr_w.addGPSData(sdr_f.gps_data, sdr_f.bestpos, sdr_f.timesync, sdr_f.gimbal)
    # Write cal data
    cal_chirp = ra[0].chirp
    cal_chirp = cal_chirp / abs(cal_chirp).max() * INT_16_MAX
    for nn in range(sdr_f[0].ncals):
        sdr_w.write(sdr_f.getPulse(nn, 0, is_cal=True), sdr_f[0].cal_time[nn], 31, 0, is_cal=True)
    init_time = sdr_f[0].sys_time[0]

# Split samples into smaller chunks to fit onto GPU
if n_samples > MAX_SAMPLE_SIZE:
    for fac in np.sort(factors(n_samples)):
        if n_samples / fac <= MAX_SAMPLE_SIZE:
            n_splits = int(fac)
            break
    samp_split = int(n_samples / n_splits)
else:
    n_splits = 1
    samp_split = n_samples
gx, gy, gz, gv = genPoints(samp_split, env, scp_offsets, method=rand_method)
gpx_gpu = cupy.array(gx, dtype=np.float64)
gpy_gpu = cupy.array(gy, dtype=np.float64)
gpz_gpu = cupy.array(gz, dtype=np.float64)
gpv_gpu = cupy.array(gv, dtype=np.float64)

# RFI point initialization
if introduce_rfi:
    p_hasRFI = np.zeros((n_frames,))
    tmp_prcs = np.random.rand(n_rfi_pts)
    if wb_rfi_pts > 0:
        tmp_wb_prcs = np.random.rand(wb_rfi_pts)
        rfi_ts = np.concatenate((tmp_prcs / sum(tmp_prcs) * rfi_perc, tmp_wb_prcs / sum(tmp_wb_prcs) * wrfi_perc))
    else:
        rfi_ts = tmp_prcs / sum(tmp_prcs) * rfi_perc
    rfi_sigs = []
    for sig in range(n_rfi_pts):
        rfi_sigs.append((0, 1e2, np.random.randint(10, 70)))
    for sig in range(wb_rfi_pts):
        rfi_sigs.append((1, 1e2, np.random.rand() * 2e9, np.random.rand() * 200e6))
else:
    p_hasRFI = None

# Now run pulse gen/backprojection for each of the antennae
print('Running range profile generation...')
overall_bpj = np.zeros(bx.shape, dtype=np.complex128)
# Get minimum slant range (which should be the center of our aperture)
pca_rngs = np.linalg.norm(ra.pos(ra.times), axis=0)
t_pca = ra.times[0]
try:
    # noinspection PyArgumentList
    t_pca = ra.times[pca_rngs == pca_rngs.min()][0]
except IndexError:
    t_pca = t_pca[0]
R0 = pca_rngs.min()

'''
#################### MAIN LOOP ########################
'''

bpjgrid = [np.zeros(bx.shape, dtype=np.complex128) for r in ra]
for rx_num, chan in enumerate(ra):
    for ch in tqdm(np.arange(0, chan.nframes, chunk_sz)):
        tt = chan.systimes[ch:min(ch + chunk_sz, chan.nframes)] / TAC
        ch_sz = len(tt)

        # Figure out our memory scheme
        rpf_shape = (chan.nsam, ch_sz)
        blocks_per_grid_rpf = (
            int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(n_samples / threads_per_block[1])))
        rcd_gpu = rpf_gpu = None
        rb_gpu = cupy.array(chan.range_bins, dtype=np.float64)
        uprb_gpu = cupy.array(chan.upsample_rbins, dtype=np.float64)
        rx = ra.rx(channel=rx_num)

        # Load receiver antenna data, since that will not change
        pd_gpu = cupy.zeros(rpf_shape, dtype=np.complex128)
        ref_gpu = cupy.array(np.tile(chan.fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
        mf_gpu = cupy.array(np.tile(chan.mf, (ch_sz, 1)).T, dtype=np.complex128)

        # Toss in all the interpolated stuff
        fprx_gpu = cupy.array(np.ascontiguousarray(rx.pos(tt), dtype=np.float64))
        panrx_gpu = cupy.array(np.ascontiguousarray(rx.pan(tt), dtype=np.float64))
        tiltrx_gpu = cupy.array(np.ascontiguousarray(rx.tilt(tt), dtype=np.float64))

        # Load constants and other parameters
        paramrx_gpu = cupy.array(np.array([np.pi / rx.el_bw, np.pi / rx.az_bw,
                                           chan.wavelength, 0, rx.el_bw, rx.az_bw,
                                           chan.near_slant_range / c0, fs * upsample, poly_interp]), dtype=np.float64)

        for tx_num, tx_chan in enumerate(ra):
            tx = ra.tx(channel=tx_num)
            fptx_gpu = cupy.array(np.ascontiguousarray(tx.pos(tt), dtype=np.float64))
            pantx_gpu = cupy.array(np.ascontiguousarray(tx.pan(tt), dtype=np.float64))
            tilttx_gpu = cupy.array(np.ascontiguousarray(tx.tilt(tt), dtype=np.float64))

            # Load constants and other parameters
            paramtx_gpu = cupy.array(np.array([np.pi / tx.el_bw, np.pi / tx.az_bw,
                                               tx_chan.wavelength, 0, tx.el_bw, tx.az_bw,
                                               tx_chan.near_slant_range / c0, fs * upsample, 1]), dtype=np.float64)
            ref_gpu = cupy.array(np.tile(tx_chan.fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
            rpf_r_gpu = cupy.random.normal(0, noise_level, rpf_shape, dtype=np.float64)
            rpf_i_gpu = cupy.zeros_like(rpf_r_gpu)
            # rpf_i_gpu = cupy.random.normal(0, noise_level, rpf_shape, dtype=np.float64)

            for rnd in range(n_splits):
                if rnd >= 1:
                    # Load grid coordinates onto GPU
                    gx, gy, gz, gv = genPoints(samp_split, env, scp_offsets, method=rand_method)
                    gpx_gpu = cupy.array(gx, dtype=np.float64)
                    gpy_gpu = cupy.array(gy, dtype=np.float64)
                    gpz_gpu = cupy.array(gz, dtype=np.float64)
                    gpv_gpu = cupy.array(gv, dtype=np.float64)
                blocks_per_grid_rpf = (
                    int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(samp_split / threads_per_block[1])))

                # Run range profile generation
                genRangeProfile[blocks_per_grid_rpf, threads_per_block](fptx_gpu, fprx_gpu, gpx_gpu, gpy_gpu, gpz_gpu,
                                                                        gpv_gpu, panrx_gpu, pantx_gpu,
                                                                        tiltrx_gpu, tilttx_gpu, rpf_r_gpu, rpf_i_gpu,
                                                                        shift_gpu,
                                                                        paramrx_gpu, paramtx_gpu)
                cupy.cuda.Device().synchronize()

            # Calculate the pulse data on the GPU using FFT
            rpf_gpu = rpf_r_gpu + 1j * rpf_i_gpu

            if introduce_rfi:
                # Create RFI signals, both wide and narrow band
                rbi_gpu = cupy.zeros((chan.nsam, ch_sz), dtype=np.complex128)
                nbi_corrupt = np.zeros((ch_sz,))
                nbi_signal = np.zeros((chan.nsam, ch_sz), dtype=np.complex128)
                for idx, sig in enumerate(rfi_sigs):
                    isoff = np.random.rand(ch_sz) > rfi_ts[idx]
                    for npulse in range(ch_sz):
                        if not isoff[npulse]:
                            if sig[0] == 0:
                                nbi_signal[:, npulse] = sig[1] * getTVSignal(sig[2], chan.nsam)
                            elif sig[0] == 1:
                                nbi_signal[:, npulse] = sig[1] * getWidebandSignal(sig[2], sig[3], chan.nsam)
                    nbi_corrupt += np.logical_not(isoff)
                    rbi_gpu = rbi_gpu + cupy.array(nbi_signal, dtype=np.complex128)

                rbi_gpu = cupy.fft.fft(rbi_gpu, axis=0, n=tx_chan.fft_len)

                # Add RFI signals to the data generation FFTs
                pd_gpu = pd_gpu + cupy.fft.ifft(
                    cupy.fft.fft(rpf_gpu, n=tx_chan.fft_len, axis=0) * ref_gpu + rbi_gpu,
                    axis=0)[:chan.nsam, :]

                p_hasRFI[ch:ch + ch_sz] = nbi_corrupt > 0

                # Delete the GPU RFI stuff
                del rbi_gpu
            else:
                pd_gpu = pd_gpu + cupy.fft.ifft(
                    cupy.fft.fft(rpf_gpu, n=tx_chan.fft_len, axis=0) * ref_gpu, axis=0)[:chan.nsam, :]
            cupy.cuda.Device().synchronize()

            del rpf_r_gpu
            del rpf_i_gpu
            del fptx_gpu
            del ref_gpu

        # Run the actual backprojection
        if do_backproject:
            inter_gpu = cupy.fft.fft(pd_gpu, n=chan.fft_len, axis=0) * mf_gpu
            ifft_gpu = cupy.zeros((chan.fft_len * upsample, ch_sz), dtype=np.complex128)
            ifft_gpu[:chan.fft_len // 2 + 1, :] = inter_gpu[:chan.fft_len // 2 + 1, :]
            ifft_gpu[-chan.fft_len // 2:, :] = inter_gpu[-chan.fft_len // 2:, :]
            rcd_gpu = cupy.fft.ifft(ifft_gpu, axis=0)[:chan.upsample_nsam, :]
            bpjgrid_gpu = cupy.zeros(bx.shape, dtype=np.complex128)
            backproject[blocks_per_grid_bpj, threads_per_block](fprx_gpu, fprx_gpu, bx_gpu, by_gpu, bz_gpu,
                                                                uprb_gpu, panrx_gpu,
                                                                tiltrx_gpu, rcd_gpu, bpjgrid_gpu,
                                                                paramrx_gpu)
            cupy.cuda.Device().synchronize()
            bpjgrid[rx_num] = bpjgrid[rx_num] + bpjgrid_gpu.get()
            del bpjgrid_gpu
            del ifft_gpu
            del inter_gpu

        if write_to_file:
            # Write to .dat files for backprojection
            pdata = pd_gpu.get()
            for n in range(ch_sz):
                # Rescale everything so it fits into an int16
                scale = abs(pdata[:, n]).max() / INT_16_MAX
                try:
                    att_value = np.where(att_rng > scale)[0][0]
                except IndexError:
                    att_value = 30
                sdr_w.write(pdata[:, n], int(init_time + tt[n] * TAC), att_value, 0, False)

        # Find the point of closest approach and get all stuff associated with it
        if tt[-1] >= t_pca >= tt[0]:
            if do_backproject:
                rc_data[rx_num] = rcd_gpu.get()
            pulses[rx_num] = pd_gpu.get()
            range_prof[rx_num] = rpf_gpu.get()
            check_tt = tt
            check_pnum = np.arange(ch, ch + ch_sz)

        # Delete the range compressed pulse block to free up memory on the GPU
        del rpf_gpu
        del pd_gpu
        del rcd_gpu
        del panrx_gpu
        del tiltrx_gpu
        del fprx_gpu
        del mf_gpu
        mempool.free_all_blocks()

# Delete all of our other parameters and free the GPU memory
del paramrx_gpu
del gpx_gpu
del gpy_gpu
del gpz_gpu
del gpv_gpu
del rb_gpu
del uprb_gpu
mempool.free_all_blocks()

pca_idx = np.where(check_tt == check_tt.min())[0][0]

# After run, diagnostics, etc.
if write_to_file and introduce_rfi:
    rfi_fnme = sdr_w.f_sar[:-4] + '_rfi.dat'
    if not Path(rfi_fnme).exists():
        Path(rfi_fnme).touch()
    else:
        with open(rfi_fnme, 'wb') as f:
            f.write(np.int8(p_hasRFI).tobytes())
    print('File saved to ' + rfi_fnme)

'''
################### PLOTS AND FIGURES ##########################
'''
# Generate all the plot figures we'll need
if display_figures:
    if introduce_rfi:
        print(f'RFI produced on {sum(p_hasRFI) / ra[0].nframes * 100:.2f}% of pulses.')
    grid_num = 1
    if ra.n_channels < 4:
        if do_backproject:
            bpj_fig, bpj_ax = plt.subplots(1, ra.n_channels, num='Backproject')
            dopp_fig, dopp_ax = plt.subplots(1, ra.n_channels, num='Doppler Shifted')
            rng_fig, rng_ax = plt.subplots(1, ra.n_channels, num='Range Profiles')
    else:
        grid_num = int(np.ceil(np.sqrt(ra.n_channels)))
        if do_backproject:
            bpj_fig, bpj_ax = plt.subplots(grid_num, grid_num, num='Backproject')
            dopp_fig, dopp_ax = plt.subplots(grid_num, grid_num, num='Doppler Shifted')
            rng_fig, rng_ax = plt.subplots(grid_num, grid_num, num='Range Profiles')
    param_fig, param_ax = plt.subplots(2, 1, num='Params')
    chirp_fig, chirp_ax = plt.subplots(4, ra.n_channels, num='Chirp Data')
    if do_backproject:
        slice_fig, slice_ax = plt.subplots(3, ra.n_channels, num='Slices')
        pulse_fig, pulse_ax = plt.subplots(2, ra.n_channels, num='Pulse Data')

    for ant_num, r_rx in enumerate(ra):
        rx = ra.rx(ant_num)
        print('Channel {}'.format(ant_num))
        pos_pca = enu2llh(*rx.pos(t_pca), env.scp)
        pca_slant_range = np.linalg.norm(rx.pos(t_pca))
        print('PCA-SCP slant range is {:.2f}'.format(pca_slant_range))
        print('PCA-SCP ground range is {:.2f}'.format(np.sqrt(pca_slant_range ** 2 - (pos_pca[2] - env.scp[2]) ** 2)))
        print('PCA radar pan is {:.2f}'.format(rx.pan(t_pca) / DTR))
        print('PCA radar tilt is {:.2f}'.format(rx.tilt(t_pca) / DTR))
        print('Plane pos at PCA is {:.6f}, {:.6f}, {:.2f}'.format(*pos_pca))
        sh_los = -rx.pos(ra.times)
        rngs = np.linalg.norm(sh_los, axis=0)
        pt_el = np.array([math.asin(-sh_los[2, i] / rngs[i]) for i in range(len(rngs))])
        pt_az = np.array([math.atan2(sh_los[0, i], sh_los[1, i]) for i in range(len(rngs))])
        ant_x_plot = ant_num // grid_num
        ant_y_plot = ant_num % grid_num

        az_diffs = np.array([adiff(pt_az[i], rx.pan(ra.times[i])) for i in range(len(rngs))])
        param_ax[0].set_title('Ranges')
        param_ax[0].plot(ra.times, rngs)
        param_ax[0].plot(ra.times[np.logical_and(ra.times <= check_tt[-1], ra.times >= check_tt[0])],
                         rngs[np.logical_and(ra.times <= check_tt[-1], ra.times >= check_tt[0])])
        param_ax[1].set_title('Az Diffs')
        param_ax[1].plot(ra.times, az_diffs)
        param_ax[1].plot(ra.times[np.logical_and(ra.times <= check_tt[-1], ra.times >= check_tt[0])],
                         az_diffs[np.logical_and(ra.times <= check_tt[-1], ra.times >= check_tt[0])])

        # plt.figure('Chirp Data Ant {}'.format(ant_num))
        try:
            chirp_ax[0, ant_num].set_title('Ref. Chirp')
            chirp_ax[0, ant_num].plot(np.real(r_rx.chirp))
            chirp_ax[1, ant_num].set_title('Ref. Spectrum')
            chirp_ax[1, ant_num].plot(db(fft(r_rx.chirp)))
            chirp_ax[2, ant_num].set_title('Matched Filter')
            chirp_ax[2, ant_num].plot(db(r_rx.mf))
            chirp_ax[3, ant_num].set_title('Range Compression')
            chirp_ax[3, ant_num].plot(r_rx.upsample_rbins,
                                      np.fft.fftshift(db(ifft(r_rx.fft_chirp * r_rx.mf, n=r_rx.upsample_nsam))))
        except IndexError:
            chirp_ax[0].set_title('Ref. Chirp')
            chirp_ax[0].plot(np.real(r_rx.chirp))
            chirp_ax[1].set_title('Ref. Spectrum')
            chirp_ax[1].plot(db(fft(r_rx.chirp)))
            chirp_ax[2].set_title('Matched Filter')
            chirp_ax[2].plot(db(r_rx.mf))
            chirp_ax[3].set_title('Range Compression')
            chirp_ax[3].plot(r_rx.upsample_rbins,
                             np.fft.fftshift(db(ifft(r_rx.fft_chirp * r_rx.mf, r_rx.upsample_nsam))))

        # Apply range roll-off corrections
        if do_backproject and abs(bpjgrid[ant_num]).max() > 0:
            dpshift_data = db(fftshift(fft(rc_data[ant_num], axis=1), axes=1))
            med_curve = np.median(abs(bpjgrid[ant_num]), axis=0)
            med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
            rng_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
            med_curve = np.median(abs(bpjgrid[ant_num]), axis=1)
            med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
            az_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
            for az in range(len(az_corr)):
                for rng in range(len(rng_corr)):
                    bpjgrid[ant_num][az, rng] = bpjgrid[ant_num][az, rng] * rng_corr[rng] * az_corr[az]
            # plt.figure('Range Profile Ant {}'.format(ant_num))
            if ra.n_channels < 4:
                try:
                    rng_ax[ant_num].imshow(
                        np.real(range_prof[ant_num][:np.arange(
                            len(r_rx.upsample_rbins))[r_rx.upsample_rbins < rngs.max()][-1] + 5]))
                    rng_ax[ant_num].axis('tight')
                    # plt.figure('Backproject Ant {}'.format(ant_num))
                    db_bpj = db(bpjgrid[ant_num]).T
                    bpj_ax[ant_num].imshow(db_bpj, origin='lower', cmap='gray',
                                           clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])
                    bpj_ax[ant_num].axis('off')

                    # Calculate out image metrics
                    met_bpj = ((db_bpj - db_bpj.mean()) / db_bpj.std() * env.data.std() + env.data.mean())
                    dbj_hist = np.histogram(met_bpj, bins=env.bins)
                    dbj_hist = dbj_hist[0].astype(float)
                    dbj_hist[dbj_hist == 0] = 1e-9
                    print('Antenna {}: HFM {:.2f} HS {:.2f}'.format(
                        ant_num, gmean(dbj_hist) / dbj_hist.mean(),
                        (np.percentile(dbj_hist, 75) - np.percentile(dbj_hist, 25)) /
                        (dbj_hist.max() - dbj_hist.min())))

                    # plt.figure('Doppler Shifted Data Ant {}'.format(ant_num))
                    dopp_ax[ant_num].imshow(dpshift_data)
                    dopp_ax[ant_num].axis('tight')
                except TypeError:
                    rng_ax.imshow(
                        np.real(range_prof[ant_num][:np.arange(
                            len(r_rx.upsample_rbins))[r_rx.upsample_rbins < rngs.max()][-1] + 5]))
                    rng_ax.axis('tight')
                    # plt.figure('Backproject Ant {}'.format(ant_num))
                    db_bpj = db(bpjgrid[ant_num]).T
                    bpj_ax.imshow(db_bpj, origin='lower', cmap='gray',
                                  clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])
                    bpj_ax.axis('off')

                    # plt.figure('Doppler Shifted Data Ant {}'.format(ant_num))
                    dopp_ax.imshow(dpshift_data)
                    dopp_ax.axis('tight')
            else:
                rng_ax[ant_x_plot, ant_y_plot].imshow(
                    np.real(range_prof[ant_num][:np.arange(
                        len(r_rx.upsample_rbins))[r_rx.upsample_rbins < rngs.max()][-1] + 5]))
                rng_ax[ant_x_plot, ant_y_plot].axis('tight')
                # plt.figure('Backproject Ant {}'.format(ant_num))
                db_bpj = db(bpjgrid[ant_num]).T
                bpj_ax[ant_x_plot, ant_y_plot].imshow(db_bpj, origin='lower', cmap='gray',
                                                      clim=[db_bpj.mean() - db_bpj.std() * 2,
                                                            db_bpj.mean() + db_bpj.std() * 2])
                bpj_ax[ant_x_plot, ant_y_plot].axis('off')

                # plt.figure('Doppler Shifted Data Ant {}'.format(ant_num))
                dopp_ax[ant_x_plot, ant_y_plot].imshow(dpshift_data)
                dopp_ax[ant_x_plot, ant_y_plot].axis('tight')

            # plt.figure('Slices Ant {}'.format(ant_num))
            try:
                slice_ax[0, ant_num].set_title('Shifted Spectrum')
                slice_ax[0, ant_num].plot(db(fft(pulses[ant_num][:, pca_idx], n=r_rx.fft_len)))
                slice_ax[1, ant_num].set_title('Time Series')
                slice_ax[1, ant_num].plot(np.real(pulses[ant_num][:, pca_idx]))
                slice_ax[2, ant_num].set_title('Range Compression')
                slice_ax[2, ant_num].plot(r_rx.upsample_rbins, db(rc_data[ant_num][:, pca_idx]))

                # plt.figure('Pulses Ant {}'.format(ant_num))
                pulse_ax[0, ant_num].imshow(db(pulses[ant_num]), extent=[check_tt[0], check_tt[-1],
                                                                         ra[ant_num].range_bins[0],
                                                                         ra[ant_num].range_bins[-1]],
                                            origin='lower')
                pulse_ax[0, ant_num].axis('tight')
                pulse_ax[1, ant_num].imshow(db(rc_data[ant_num]), extent=[check_tt[0], check_tt[-1],
                                                                          ra[ant_num].range_bins[0],
                                                                          ra[ant_num].range_bins[-1]],
                                            origin='lower')
                pulse_ax[1, ant_num].axis('tight')
            except IndexError:
                slice_ax[0].set_title('Shifted Spectrum')
                slice_ax[0].plot(db(fft(pulses[ant_num][:, pca_idx], n=r_rx.fft_len)))
                slice_ax[1].set_title('Time Series')
                slice_ax[1].plot(np.real(pulses[ant_num][:, pca_idx]))
                slice_ax[2].set_title('Range Compression')
                slice_ax[2].plot(r_rx.upsample_rbins, db(rc_data[ant_num][:, pca_idx]))

                # plt.figure('Pulses Ant {}'.format(ant_num))
                pulse_ax[0].imshow(db(pulses[ant_num]), extent=[check_tt[0], check_tt[-1], ra[ant_num].range_bins[0],
                                                                ra[ant_num].range_bins[-1]],
                                   origin='lower')
                pulse_ax[0].axis('tight')
                pulse_ax[1].imshow(db(rc_data[ant_num]), extent=[check_tt[0], check_tt[-1], ra[ant_num].range_bins[0],
                                                                 ra[ant_num].range_bins[-1]],
                                   origin='lower')
                pulse_ax[1].axis('tight')

            if n_samples == 1:
                b0 = np.arange(ra[ant_num].upsample_nsam)[rngs.min() <= ra[ant_num].upsample_rbins][-1]
                b1 = np.arange(ra[ant_num].upsample_nsam)[ra[ant_num].upsample_rbins <= rngs.min()][0]
                boca = b0 if abs(
                    ra[ant_num].upsample_rbins[b0] - rngs.min()) < \
                    abs(ra[ant_num].upsample_rbins[b1] - rngs.min()) else b1
                plt.figure('Cuts')
                plt.subplot(3, 1, 1)
                plt.title('Range')
                plt.plot(dpshift_data[:, dpshift_data.shape[1] // 2])
                plt.subplot(3, 1, 2)
                plt.title('Azimuth')
                plt.plot(dpshift_data[b1, :])
                plt.subplot(3, 1, 3)
                plt.title('Doppler')
                plt.plot(np.real(rc_data[b1, :]))

    # Apply range roll-off corrections
    if do_backproject:
        ov_bpj = np.sum(bpjgrid, axis=0)
        if abs(ov_bpj).max() > 0:
            med_curve = np.median(abs(ov_bpj), axis=0)
            med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
            rng_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
            med_curve = np.median(abs(ov_bpj), axis=1)
            med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
            az_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
            for az in range(len(az_corr)):
                for rng in range(len(rng_corr)):
                    ov_bpj[az, rng] = ov_bpj[az, rng] * rng_corr[rng] * az_corr[az]

            db_bpj = db(ov_bpj).T
            plt.figure('Overall Backprojection')
            plt.imshow(db_bpj, origin='lower', cmap='gray',
                       clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])
            plt.axis('off')

    plt.figure('Original Data')
    plt.imshow(db(env.data), origin='lower', cmap='gray', clim=[40, 100])

    '''
    plt.figure('Interpolated DTED')
    plt.imshow(env.hdata - env.scp[2], origin='lower')
    '''

    if n_samples != 1:
        fig, ax = plt.subplots(num='Samples')
        ax.scatter(gx, gy, s=.1, c=db(gv))
        rect = patches.Rectangle((bx.min(), by.min()), bpj_size[0], bpj_size[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    amb_fig = plt.figure('Ambiguity')
    rads = [pair for pair in combinations(ra.channels, 2)]
    rad_labs = [pair for pair in combinations(np.arange(len(ra)), 2)]
    nplots = ra.n_channels + len(rads)
    nrad = 1
    for n, rx in enumerate(ra):
        amb = ambiguity(rx.chirp, rx.chirp, rx.prf, 64, mag=False)
        ax = amb_fig.add_subplot(2, ra.n_channels, nrad)
        ax.imshow(db(amb[0]), cmap='jet', extent=[amb[1][0], amb[1][-1], amb[2][-1], amb[2][0]])
        plt.axis('tight')
        nrad += 1

    for n, pair in enumerate(rads):
        amb = ambiguity(pair[0].chirp, pair[1].chirp, pair[0].prf, 64, mag=False)
        ax = amb_fig.add_subplot(2, ra.n_channels, nrad)
        ax.set_title('{}, {}'.format(*rad_labs[n]))
        ax.imshow(db(amb[0]), cmap='jet', extent=[amb[1][0], amb[1][-1], amb[2][-1], amb[2][0]])
        plt.axis('tight')
        nrad += 1

    plt.show()

    if introduce_rfi:
        pnum = np.arange(len(check_pnum))[p_hasRFI[check_pnum].astype(bool)][0] if np.any(p_hasRFI[check_pnum]) else 0
        stft_data = np.fft.fftshift(stft(pulses[0][:, pnum], return_onesided=False)[2], axes=0)

        plt.figure('Pulse Examination')
        plt.subplot(2, 1, 1)
        plt.imshow(db(stft_data))
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.magnitude_spectrum(pulses[0][:, pnum], pad_to=findPowerOf2(sdr_f[0].nsam), window=lambda x: x, Fs=2e9)
