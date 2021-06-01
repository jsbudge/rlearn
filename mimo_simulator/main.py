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
from numpy.fft import fft, ifft, fftshift
from scipy.io import loadmat
from scipy.stats import trapezoid
from tqdm import tqdm

from cuda_kernels import genRangeProfile, backproject, genDoppProfile
from environment import Environment
import matplotlib.pyplot as plt
from pathlib import Path

from radar import Radar
from rawparser import loadReferenceChirp, loadMatchedFilter, loadASHFile, getRawDataGen, getRawSDRParams
from simlib import getElevation, enu2llh
from useful_lib import findAllFilenames, factors, db, findPowerOf2, gaus


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def pdf_dist(pmax, size):
    dist = lambda x: gaus(x, 1, 0, size) if np.any(abs(x)) < size else 1
    bounds = (-size * 3, size * 3)
    while True:
        x = np.random.rand(size) * (bounds[1] - bounds[0]) + bounds[0]
        y = np.random.rand(size) * pmax
        if np.all(y <= dist(x)):
            return x


def genPoints(nn, e, method='uniform'):
    np.random.seed(666)
    if nn == 1:
        _gx = 0
        _gy = 0
        _gz, _gv = e(_gx, _gy)
        _gv = 1e9
    else:
        if method == 'uniform':
            _gx = np.random.rand(nn) * e.shape[0] - e.shape[0] / 2
            _gy = np.random.rand(nn) * e.shape[1] - e.shape[1] / 2
            _gz, _gv = e(_gx, _gy)
            while np.any(_gv == 0):
                bad_len = sum(_gv == 0)
                _gx[_gv == 0] = np.random.rand(bad_len) * e.shape[0] - e.shape[0] / 2
                _gy[_gv == 0] = np.random.rand(bad_len) * e.shape[1] - e.shape[1] / 2
                _gz[_gv == 0], _gv[_gv == 0] = e(_gx[_gv == 0], _gy[_gv == 0])
        elif method == 'gauss':
            _gx = np.random.normal(0, e.shape[0] / 2, nn)
            while np.any(abs(_gx) > e.data_shape[0]):
                _gx[abs(_gx) > e.data_shape[0]] = np.random.normal(0, e.shape[0] / 2, sum(abs(_gx) > e.data_shape[0]))
            _gy = np.random.normal(0, e.shape[1] / 2, nn)
            while np.any(abs(_gy) > e.data_shape[1]):
                _gy[abs(_gy) > e.data_shape[1]] = np.random.normal(0, e.shape[1] / 2, sum(abs(_gy) > e.data_shape[1]))
        elif method == 'trap':
            _gx = trapezoid.rvs(0.2, .8, size=nn) * (e.shape[0] + e.shape[0] * .4) - (e.shape[0] + e.shape[0] * .4) / 2
            _gy = trapezoid.rvs(0.2, .8, size=nn) * (e.shape[1] + e.shape[1] * .4) - (e.shape[1] + e.shape[1] * .4) / 2
        _gz, _gv = e(_gx, _gy)
    return _gx, _gy, _gz, _gv


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
plt.close('all')

fnme = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112239.sar'
output_fnme = './test.dat'

threads_per_block = (16, 16)
upsample = 1
chunk_sz = 256 // upsample
n_samples = 1000000
subgrid_size = (600, 600)
bpj_size = (300, 300)
rand_method = 'trap'
do_backproject = True
write_to_file = False

files = findAllFilenames(fnme)

# Initialize all the variables
match_filt = None
gpx_gpu = gpy_gpu = gpz_gpu = gpv_gpu = None
n_splits = None
check_tt = None
rc_data = None
pulses = None
gx = gy = gz = gv = None
range_prof = None

print('Loading environment...')
env = Environment(files['ash'], files['asi'], subgrid_size=subgrid_size, dec_fac=4)

print('Loading radar...')
radar = Radar(fnme, env.scp, use_xml_flightpath=True, presum=14)
radar.resampleRangeBins(upsample)
ashfile = loadASHFile(files['ash'])

mempool = cupy.get_default_memory_pool()

# Get minimum slant range (which should be the center of our aperture)
pca_rngs = np.linalg.norm(radar.pos(radar.times), axis=0)
t_pca = 0
try:
    # noinspection PyArgumentList
    t_pca = radar.times[pca_rngs == pca_rngs.min()][0]
except IndexError:
    t_pca = t_pca[0]
R0 = pca_rngs.min()

# Matched filter chirp
ref_set = False
try:
    ap_output = loadmat('/home/jeff/repo/Debug/05052021/refchirp')
    ref_chirp = ap_output['Channel_1_X_Band_9_GHz_Cal_Data_I'] + 1j * ap_output['Channel_1_X_Band_9_GHz_Cal_Data_Q']
    # ref_chirp = ref_chirp / (10 ** (32 / 20))
    r_chirp = np.zeros((radar.fft_len,), dtype=np.complex128)
    ref_chirp = np.mean(ref_chirp, axis=1)
    ref_chirp = radar.chirp(px=np.linspace(0, 1, 10), py=np.linspace(0, 1, 10))
    r_chirp[:len(ref_chirp)] = ref_chirp
    r_chirp = ref_chirp
    if upsample == 1:
        match_filt = loadMatchedFilter(files['MatchedFilter'])
        match_filt = fft(r_chirp, n=radar.fft_len * upsample).conj().T
    else:
        match_filt = fft(r_chirp, n=radar.fft_len * upsample).conj().T
except IndexError:
    r_chirp = None
    print('Reference chirp not loaded properly.')
    ref_set = True

# Get chirp to the right size
fft_chirp = fft(r_chirp, radar.fft_len * upsample)
ref_gpu = cupy.array(np.tile(fft_chirp, (chunk_sz, 1)).T, dtype=np.complex128)
mf_gpu = cupy.array(np.tile(match_filt, (chunk_sz, 1)).T, dtype=np.complex128)

# Get grid for debug testing
bx, by = np.meshgrid(np.linspace(-bpj_size[0]//2, bpj_size[0]//2, bpj_size[0] * 4),
                     np.linspace(-bpj_size[1]//2, bpj_size[1]//2, bpj_size[1] * 4))
gs = bx.shape
R = np.array([[np.cos(env.cta), -np.sin(env.cta)],
             [np.sin(env.cta), np.cos(env.cta)]])
b_p = R.dot(np.array([bx.flatten(), by.flatten()]))
bx = b_p[1, :].reshape(gs)
by = b_p[0, :].reshape(gs)
bz, _ = env(bx, by)
bx_gpu = cupy.array(bx, dtype=np.float64)
by_gpu = cupy.array(by, dtype=np.float64)
bz_gpu = cupy.array(bz, dtype=np.float64)
rb_gpu = cupy.array(radar.range_bins, dtype=np.float64)
blocks_per_grid_bpj = (
    int(np.ceil(bx.shape[0] / threads_per_block[0])), int(np.ceil(bx.shape[1] / threads_per_block[1])))

rpf_shape = (radar.upsample_nsam, chunk_sz)
blocks_per_grid_rpf = (
    int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(n_samples / threads_per_block[1])))

# Load constants and other parameters
param_gpu = cupy.array(np.array([np.pi / radar.el_bw, np.pi / radar.az_bw,
                                 radar.wavelength, radar.velocity, radar.el_bw, radar.az_bw,
                                 radar.near_slant_range / c0, 2e9 * upsample, 0, radar.prf]), dtype=np.float64)

n_frames, sdr_samples, atts, sys_times = getRawSDRParams(files['RawData'])

# Write header stuff to .dat file
if write_to_file:
    print('Writing to file ' + output_fnme)
    if not Path(output_fnme).exists():
        Path(output_fnme).touch()
    else:
        print('File already exists. Overwriting...')
    with open(output_fnme, 'wb') as f:
        f.write(np.uint32(n_frames).tobytes())
        f.write(np.uint32(sdr_samples).tobytes())
        f.write(np.int8(atts).tobytes())
        f.write(np.double(sys_times).tobytes())

print('Running range profile generation...')
ch_sz = chunk_sz
on_init = True
bpjgrid = np.zeros(bx.shape, dtype=np.complex128)
# Split samples into smaller chunks to fit onto GPU
if n_samples > 1e6:
    for fac in np.sort(factors(n_samples)):
        if n_samples / fac <= 1e6:
            n_splits = int(fac)
            break
    samp_split = int(n_samples / n_splits)
else:
    n_splits = 1
    samp_split = n_samples
    gx, gy, gz, gv = genPoints(n_samples, env, method=rand_method)
    gpx_gpu = cupy.array(gx, dtype=np.float64)
    gpy_gpu = cupy.array(gy, dtype=np.float64)
    gpz_gpu = cupy.array(gz, dtype=np.float64)
    gpv_gpu = cupy.array(gv, dtype=np.float64)

n_pulses = len(radar.times) if radar.is_presummed else n_frames
for ch in tqdm(np.arange(0, n_pulses, chunk_sz)):
    tt = radar.systimes[ch:min(ch + chunk_sz, n_pulses)] / TAC

    # This is usually only on the last chunk in the file, change the size of the
    # reference pulse block
    if len(tt) != chunk_sz:
        ch_sz = len(tt)
        ref_gpu = cupy.array(np.tile(fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
        rpf_shape = (radar.upsample_nsam, ch_sz)
        mf_gpu = cupy.array(np.tile(match_filt, (ch_sz, 1)).T, dtype=np.complex128)

    # Toss in all the interpolated stuff
    fp_gpu = cupy.array(np.ascontiguousarray(radar.pos(tt), dtype=np.float64))
    rad_pan_gpu = cupy.array(np.ascontiguousarray(radar.pan(tt), dtype=np.float64))
    rad_tilt_gpu = cupy.array(np.ascontiguousarray(radar.tilt(tt), dtype=np.float64))

    rpf_r_gpu = cupy.random.rand(*rpf_shape, dtype=np.float64) * 1
    # rpf_r_gpu = cupy.zeros(rpf_shape, dtype=np.float64)
    rpf_i_gpu = cupy.random.rand(*rpf_shape, dtype=np.float64) * 1
    # rpf_i_gpu = cupy.zeros(rpf_shape, dtype=np.float64)
    times_gpu = cupy.array(np.ascontiguousarray(tt - t_pca), dtype=np.float64)

    for rnd in range(n_splits):
        if n_splits > 1:
            # Load grid coordinates onto GPU
            gx, gy, gz, gv = genPoints(samp_split, env, method=rand_method)
            gpx_gpu = cupy.array(gx, dtype=np.float64)
            gpy_gpu = cupy.array(gy, dtype=np.float64)
            gpz_gpu = cupy.array(gz, dtype=np.float64)
            gpv_gpu = cupy.array(gv, dtype=np.float64)
        blocks_per_grid_rpf = (
            int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(samp_split / threads_per_block[1])))

        # Run range profile generation
        genRangeProfile[blocks_per_grid_rpf, threads_per_block](fp_gpu, gpx_gpu, gpy_gpu, gpz_gpu,
                                                                gpv_gpu, times_gpu, rad_pan_gpu,
                                                                rad_tilt_gpu, rpf_r_gpu, rpf_i_gpu, param_gpu)
        cupy.cuda.Device().synchronize()

    # Calculate the pulse data on the GPU using FFT
    rpf_gpu = rpf_r_gpu + 1j * rpf_i_gpu
    # dopp_gpu =

    pd_gpu = cupy.fft.fft(rpf_gpu, n=radar.fft_len * upsample, axis=0) * ref_gpu
    pd_gpu = cupy.fft.ifft(pd_gpu, axis=0)[:radar.upsample_nsam, :]

    rcd_gpu = cupy.fft.ifft(cupy.fft.fft(pd_gpu, n=radar.fft_len * upsample, axis=0) * mf_gpu,
                            axis=0)[:radar.upsample_nsam, :]
    cupy.cuda.Device().synchronize()

    # Run the actual backprojection
    if do_backproject:
        bpjgrid_gpu = cupy.zeros(bx.shape, dtype=np.complex128)
        backproject[blocks_per_grid_bpj, threads_per_block](fp_gpu, bx_gpu, by_gpu, bz_gpu,
                                                            rb_gpu, rad_pan_gpu,
                                                            rad_tilt_gpu, rpf_gpu, bpjgrid_gpu,
                                                            param_gpu)
        cupy.cuda.Device().synchronize()
        bpjgrid = bpjgrid + bpjgrid_gpu.get()
        del bpjgrid_gpu

    if write_to_file:
        # Write to .dat files for backprojection
        pdata = pd_gpu.get()
        write_pulse = np.zeros((sdr_samples * 2,), dtype=np.int16)
        with open(output_fnme, 'ab') as fid:
            for n in range(ch_sz):
                write_pulse[0::2] = (np.real(pdata[:, n]) / (10 ** (atts[ch + n] / 20))).astype(np.int16)
                write_pulse[1::2] = (np.imag(pdata[:, n]) / (10 ** (atts[ch + n] / 20))).astype(np.int16)
                write_pulse.tofile(fid, '')

    # Find the point of closest approach and get all stuff associated with it
    if on_init:
        rc_data = rcd_gpu.get()
        pulses = pd_gpu.get()
        range_prof = rpf_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        on_init = False
        disp_atts = radar.att[ch:min(ch + chunk_sz, n_pulses)]
    if tt[-1] >= t_pca >= tt[0]:
        rc_data = rcd_gpu.get()
        pulses = pd_gpu.get()
        range_prof = rpf_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        disp_atts = radar.att[ch:min(ch + chunk_sz, n_pulses)]

    # Delete the range compressed pulse block to free up memory on the GPU
    del pd_gpu
    del rpf_gpu
    del rpf_r_gpu
    del rpf_i_gpu
    del times_gpu
    del rcd_gpu
    del rad_pan_gpu
    del rad_tilt_gpu
    del fp_gpu
    mempool.free_all_blocks()

del ref_gpu
del param_gpu
del gpx_gpu
del gpy_gpu
del gpz_gpu
del gpv_gpu
del mf_gpu
mempool.free_all_blocks()

# Apply range roll-off corrections
if do_backproject and abs(bpjgrid).max() > 0:
    med_curve = np.median(abs(bpjgrid), axis=0)
    med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
    rng_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
    med_curve = np.median(abs(bpjgrid), axis=1)
    med_curve = med_curve / med_curve.max() if med_curve.max() > 0 else med_curve
    az_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
    for az in range(len(az_corr)):
        for rng in range(len(rng_corr)):
            bpjgrid[az, rng] = bpjgrid[az, rng] * rng_corr[rng] * az_corr[az]
pca_idx = np.where(check_tt == t_pca)[0][0]
dpshift_data = db(fftshift(fft(rc_data, axis=1), axes=1))

# After run, diagnostics, etc.
pos_pca = enu2llh(*radar.pos(t_pca), env.scp)
pca_slant_range = np.linalg.norm(radar.pos(t_pca))
print('PCA-SCP slant range is {:.2f}'.format(pca_slant_range))
print('PCA-SCP ground range is {:.2f}'.format(np.sqrt(pca_slant_range ** 2 - (pos_pca[2] - env.scp[2]) ** 2)))
print('PCA radar pan is {:.2f}'.format(radar.pan(t_pca) * 180 / np.pi))
print('PCA radar tilt is {:.2f}'.format(radar.tilt(t_pca) * 180 / np.pi))
print('Plane pos at PCA is {:.6f}, {:.6f}, {:.2f}'.format(*pos_pca))
sh_los = -radar.pos(radar.times)
rngs = np.linalg.norm(sh_los, axis=0)
pt_el = np.array([math.asin(-sh_los[2, i] / rngs[i]) for i in range(len(rngs))])
pt_az = np.array([math.atan2(sh_los[0, i], sh_los[1, i]) for i in range(len(rngs))])

plt.figure('Pulses')
plt.subplot(2, 1, 1)
plt.imshow(db(pulses), extent=[check_tt[0], check_tt[-1], radar.range_bins[0], radar.range_bins[-1]],
           origin='lower')
plt.axis('tight')
plt.subplot(2, 1, 2)
plt.imshow(db(rc_data), extent=[check_tt[0], check_tt[-1], radar.range_bins[0], radar.range_bins[-1]],
           origin='lower')
plt.axis('tight')

plt.figure('Backproject')
plt.imshow(db(bpjgrid).T, origin='lower', cmap='gray')

plt.figure('Doppler Shifted Data')
plt.imshow(dpshift_data)
plt.axis('tight')

plt.figure('Doppler UnShifted Data')
plt.imshow(fftshift(dpshift_data, axes=1))
plt.axis('tight')

'''
plt.figure('Original Data')
plt.imshow(db(env.data), origin='lower', cmap='gray')

plt.figure('Interpolated DTED')
plt.imshow(env.hdata - env.scp[2], origin='lower')
'''

if n_samples == 1:
    b0 = np.arange(radar.upsample_nsam)[rngs.min() <= radar.range_bins][-1]
    b1 = np.arange(radar.upsample_nsam)[radar.range_bins <= rngs.min()][0]
    boca = b0 if abs(radar.range_bins[b0] - rngs.min()) < abs(radar.range_bins[b1] - rngs.min()) else b1
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
else:
    boca = rc_data.shape[0] // 2
    plt.figure('Sampled Data')
    plt.scatter(gx, gy, s=.1, c=db(gv))

az_diffs = np.array([adiff(pt_az[i], radar.pan(radar.times[i])) for i in range(len(rngs))])
plt.figure('Params')
plt.subplot(3, 1, 1)
plt.title('Doppler Chirp')
plt.plot(radar.times, np.cos(-4 * np.pi / radar.wavelength * rngs))
plt.plot(radar.times[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])],
         np.real(rc_data[boca, :]) / np.real(rc_data[boca, :]).max())
plt.subplot(3, 1, 2)
plt.title('Ranges')
plt.plot(radar.times, rngs)
plt.plot(radar.times[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])],
         rngs[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])])
plt.subplot(3, 1, 3)
plt.title('Az Diffs')
plt.plot(radar.times, az_diffs)
plt.plot(radar.times[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])],
         az_diffs[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])])

plt.figure('Chirp Data')
plt.subplot(4, 1, 1)
plt.title('Ref. Chirp')
plt.plot(np.real(r_chirp))
plt.subplot(4, 1, 2)
plt.title('Ref. Spectrum')
plt.plot(db(fft(r_chirp)))
plt.subplot(4, 1, 3)
plt.title('Matched Filter')
plt.plot(db(match_filt))
plt.subplot(4, 1, 4)
plt.title('Range Compression')
plt.plot(radar.range_bins, db(ifft(fft_chirp * match_filt))[:radar.upsample_nsam])

plt.figure('Range Profile')
plt.imshow(np.real(range_prof[:np.arange(len(radar.range_bins))[radar.range_bins < rngs.max()][-1] + 5]))
plt.axis('tight')

plt_test = fft(range_prof, n=radar.fft_len * upsample, axis=0)[:, pca_idx] * fft_chirp
iplt_test = ifft(plt_test)[:radar.upsample_nsam]
plt.figure('Slices')
plt.subplot(3, 1, 1)
plt.title('Shifted Spectrum')
plt.plot(db(plt_test))
plt.plot(db(fft(pulses[:, pca_idx], n=radar.fft_len * upsample)))
plt.subplot(3, 1, 2)
plt.title('Time Series')
plt.plot(np.real(iplt_test))
plt.plot(np.real(pulses[:, pca_idx]))
plt.subplot(3, 1, 3)
plt.title('Range Compression')
plt.plot(radar.range_bins, db(ifft(fft(iplt_test, n=radar.fft_len * upsample) * match_filt))[:radar.upsample_nsam])
plt.plot(radar.range_bins, db(rc_data[:, pca_idx]))
plt.legend(['CPU Comp.', 'GPU Comp.'])
