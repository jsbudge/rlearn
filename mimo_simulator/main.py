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
from tqdm import tqdm

from cuda_kernels import genRangeProfile, backproject, genDoppProfile
from environment import Environment
import matplotlib.pyplot as plt
from pathlib import Path

from radar import Radar
from rawparser import loadReferenceChirp, loadMatchedFilter, loadASHFile, getRawDataGen, getRawSDRParams
from simlib import getElevation, enu2llh
from useful_lib import findAllFilenames, factors, db, findPowerOf2


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def genPoints(nn, e):
    np.random.seed(666)
    if nn == 1:
        _gx = 0
        _gy = 0
        _gz, _gv = e(_gx, _gy)
        _gv = 1e9
    else:
        _gx = np.random.rand(nn) * e.shape[0] - e.shape[0] / 2
        _gy = np.random.rand(nn) * e.shape[1] - e.shape[1] / 2
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
chunk_sz = 256
upsample = 1
n_samples = 1
do_backproject = False
write_to_file = True

files = findAllFilenames(fnme)

print('Loading environment...')
env = Environment(files['ash'], files['asi'], dec_fac=4)

print('Loading radar...')
radar = Radar(fnme, env.scp)
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
    r_chirp = np.zeros((radar.fft_len,), dtype=np.complex128)
    chirp = radar.chirp()
    r_chirp[160:160 + len(chirp)] = chirp
    if upsample == 1:
        match_filt = loadMatchedFilter(files['MatchedFilter'])
    else:
        match_filt = fft(r_chirp, n=radar.fft_len * upsample).conj().T
except IndexError:
    r_chirp = None
    print('Reference chirp not loaded properly.')
    ref_set = True

# Get chirp to the right size
fft_chirp = fft(r_chirp, radar.fft_len * upsample)
ref_gpu = cupy.array(np.tile(fft_chirp, (chunk_sz, 1)).T, dtype=np.complex128)

# Get grid for debug testing
bx, by = np.meshgrid(np.linspace(-100, 100, 200), np.linspace(-100, 100, 200))
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
                                 radar.wavelength, radar.params['Velocity_Knots'] * .514444, radar.el_bw, radar.az_bw,
                                 radar.near_slant_range / c0, 2e9 * upsample, radar.prf]), dtype=np.float64)

# Only get pulses that contribute to the image
numSupportedPulses = radar.supported_pulses

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
    gx, gy, gz, gv = genPoints(n_samples, env)
    gpx_gpu = cupy.array(gx, dtype=np.float64)
    gpy_gpu = cupy.array(gy, dtype=np.float64)
    gpz_gpu = cupy.array(gz, dtype=np.float64)
    gpv_gpu = cupy.array(gv, dtype=np.float64)

for ch in tqdm(np.arange(0, n_frames, chunk_sz)):
    tt = sys_times[ch:min(ch + chunk_sz, n_frames)] / TAC

    # This is usually only on the last chunk in the file, change the size of the
    # reference pulse block
    if len(tt) != chunk_sz:
        ch_sz = len(tt)
        ref_gpu = cupy.array(np.tile(fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
        rpf_shape = (radar.upsample_nsam, ch_sz)

    # Toss in all the interpolated stuff
    fp_gpu = cupy.array(np.ascontiguousarray(radar.pos(tt), dtype=np.float64))
    rad_pan_gpu = cupy.array(np.ascontiguousarray(radar.pan(tt), dtype=np.float64))
    rad_tilt_gpu = cupy.array(np.ascontiguousarray(radar.tilt(tt), dtype=np.float64))

    rpf_r_gpu = cupy.random.rand(*rpf_shape, dtype=np.float64) * 0
    rpf_i_gpu = cupy.zeros(rpf_shape, dtype=np.float64)
    dopp_chirp_r_gpu = cupy.zeros((ch_sz,), dtype=np.float64)
    dopp_chirp_i_gpu = cupy.zeros((ch_sz,), dtype=np.float64)

    for rnd in range(n_splits):
        if n_splits > 1:
            # Load grid coordinates onto GPU
            gx, gy, gz, gv = genPoints(samp_split, env)
            gpx_gpu = cupy.array(gx, dtype=np.float64)
            gpy_gpu = cupy.array(gy, dtype=np.float64)
            gpz_gpu = cupy.array(gz, dtype=np.float64)
            gpv_gpu = cupy.array(gv, dtype=np.float64)
        blocks_per_grid_rpf = (
            int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(samp_split / threads_per_block[1])))

        # Run range profile generation
        genRangeProfile[blocks_per_grid_rpf, threads_per_block](fp_gpu, gpx_gpu, gpy_gpu, gpz_gpu,
                                                                gpv_gpu, rad_pan_gpu,
                                                                rad_tilt_gpu, rpf_r_gpu, rpf_i_gpu, param_gpu)
        genDoppProfile[blocks_per_grid_rpf, threads_per_block](fp_gpu, gpx_gpu, gpy_gpu, gpz_gpu, rad_pan_gpu,
                                                                dopp_chirp_r_gpu, dopp_chirp_i_gpu, param_gpu)
        cupy.cuda.Device().synchronize()

    # Calculate the pulse data on the GPU using FFT
    rpf_gpu = rpf_r_gpu + 1j * rpf_i_gpu
    dopp_gpu = dopp_chirp_r_gpu + 1j * dopp_chirp_i_gpu

    # Run the actual backprojection
    if do_backproject:
        bpjgrid_gpu = cupy.zeros(bx.shape, dtype=np.complex128)
        backproject[blocks_per_grid_bpj, threads_per_block](fp_gpu, bx_gpu, by_gpu, bz_gpu,
                                                            rb_gpu, rad_pan_gpu,
                                                            rad_tilt_gpu, rpf_gpu, bpjgrid_gpu,
                                                            param_gpu)
        bpjgrid = bpjgrid + bpjgrid_gpu.get()
        del bpjgrid_gpu
    pd_gpu = cupy.fft.fft(rpf_gpu, n=radar.fft_len * upsample, axis=0) * ref_gpu
    pd_gpu = cupy.fft.ifft(pd_gpu, axis=0)
    pd_gpu = cupy.fft.fft(pd_gpu, axis=1) * dopp_gpu
    pd_gpu = cupy.fft.ifft(pd_gpu, axis=1)[:radar.upsample_nsam:upsample]
    cupy.cuda.Device().synchronize()

    # Run interpolation to correct sample size
    # interpolate[blocks_per_grid_int, threads_per_block](dec_params_gpu, pd_gpu, dec_pd_gpu)

    if write_to_file:
        # Write to .dat files for backprojection
        pdata = pd_gpu.get()
        write_pulse = np.zeros((sdr_samples * 2,), dtype=np.int16)
        with open(output_fnme, 'ab') as fid:
            for n in range(ch_sz):
                write_pulse[0::2] = (np.real(pdata[:, n]) / 10 ** (atts[ch + n] / 20)).astype(np.int16)
                write_pulse[1::2] = (np.imag(pdata[:, n]) / 10 ** (atts[ch + n] / 20)).astype(np.int16)
                write_pulse.tofile(fid, '')

    # Find the point of closest approach and get all stuff associated with it
    if on_init:
        pulses = pd_gpu.get()
        range_prof = rpf_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        on_init = False
        disp_atts = atts[ch:min(ch + chunk_sz, n_frames)]
    if tt[-1] > t_pca > tt[0]:
        pulses = pd_gpu.get()
        range_prof = rpf_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        disp_atts = atts[ch:min(ch + chunk_sz, n_frames)]

    # Delete the range compressed pulse block to free up memory on the GPU
    del pd_gpu
    del rpf_gpu
    del rpf_r_gpu
    del rpf_i_gpu
    del dopp_chirp_i_gpu
    del dopp_chirp_r_gpu
    del dopp_gpu
    mempool.free_all_blocks()

del rad_pan_gpu
del rad_tilt_gpu
del fp_gpu
del ref_gpu
del param_gpu
del gpx_gpu
del gpy_gpu
del gpz_gpu
del gpv_gpu
mempool.free_all_blocks()

# Apply range roll-off corrections
if do_backproject:
    med_curve = np.median(abs(bpjgrid), axis=0)
    med_curve = med_curve / med_curve.max()
    rng_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
    med_curve = np.median(abs(bpjgrid), axis=1)
    med_curve = med_curve / med_curve.max()
    az_corr = 1 / np.poly1d(np.polyfit(np.arange(len(med_curve)), med_curve, 3))(np.arange(len(med_curve)))
    for az in range(len(az_corr)):
        for rng in range(len(rng_corr)):
            bpjgrid[az, rng] = bpjgrid[az, rng] * rng_corr[rng] * az_corr[az]
rc_data = ifft(fft(pulses, radar.fft_len * upsample, axis=0) * match_filt[:, None], n=radar.nsam, axis=0)
pca_idx = np.where(check_tt == t_pca)[0][0]
dpshift_data = db(fftshift(fft(rc_data, axis=1)))

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
plt.imshow(db(pulses))
plt.axis('tight')
plt.subplot(2, 1, 2)
plt.imshow(db(rc_data))
plt.axis('tight')

plt.figure('Backproject')
plt.imshow(db(bpjgrid), origin='lower')

plt.figure('Doppler Shifted Data')
plt.imshow(dpshift_data)
plt.axis('tight')

plt.figure('Original Data')
plt.imshow(db(env.data), origin='lower', cmap='gray')

plt.figure('Interpolated DTED')
plt.imshow(env.hdata - env.scp[2], origin='lower')

if n_samples == 1:
    plt.figure('Cuts')
    plt.subplot(3, 1, 1)
    plt.title('Range')
    plt.plot(dpshift_data[:, dpshift_data.shape[1] // 2])
    plt.subplot(3, 1, 2)
    plt.title('Azimuth')
    plt.plot(dpshift_data[dpshift_data.shape[0] // 2, :])
    plt.subplot(3, 1, 3)
    plt.title('Doppler')
    plt.plot(np.real(rc_data[rc_data.shape[0] // 2, :]))
else:
    plt.figure('Sampled Data')
    plt.scatter(gx, gy, s=.1, c=db(gv))

az_diffs = np.array([adiff(pt_az[i], radar.pan(radar.times[i])) for i in range(len(rngs))])
plt.figure('Params')
plt.subplot(3, 1, 1)
plt.title('Doppler Chirp')
plt.plot(radar.times, np.cos(-4 * np.pi / radar.wavelength * rngs))
plt.plot(radar.times[np.logical_and(radar.times <= check_tt[-1], radar.times >= check_tt[0])],
         np.real(rc_data[rc_data.shape[0] // 2, :]) / np.real(rc_data[rc_data.shape[0] // 2, :]).max())
plt.subplot(3, 1, 2)
plt.title('Ranges')
plt.plot(radar.times, rngs)
plt.subplot(3, 1, 3)
plt.title('Az Diffs')
plt.plot(radar.times, az_diffs)

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
plt.plot(db(ifft(fft_chirp * match_filt)))

plt.figure('Range Profile')
plt.imshow(np.real(range_prof[:np.arange(len(radar.range_bins))[radar.range_bins < rngs.max()][-1] + 5]))
plt.axis('tight')

plt_test = fft(range_prof, n=radar.fft_len * upsample, axis=0)[:, pca_idx] * fft_chirp
iplt_test = ifft(plt_test)[:radar.nsam]
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
plt.plot(radar.range_bins, db(ifft(fft(iplt_test, n=radar.fft_len * upsample) * match_filt, n=radar.upsample_nsam)))
plt.plot(radar.range_bins[::upsample], db(rc_data[:, pca_idx]))
plt.legend(['CPU Comp.', 'GPU Comp.'])
