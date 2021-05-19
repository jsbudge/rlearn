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

from cuda_kernels import genRangeProfile
from environment import Environment
import matplotlib.pyplot as plt
from pathlib import Path

from radar import Radar
from rawparser import loadReferenceChirp, loadMatchedFilter, loadASHFile, getRawDataGen, getRawSDRParams
from simlib import getElevation, enu2llh
from useful_lib import findAllFilenames, factors, db


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


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
grid_size = (400, 400)

files = findAllFilenames(fnme)

print('Loading environment...')
env = Environment(files['ash'], files['asi'], size=grid_size)

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

print('Generating sample set...')
# Random sampling method
if n_samples == 1:
    gx = 0
    gy = 0
    gz, gv = env(gx, gy)
    gv = 133274.24348242042
else:
    gx = np.random.rand(n_samples) * env.shape[0] - env.shape[0] / 2
    gy = np.random.rand(n_samples) * env.shape[1] - env.shape[1] / 2
    gz, gv = env(gx, gy)

# Matched filter chirp
ref_set = False
try:
    r_chirp = np.zeros((radar.fft_len,), dtype=np.complex128)
    chirp = radar.chirp()
    r_chirp[-len(chirp):] = chirp
    match_filt = loadMatchedFilter(files['MatchedFilter'])
except IndexError:
    r_chirp = None
    print('Reference chirp not loaded properly.')
    ref_set = True

# Get chirp to the right size
fft_chirp = fft(r_chirp, radar.fft_len * upsample)
ref_gpu = cupy.array(np.tile(fft_chirp, (chunk_sz, 1)).T, dtype=np.complex128)

pdata_shape = (radar.upsample_nsam, chunk_sz)

gridx_per_grid = int(np.ceil(pdata_shape[1] / threads_per_block[0]))
gridy_per_grid = int(np.ceil(n_samples / threads_per_block[1]))
blocks_per_grid = (gridx_per_grid, gridy_per_grid)

# Load grid coordinates and range bins onto GPU
gpx_gpu = cupy.array(gx, dtype=np.float64)
gpy_gpu = cupy.array(gy, dtype=np.float64)
gpz_gpu = cupy.array(gz, dtype=np.float64)
gpv_gpu = cupy.array(gv, dtype=np.float64)
rb_gpu = cupy.array(radar.range_bins, dtype=np.float64)

# Load constants and other parameters
param_gpu = cupy.array(np.array([np.pi / radar.el_bw, np.pi / radar.az_bw,
                                 radar.wavelength, radar.params['Velocity_Knots'] * .514444, radar.el_bw, radar.az_bw,
                                 radar.near_slant_range / c0, 2e9 * upsample]), dtype=np.float64)

# Only get pulses that contribute to the image
numSupportedPulses = radar.supported_pulses

n_frames, n_samples, atts, sys_times = getRawSDRParams(files['RawData'])

# Write header stuff to .dat file
if not Path(output_fnme).exists():
    Path(output_fnme).touch()
else:
    print('File already exists. Overwriting...')
with open(output_fnme, 'wb') as f:
    f.write(np.uint32(n_frames).tobytes())
    f.write(np.uint32(n_samples).tobytes())
    f.write(np.int8(atts).tobytes())
    f.write(np.double(sys_times).tobytes())

print('Running range profile generation...')
ch_sz = chunk_sz
on_init = True
for ch in tqdm(np.arange(0, n_frames, chunk_sz)):
    tt = sys_times[ch:min(ch + chunk_sz, n_frames) + 1] / TAC

    # This is usually only on the last chunk in the file, change the size of the
    # reference pulse block
    if len(tt) != chunk_sz:
        ch_sz = len(tt)
        ref_gpu = cupy.array(np.tile(fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
        pdata_shape = (radar.upsample_nsam, ch_sz)
        gridx_per_grid = int(np.ceil(pdata_shape[1] / threads_per_block[0]))
        blocks_per_grid = (gridx_per_grid, gridy_per_grid)

    # Toss in all the interpolated stuff
    fp_gpu = cupy.array(np.ascontiguousarray(radar.pos(tt), dtype=np.float64))
    rad_pan_gpu = cupy.array(np.ascontiguousarray(radar.pan(tt), dtype=np.float64))
    rad_tilt_gpu = cupy.array(np.ascontiguousarray(radar.tilt(tt), dtype=np.float64))
    times_gpu = cupy.array(np.ascontiguousarray(tt - t_pca, dtype=np.float64))

    pdata_r_gpu = cupy.random.rand(*pdata_shape, dtype=np.float64)
    pdata_i_gpu = cupy.random.rand(*pdata_shape, dtype=np.float64)

    # Run range profile generation
    genRangeProfile[blocks_per_grid, threads_per_block](fp_gpu, gpx_gpu, gpy_gpu, gpz_gpu,
                                                        gpv_gpu, rb_gpu, rad_pan_gpu,
                                                        rad_tilt_gpu, pdata_r_gpu, pdata_i_gpu,
                                                        times_gpu, param_gpu)

    cupy.cuda.Device().synchronize()
    # Calculate the pulse data on the GPU using FFT
    pdata_gpu = pdata_r_gpu + 1j * pdata_i_gpu
    pd_gpu = cupy.fft.fft(pdata_gpu, axis=0) * ref_gpu
    ifft_gpu = cupy.fft.ifft(pd_gpu, n=radar.nsam, axis=0)
    cupy.cuda.Device().synchronize()

    # Write to .dat files for backprojection
    pdata = ifft_gpu.get()
    # pdata += np.random.rand(*pdata.shape)
    write_pulse = np.zeros((n_samples * 2,), dtype=np.int16)
    with open(output_fnme, 'ab') as fid:
        for n in range(ch_sz):
            write_pulse[0::2] = (np.real(pdata[:, n]) / 10**(atts[ch + n] / 20)).astype(np.int16)
            write_pulse[1::2] = (np.imag(pdata[:, n]) / 10**(atts[ch + n] / 20)).astype(np.int16)
            write_pulse.tofile(fid, '')

    # Find the point of closest approach and get all stuff associated with it
    if on_init:
        pulses = pdata
        range_prof = pdata_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        on_init = False
        disp_atts = atts[ch:min(ch + chunk_sz, n_frames)]
    if tt[-1] > t_pca > tt[0]:
        pulses = pdata
        range_prof = pdata_gpu.get()
        flight_path = fp_gpu.get()
        check_tt = tt
        disp_atts = atts[ch:min(ch + chunk_sz, n_frames)]

    # Delete the range compressed pulse block to free up memory on the GPU
    del pd_gpu
    del pdata_gpu
    del pdata_r_gpu
    del pdata_i_gpu
    del ifft_gpu
    mempool.free_all_blocks()

del gpx_gpu
del gpy_gpu
del gpz_gpu
del gpv_gpu
del rad_pan_gpu
del rad_tilt_gpu
del times_gpu
del rb_gpu
del fp_gpu
del ref_gpu
del param_gpu
mempool.free_all_blocks()

rc_data = ifft(fft(pulses, radar.fft_len * upsample, axis=0) * match_filt[:, None], n=radar.nsam, axis=0)

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

plt.figure('Doppler Shifted Data')
plt.imshow(db(fftshift(fft(rc_data, axis=1))))
plt.axis('tight')

plt.figure('Original Data')
plt.imshow(db(env.data), origin='lower')

plt.figure('Sampled Data')
plt.scatter(gx, gy, s=.1, c=db(gv))

az_diffs = np.array([adiff(pt_az[i], radar.pan(radar.times[i])) for i in range(len(rngs))])
plt.figure('Params')
plt.subplot(3, 1, 1)
plt.title('Doppler Chirp')
plt.plot(radar.times, np.cos(4 * np.pi / radar.wavelength * rngs))
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
plt.plot(db(ifft(fft(r_chirp) * match_filt)))
plt.plot(db(ifft(fft_chirp * match_filt)))

plt.figure('Range Profile')
plt.imshow(np.real(range_prof[:np.arange(len(radar.range_bins))[radar.range_bins < rngs.max()][-1] + 5]))
plt.axis('tight')


plt_test = fft(range_prof, axis=0)[:, 0] * fft_chirp
plt.figure('Slices')
plt.subplot(4, 1, 1)
plt.title('Shifted Spectrum')
plt.plot(db(plt_test))
plt.subplot(4, 1, 2)
plt.title('Time Series')
plt.plot(np.real(ifft(plt_test)))
plt.subplot(4, 1, 3)
plt.title('Range Compression')
plt.plot(db(ifft(plt_test * match_filt, n=radar.nsam)))
plt.plot(db(rc_data[:, 0]))
plt.legend(['CPU Comp.', 'GPU Comp.'])


