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
import imageio
from itertools import combinations

from cuda_kernels import genRangeProfile, backproject
from environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from pathlib import Path

from radar import Radar, RadarArray, ambiguity, genPulse
from rawparser import loadReferenceChirp, loadMatchedFilter, loadASHFile, getRawDataGen, getRawSDRParams
from simlib import getElevation, enu2llh
from useful_lib import findAllFilenames, factors, db, findPowerOf2, gaus


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def pdf_dist(pmax, size):
    dist = lambda xx: gaus(xx, 1, 0, size) if np.any(abs(xx)) < size else 1
    bounds = (-size * 3, size * 3)
    while True:
        x = np.random.rand(size) * (bounds[1] - bounds[0]) + bounds[0]
        y = np.random.rand(size) * pmax
        if np.all(y <= dist(x)):
            return x


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
        _gz, _gv = e(_gx, _gy)
    return _gx, _gy, _gz, _gv


''' CONSTANTS '''
c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
MAX_SAMPLE_SIZE = 1e6
plt.close('all')

''' FILENAMES '''
fnme = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112239.sar'
output_fnme = '/home/jeff/repo/rfi_mitigation/rfi_sim.dat'

''' CUSTOM OPTIONS '''
threads_per_block = (16, 16)
upsample = 1
chunk_sz = 256 // upsample
n_samples = 1000000
presum = 1
noise_level = 1
poly_interp = 1
rfi_percent = .2
subgrid_size = (500, 500)
bpj_size = (150, 150)
rand_method = 'trap'
scp = (40.141216, -111.699566)

''' FLAGS '''
do_backproject = False
introduce_rfi = True
write_to_file = True
use_background_image = False

# Load in files
files = findAllFilenames(fnme)
scp_el = (scp[0], scp[1], getElevation(scp))

# Initialize all the variables
match_filt = None
gpx_gpu = gpy_gpu = gpz_gpu = gpv_gpu = None
n_splits = rcd_gpu = rpf_gpu = None
check_tt = None
gx = gy = gz = gv = None

# Make the user aware of everything that's going on
if introduce_rfi:
    print('Introducing RFI.')
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
    im = imageio.imread('/home/jeff/Pictures/max.png')
    crop_im = np.flipud(np.sum(im, axis=2))
    env = Environment(llr=scp_el, bg_data=crop_im, bg_res=(.85, .85), subgrid_size=subgrid_size, dec_fac=1)
else:
    print('Using .asi image.')
    env = Environment(ashfile=files['ash'], asifile=files['asi'], subgrid_size=subgrid_size, dec_fac=8)
shift_gpu = cupy.array(np.ascontiguousarray([[0, 0]]), dtype=np.float64)
scp_offsets = env.getOffset(*scp, False)

print('Loading radar list...')
ra = RadarArray()

# List out the radar chirp and relative position from center of array for each antenna/channel
ra_pos = [(np.linspace(0, 1, 100), np.array([0, 0, 0]))]
# chirp_py = ['/home/jeff/repo/mimo_simulator/frank500.dat']
for sub_radar in ra_pos:
    r_rx = Radar(fnme, env.scp, offsets=sub_radar[1], use_xml_flightpath=False, presum=presum)
    try:
        r_rx.genChirp(px=np.linspace(0, 1, 100), py=sub_radar[0])
    except IndexError:
        r_rx.loadChirpFromFile(sub_radar[0])
    r_rx.upsampleData(upsample)
    ra.append(r_rx)

# Get all the stuff associated with the overall array
ra.calcFactors()
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
rb_gpu = cupy.array(ra.range_bins, dtype=np.float64)
blocks_per_grid_bpj = (
    int(np.ceil(bx.shape[0] / threads_per_block[0])), int(np.ceil(bx.shape[1] / threads_per_block[1])))

n_frames, sdr_samples, atts, sys_times = getRawSDRParams(files['RawData'])

if introduce_rfi:
    p_hasRFI = np.zeros((n_frames,))

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

# Now run pulse gen/backprojection for each of the antennae
print('Running range profile generation...')
overall_bpj = np.zeros(bx.shape, dtype=np.complex128)
# Get minimum slant range (which should be the center of our aperture)
pca_rngs = np.linalg.norm(ra.pos(ra.times), axis=0)
t_pca = 0
try:
    # noinspection PyArgumentList
    t_pca = ra.times[pca_rngs == pca_rngs.min()][0]
except IndexError:
    t_pca = t_pca[0]
R0 = pca_rngs.min()

bpjgrid = [np.zeros(bx.shape, dtype=np.complex128) for r in ra]
n_pulses = len(ra.times) if ra.is_presummed else n_frames
for ch in tqdm(np.arange(0, n_pulses, chunk_sz)):
    tt = ra.systimes[ch:min(ch + chunk_sz, n_pulses)] / TAC
    ch_sz = len(tt)
    rpf_shape = (ra.upsample_nsam, ch_sz)
    blocks_per_grid_rpf = (
        int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(n_samples / threads_per_block[1])))
    for ant_num, r_rx in enumerate(ra):
        rcd_gpu = rpf_gpu = None
        # Load receiver antenna data, since that will not change
        pd_gpu = cupy.zeros(rpf_shape, dtype=np.complex128)
        ref_gpu = cupy.array(np.tile(r_rx.fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
        mf_gpu = cupy.array(np.tile(r_rx.mf, (ch_sz, 1)).T, dtype=np.complex128)

        # Toss in all the interpolated stuff
        fprx_gpu = cupy.array(np.ascontiguousarray(r_rx.pos(tt), dtype=np.float64))
        panrx_gpu = cupy.array(np.ascontiguousarray(r_rx.pan(tt), dtype=np.float64))
        tiltrx_gpu = cupy.array(np.ascontiguousarray(r_rx.tilt(tt), dtype=np.float64))

        # Load constants and other parameters
        paramrx_gpu = cupy.array(np.array([np.pi / r_rx.el_bw, np.pi / r_rx.az_bw,
                                           r_rx.wavelength, r_rx.velocity, r_rx.el_bw, r_rx.az_bw,
                                           r_rx.near_slant_range / c0, fs * upsample, poly_interp]), dtype=np.float64)

        for tx in ra:
            fptx_gpu = cupy.array(np.ascontiguousarray(tx.pos(tt), dtype=np.float64))
            pantx_gpu = cupy.array(np.ascontiguousarray(tx.pan(tt), dtype=np.float64))
            tilttx_gpu = cupy.array(np.ascontiguousarray(r_rx.tilt(tt), dtype=np.float64))

            # Load constants and other parameters
            paramtx_gpu = cupy.array(np.array([np.pi / tx.el_bw, np.pi / tx.az_bw,
                                               tx.wavelength, tx.velocity, tx.el_bw, tx.az_bw,
                                               tx.near_slant_range / c0, fs * upsample, 1]), dtype=np.float64)
            ref_gpu = cupy.array(np.tile(tx.fft_chirp, (ch_sz, 1)).T, dtype=np.complex128)
            rpf_r_gpu = cupy.random.normal(0, noise_level, rpf_shape, dtype=np.float64)
            rpf_i_gpu = cupy.random.normal(0, noise_level, rpf_shape, dtype=np.float64)

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
                rbi_t = np.arange(tx.nsam) / fs
                rbi_envelope = gaus(np.linspace(-1, 1, ch_sz), .1, 0, .02)
                nbi_corrupt = np.convolve(np.random.rand(ch_sz) < rfi_percent / 2,
                                          rbi_envelope, mode='same') + 1
                nbi_signal = np.random.rand() * .1 * np.exp(1j * 2 * np.pi * np.random.rand() * tx.bandwidth * rbi_t)
                nbi_signal = np.tile(nbi_signal, (ch_sz, 1)).T * nbi_corrupt[None, :]

                # Don't add as much WBI since it doesn't seem as prevalent in the real world
                wbi_corrupt = np.convolve(np.random.rand(ch_sz) < rfi_percent / 7,
                                          rbi_envelope, mode='same') + 1

                # Simple linear chirp with random bandwidth, use a percentage of the
                wbi_signal = np.random.rand() * .1 * genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), tx.nsam,
                                                              tx.nsam / fs, np.random.rand() * tx.bandwidth,
                                                              max(np.random.rand(), .001) * .1 * tx.bandwidth)
                wbi_signal = np.tile(wbi_signal, (ch_sz, 1)).T * wbi_corrupt[None, :]
                rbi_gpu = cupy.array(nbi_signal + wbi_signal, dtype=np.complex128)
                rbi_gpu = cupy.fft.fft(rbi_gpu, axis=0, n=tx.fft_len)
                rbi_gpu[:, np.logical_and(nbi_corrupt == 1, wbi_corrupt == 1)] = 1.

                # Add RFI signals to the data generation FFTs
                pd_gpu = pd_gpu + cupy.fft.ifft(
                    cupy.fft.fft(rpf_gpu, n=tx.fft_len, axis=0) * ref_gpu * rbi_gpu,
                    axis=0)[:r_rx.upsample_nsam, :]

                p_hasRFI[ch:ch + ch_sz] = np.logical_and(nbi_corrupt != 1, wbi_corrupt != 1)

                # Delete the GPU RFI stuff
                del rbi_gpu
            else:
                pd_gpu = pd_gpu + cupy.fft.ifft(cupy.fft.fft(rpf_gpu, n=tx.fft_len, axis=0) * ref_gpu,
                                                axis=0)[:r_rx.upsample_nsam, :]
            cupy.cuda.Device().synchronize()

            del rpf_r_gpu
            del rpf_i_gpu
            del fptx_gpu
            del ref_gpu

        # Run the actual backprojection
        if do_backproject:
            rcd_gpu = cupy.fft.ifft(cupy.fft.fft(pd_gpu, n=r_rx.fft_len, axis=0) * mf_gpu,
                                    axis=0)[:r_rx.upsample_nsam, :]
            bpjgrid_gpu = cupy.zeros(bx.shape, dtype=np.complex128)
            backproject[blocks_per_grid_bpj, threads_per_block](fprx_gpu, fprx_gpu, bx_gpu, by_gpu, bz_gpu,
                                                                rb_gpu, panrx_gpu,
                                                                tiltrx_gpu, rcd_gpu, bpjgrid_gpu,
                                                                paramrx_gpu)
            cupy.cuda.Device().synchronize()
            bpjgrid[ant_num] = bpjgrid[ant_num] + bpjgrid_gpu.get()
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
        if tt[-1] >= t_pca >= tt[0]:
            if do_backproject:
                rc_data[ant_num] = rcd_gpu.get()
            pulses[ant_num] = pd_gpu.get()
            range_prof[ant_num] = rpf_gpu.get()
            check_tt = tt

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
mempool.free_all_blocks()

pca_idx = np.where(check_tt == t_pca)[0][0]

# After run, diagnostics, etc.
if write_to_file:
    rfi_fnme = output_fnme[:-4] + '_rfi.dat'
    if not Path(rfi_fnme).exists():
        Path(rfi_fnme).touch()
    else:
        with open(rfi_fnme, 'wb') as f:
            f.write(np.int8(p_hasRFI).tobytes())

print('File saved to ' + output_fnme)
# Generate all the plot figures we'll need
grid_num = 1
if ra.num < 3:
    if do_backproject:
        bpj_fig, bpj_ax = plt.subplots(1, ra.num, num='Backproject')
        dopp_fig, dopp_ax = plt.subplots(1, ra.num, num='Doppler Shifted')
        rng_fig, rng_ax = plt.subplots(1, ra.num, num='Range Profiles')
else:
    grid_num = int(np.ceil(np.sqrt(ra.num)))
    if do_backproject:
        bpj_fig, bpj_ax = plt.subplots(grid_num, grid_num, num='Backproject')
        dopp_fig, dopp_ax = plt.subplots(grid_num, grid_num, num='Doppler Shifted')
        rng_fig, rng_ax = plt.subplots(grid_num, grid_num, num='Range Profiles')
param_fig, param_ax = plt.subplots(2, 1, num='Params')
chirp_fig, chirp_ax = plt.subplots(4, ra.num, num='Chirp Data')
if do_backproject:
    slice_fig, slice_ax = plt.subplots(3, ra.num, num='Slices')
    pulse_fig, pulse_ax = plt.subplots(2, ra.num, num='Pulse Data')

for ant_num, r_rx in enumerate(ra):
    print('Antenna {}'.format(ant_num))
    pos_pca = enu2llh(*r_rx.pos(t_pca), env.scp)
    pca_slant_range = np.linalg.norm(r_rx.pos(t_pca))
    print('PCA-SCP slant range is {:.2f}'.format(pca_slant_range))
    print('PCA-SCP ground range is {:.2f}'.format(np.sqrt(pca_slant_range ** 2 - (pos_pca[2] - env.scp[2]) ** 2)))
    print('PCA radar pan is {:.2f}'.format(r_rx.pan(t_pca) / DTR))
    print('PCA radar tilt is {:.2f}'.format(r_rx.tilt(t_pca) / DTR))
    print('Plane pos at PCA is {:.6f}, {:.6f}, {:.2f}'.format(*pos_pca))
    sh_los = -r_rx.pos(r_rx.times)
    rngs = np.linalg.norm(sh_los, axis=0)
    pt_el = np.array([math.asin(-sh_los[2, i] / rngs[i]) for i in range(len(rngs))])
    pt_az = np.array([math.atan2(sh_los[0, i], sh_los[1, i]) for i in range(len(rngs))])
    ant_x_plot = ant_num // grid_num
    ant_y_plot = ant_num % grid_num

    az_diffs = np.array([adiff(pt_az[i], r_rx.pan(r_rx.times[i])) for i in range(len(rngs))])
    param_ax[0].set_title('Ranges')
    param_ax[0].plot(r_rx.times, rngs)
    param_ax[0].plot(r_rx.times[np.logical_and(r_rx.times <= check_tt[-1], r_rx.times >= check_tt[0])],
                     rngs[np.logical_and(r_rx.times <= check_tt[-1], r_rx.times >= check_tt[0])])
    param_ax[1].set_title('Az Diffs')
    param_ax[1].plot(r_rx.times, az_diffs)
    param_ax[1].plot(r_rx.times[np.logical_and(r_rx.times <= check_tt[-1], r_rx.times >= check_tt[0])],
                     az_diffs[np.logical_and(r_rx.times <= check_tt[-1], r_rx.times >= check_tt[0])])

    # plt.figure('Chirp Data Ant {}'.format(ant_num))
    try:
        chirp_ax[0, ant_num].set_title('Ref. Chirp')
        chirp_ax[0, ant_num].plot(np.real(r_rx.chirp))
        chirp_ax[1, ant_num].set_title('Ref. Spectrum')
        chirp_ax[1, ant_num].plot(db(fft(r_rx.chirp)))
        chirp_ax[2, ant_num].set_title('Matched Filter')
        chirp_ax[2, ant_num].plot(db(r_rx.mf))
        chirp_ax[3, ant_num].set_title('Range Compression')
        chirp_ax[3, ant_num].plot(r_rx.range_bins,
                                  np.fft.fftshift(db(ifft(r_rx.fft_chirp * r_rx.mf))[:r_rx.upsample_nsam]))
    except IndexError:
        chirp_ax[0].set_title('Ref. Chirp')
        chirp_ax[0].plot(np.real(r_rx.chirp))
        chirp_ax[1].set_title('Ref. Spectrum')
        chirp_ax[1].plot(db(fft(r_rx.chirp)))
        chirp_ax[2].set_title('Matched Filter')
        chirp_ax[2].plot(db(r_rx.mf))
        chirp_ax[3].set_title('Range Compression')
        chirp_ax[3].plot(r_rx.range_bins,
                         np.fft.fftshift(db(ifft(r_rx.fft_chirp * r_rx.mf))[:r_rx.upsample_nsam]))

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
        if ra.num < 3:
            try:
                rng_ax[ant_num].imshow(
                    np.real(range_prof[ant_num][:np.arange(len(r_rx.range_bins))[r_rx.range_bins < rngs.max()][-1] + 5]))
                rng_ax[ant_num].axis('tight')
                # plt.figure('Backproject Ant {}'.format(ant_num))
                db_bpj = db(bpjgrid[ant_num]).T
                bpj_ax[ant_num].imshow(db_bpj, origin='lower', cmap='ocean',
                                       clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])

                # plt.figure('Doppler Shifted Data Ant {}'.format(ant_num))
                dopp_ax[ant_num].imshow(dpshift_data)
                dopp_ax[ant_num].axis('tight')
            except TypeError:
                rng_ax.imshow(
                    np.real(range_prof[ant_num][:np.arange(len(r_rx.range_bins))[r_rx.range_bins < rngs.max()][-1] + 5]))
                rng_ax.axis('tight')
                # plt.figure('Backproject Ant {}'.format(ant_num))
                db_bpj = db(bpjgrid[ant_num]).T
                bpj_ax.imshow(db_bpj, origin='lower', cmap='ocean',
                              clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])

                # plt.figure('Doppler Shifted Data Ant {}'.format(ant_num))
                dopp_ax.imshow(dpshift_data)
                dopp_ax.axis('tight')
        else:
            rng_ax[ant_x_plot, ant_y_plot].imshow(
                np.real(range_prof[ant_num][:np.arange(len(r_rx.range_bins))[r_rx.range_bins < rngs.max()][-1] + 5]))
            rng_ax[ant_x_plot, ant_y_plot].axis('tight')
            # plt.figure('Backproject Ant {}'.format(ant_num))
            db_bpj = db(bpjgrid[ant_num]).T
            bpj_ax[ant_x_plot, ant_y_plot].imshow(db_bpj, origin='lower', cmap='ocean',
                                                  clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])

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
            slice_ax[2, ant_num].plot(r_rx.range_bins, db(rc_data[ant_num][:, pca_idx]))

            # plt.figure('Pulses Ant {}'.format(ant_num))
            pulse_ax[0, ant_num].imshow(db(pulses[ant_num]), extent=[check_tt[0], check_tt[-1], ra.range_bins[0],
                                                                     ra.range_bins[-1]],
                                        origin='lower')
            pulse_ax[0, ant_num].axis('tight')
            pulse_ax[1, ant_num].imshow(db(rc_data[ant_num]), extent=[check_tt[0], check_tt[-1], ra.range_bins[0],
                                                                      ra.range_bins[-1]],
                                        origin='lower')
            pulse_ax[1, ant_num].axis('tight')
        except IndexError:
            slice_ax[0].set_title('Shifted Spectrum')
            slice_ax[0].plot(db(fft(pulses[ant_num][:, pca_idx], n=r_rx.fft_len)))
            slice_ax[1].set_title('Time Series')
            slice_ax[1].plot(np.real(pulses[ant_num][:, pca_idx]))
            slice_ax[2].set_title('Range Compression')
            slice_ax[2].plot(r_rx.range_bins, db(rc_data[ant_num][:, pca_idx]))

            # plt.figure('Pulses Ant {}'.format(ant_num))
            pulse_ax[0].imshow(db(pulses[ant_num]), extent=[check_tt[0], check_tt[-1], ra.range_bins[0],
                                                            ra.range_bins[-1]],
                               origin='lower')
            pulse_ax[0].axis('tight')
            pulse_ax[1].imshow(db(rc_data[ant_num]), extent=[check_tt[0], check_tt[-1], ra.range_bins[0],
                                                             ra.range_bins[-1]],
                               origin='lower')
            pulse_ax[1].axis('tight')

        if n_samples == 1:
            b0 = np.arange(ra.upsample_nsam)[rngs.min() <= ra.range_bins][-1]
            b1 = np.arange(ra.upsample_nsam)[ra.range_bins <= rngs.min()][0]
            boca = b0 if abs(ra.range_bins[b0] - rngs.min()) < abs(ra.range_bins[b1] - rngs.min()) else b1
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
        plt.imshow(db_bpj, origin='lower', cmap='ocean',
                   clim=[db_bpj.mean() - db_bpj.std() * 2, db_bpj.mean() + db_bpj.std() * 2])

plt.figure('Original Data')
plt.imshow(db(env.data), origin='lower', cmap='ocean', clim=[40, 100])

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
rads = [pair for pair in combinations(ra.radars, 2)]
rad_labs = [pair for pair in combinations(ra.names, 2)]
nplots = ra.num + len(rads)
nrad = 1
for n, rx in enumerate(ra):
    amb = ambiguity(rx.chirp, rx.chirp, rx.prf, 64, mag=False)
    ax = amb_fig.add_subplot(2, ra.num, nrad)
    ax.imshow(db(amb[0]), cmap='jet')
    nrad += 1

for n, pair in enumerate(rads):
    amb = ambiguity(pair[0].chirp, pair[1].chirp, pair[0].prf, 64, mag=False)
    ax = amb_fig.add_subplot(2, ra.num, nrad)
    ax.set_title('{}, {}'.format(*rad_labs[n]))
    ax.imshow(db(amb[0]), cmap='jet')
    nrad += 1

plt.show()
