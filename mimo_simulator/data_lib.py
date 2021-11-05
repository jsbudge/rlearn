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

from radar import XMLRadar, ConfigRadar, RadarArray, ambiguity, genPulse, Platform, XMLChannel, Antenna
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

            _gx = gx_tmp + e(gx_tmp, gy_tmp, override_data=grads[0])[1]  # + np.random.rand(n_pts) - .5
            _gy = gy_tmp + e(gx_tmp, gy_tmp, override_data=grads[1])[1]  # + np.random.rand(n_pts) - .5

        _gz, _gv = e(_gx, _gy)
    return _gx, _gy, _gz, _gv


''' CONSTANTS '''
c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
MAX_SAMPLE_SIZE = 1e6
INT_16_MAX = 2 ** 15


def genData(fnme, cust_bg=None, threads_per_block=(16, 16), chunk_sz=128, n_samples=1000000, presum=1, noise_level=.1,
            poly_interp=1, n_rfi_pts=1, rfi_perc=.33, subgrid_size=(150, 150), rand_method='contrast',
            scp=(40.098785, -111.659957), introduce_rfi=False, use_background_image=False, use_config_radar=False):

    # Load in files
    files = findAllFilenames(fnme, debug_dir='/data5/SAR_Freq_Data')
    scp_el = (scp[0], scp[1], getElevation(scp))

    # Initialize all the variables
    rfix_gpu = rfiy_gpu = rfiz_gpu = rfip_gpu = None
    rfi_sigs = rfi_ts = None
    n_splits = None

    # Make the user aware of everything that's going on
    if introduce_rfi:
        print('Introducing {} RFI points.'.format(n_rfi_pts))

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
        env = Environment(ashfile=ash_fnme, asifile=asi_fnme, subgrid_size=subgrid_size, dec_fac=8)
    shift_gpu = cupy.array(np.ascontiguousarray([[0, 0]]), dtype=np.float64)
    scp_offsets = env.getOffset(*scp, False)

    print('Loading radar list...')
    # List out the radar chirp and relative position from center of array for each antenna/channel
    # Custom waveforms:
    # (np.linspace(0, 1, 100), np.array([0, 0, 0]))
    # Load in from Strider:
    # ('/home/jeff/repo/mimo_simulator/waveforms/p4_400_300.wave', np.array([10, 1, 15]))
    ra_pos = [np.linspace(0, 1, 100)]
    sdr_f = SDRParse(files['sar'])
    ra = Platform(fnme, env.scp, use_xml_flightpath=False, sarfile=sdr_f)
    for idx, sr in enumerate(ra_pos):
        ra.addChannel(XMLChannel(sdr_f.xml['Channel_0'], sdr_f[0], env.scp, ra.alt, 0, presum=1))
        try:
            ra[idx].genChirp(px=np.linspace(0, 1, 100), py=sr)
            # ra[idx].loadChirp(sdr_f[idx].ref_chirp)
        except ValueError:
            ra[idx].loadChirpFromFile(sr)

    # Get all the stuff associated with the overall array
    n_frames = ra[0].nframes

    # Load memory for CUDA processing
    mempool = cupy.get_default_memory_pool()

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
        rfi_x, rfi_y, rfi_z, _ = genPoints(n_rfi_pts, env, scp_offsets, method=rand_method)
        tmp_prcs = np.random.rand(n_rfi_pts)
        rfi_ts = tmp_prcs / sum(tmp_prcs) * rfi_perc
        rfi_sigs = []
        for sig in range(n_rfi_pts):
            rfi_sigs.append((1e2, np.random.randint(10, 70)))
    else:
        p_hasRFI = None

    '''
    #################### MAIN LOOP ########################
    '''

    for rx_num, chan in enumerate(ra):
        for ch in np.arange(0, chan.nframes, chunk_sz):
            tt = chan.systimes[ch:min(ch + chunk_sz, chan.nframes)] / TAC
            ch_sz = len(tt)

            # Figure out our memory scheme
            rpf_shape = (chan.nsam, ch_sz)
            blocks_per_grid_rpf = (
                int(np.ceil(rpf_shape[1] / threads_per_block[0])), int(np.ceil(n_samples / threads_per_block[1])))
            rx = ra.rx(channel=rx_num)

            # Load receiver antenna data, since that will not change
            pd_gpu = cupy.zeros(rpf_shape, dtype=np.complex128)

            # Toss in all the interpolated stuff
            fprx_gpu = cupy.array(np.ascontiguousarray(rx.pos(tt), dtype=np.float64))
            panrx_gpu = cupy.array(np.ascontiguousarray(rx.pan(tt), dtype=np.float64))
            tiltrx_gpu = cupy.array(np.ascontiguousarray(rx.tilt(tt), dtype=np.float64))

            # Load constants and other parameters
            paramrx_gpu = cupy.array(np.array([np.pi / rx.el_bw, np.pi / rx.az_bw,
                                               chan.wavelength, 0, rx.el_bw, rx.az_bw,
                                               chan.near_slant_range / c0, fs, poly_interp]), dtype=np.float64)

            for tx_num, tx_chan in enumerate(ra):
                tx = ra.tx(channel=tx_num)
                fptx_gpu = cupy.array(np.ascontiguousarray(tx.pos(tt), dtype=np.float64))
                pantx_gpu = cupy.array(np.ascontiguousarray(tx.pan(tt), dtype=np.float64))
                tilttx_gpu = cupy.array(np.ascontiguousarray(tx.tilt(tt), dtype=np.float64))

                # Load constants and other parameters
                paramtx_gpu = cupy.array(np.array([np.pi / tx.el_bw, np.pi / tx.az_bw,
                                                   tx_chan.wavelength, 0, tx.el_bw, tx.az_bw,
                                                   tx_chan.near_slant_range / c0, fs, 1]), dtype=np.float64)
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
                                nbi_signal[:, npulse] = sig[0] * getTVSignal(sig[1], chan.nsam)
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

            yield pd_gpu.get()

            # Delete the range compressed pulse block to free up memory on the GPU
            del rpf_gpu
            del pd_gpu
            del panrx_gpu
            del tiltrx_gpu
            del fprx_gpu
            mempool.free_all_blocks()

    # Delete all of our other parameters and free the GPU memory
    del paramrx_gpu
    del gpx_gpu
    del gpy_gpu
    del gpz_gpu
    del gpv_gpu
    mempool.free_all_blocks()


if __name__ == '__main__':
    """ FILENAMES """
    fnme = '/data5/SAR_DATA/2021/09222021/SAR_09222021_163338.sar'
    cust_bg = '/home/jeff/Downloads/josh.png'

    test = genData(fnme, cust_bg)
    pulses = next(test)

    plt.figure('Pulse Data')
    plt.imshow(db(pulses))
    plt.axis('tight')