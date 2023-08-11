import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, least_squares, minimize_scalar
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt, find_peaks
import cupy as cupy
import cupyx.scipy.signal
from cuda_kernels import cpudiff
from numba import cuda, njit
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import SDRParse, load, jumpCorrection
from simulation_functions import azelToVec

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def calcDoppler(vel, az, el, prf, ch):
    ret = ((c0 + vel.dot(
        azelToVec(az, el))) / c0 * ch.fc - ch.fc) % (prf)
    ind = np.nonzero(ret > prf / 2)
    if isinstance(ret, float):
        if ret > prf / 2:
            ret -= prf
        elif ret < -prf / 2:
            ret += prf
    else:
        ret[ind] -= prf
        ind = np.nonzero(ret < -prf / 2)
        ret[ind] += prf
    return ret


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
upsample = 4
dopp_upsample = 1
chunk_sz = 15000
init_pulse = 1
n_iter = 1

# This is the file used to backproject data
# bg_file = '/data5/SAR_DATA/2021/12072021/SAR_12072021_165650.sar'
# bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
# bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
# bg_file = '/data6/SAR_DATA/2023/03292023/45/SAR_03292023_072613.sar'
bgfs = ['/data6/SAR_DATA/2023/03292023/45/SAR_03292023_072613.sar',
        '/data6/SAR_DATA/2023/03292023/45/SAR_03292023_072456.sar',
        '/data6/SAR_DATA/2023/03292023/45/SAR_03292023_072716.sar']
channels = [[0, 3], [0], [0, 3]]
bgfs = ['/data6/SAR_DATA/2023/03302023/SAR_03302023_082243.sar']
channels = [[0]]

total_lssol = []
total_times = []
total_phases = []
# Now, given a point on the ground, find a line of best fit
check_pt = np.array([30.562646, -86.436416, 0])
likely_spot = np.array([30.562867, -86.436460,  45])
ec, nc, uc = llh2enu(*check_pt, check_pt)
es, ns, us = llh2enu(*likely_spot, check_pt)
xo = np.array([es, ns, us]) - np.array([ec, nc, uc])
min_rngs = []

pvec = -np.array([0.33216719, 0.06322961, 0.00075249])
# pvec = -np.array([.5, .5, 0.])
pvec /= np.linalg.norm(pvec)

for bg_idx, bg_file in enumerate(bgfs):
    sdr_f = load(bg_file)

    for ch in channels[bg_idx]:
        chan = sdr_f[ch]
        dopp_fft_len = findPowerOf2(chunk_sz) * dopp_upsample
        dopp_freq = np.fft.fftfreq(dopp_fft_len, 1 / chan.prf)
        nrange = ((chan.xml['Receive_On_TAC'] - chan.xml['Transmit_On_TAC']) / TAC -
                  chan.pulse_length_S * .3) * c0
        bin_sz = c0 / (chan.fs * upsample)
        rbins = nrange + bin_sz * np.arange(chan.nsam * upsample)
        wavelength = c0 / chan.fc
        wavenumber = 2 * np.pi / wavelength
        spd = np.linalg.norm(sdr_f.gps_data[['ve', 'vn', 'vu']], axis=1)
        gps_dist = np.interp(chan.pulse_time, sdr_f.gps_data.index, np.cumsum(spd) * .01)
        lsline = np.outer(pvec, gps_dist).T + np.array([es, ns, us])

        fft_len = findPowerOf2(chan.nsam + chan.pulse_length_N)
        mfilt = GetAdvMatchedFilter(chan, fft_len=fft_len)
        mfilt = cupy.array(np.tile(mfilt, (chunk_sz, 1)).T, dtype=np.complex128)
        range_freq = np.fft.fftfreq(fft_len, 1 / chan.fs)

        nf_todo = min(chan.nframes - init_pulse, chunk_sz * n_iter)
        calc_rngs = np.zeros((nf_todo,))
        calc_dopps = np.zeros((nf_todo,))
        phases = np.zeros((nf_todo,))
        brngs = np.zeros((nf_todo,))

        it = 0
        phase_chunk = 1024
        max_rng = -1
        for f in tqdm(np.arange(init_pulse, init_pulse + nf_todo, phase_chunk)):
            frames = f + np.arange(min(phase_chunk, init_pulse + nf_todo - f))
            pulses_gpu = cupy.zeros((fft_len * upsample, len(frames)), dtype=np.complex128)
            pulse_fft = cupy.fft.fft(cupy.array(sdr_f.getPulses(frames, ch), dtype=np.complex128),
                                     fft_len, axis=0) * mfilt[:, :len(frames)]
            pulses_gpu[:fft_len // 2, :] = pulse_fft[:fft_len // 2, :]
            pulses_gpu[-fft_len // 2:, :] = pulse_fft[-fft_len // 2:, :]
            pulses_gpu = cupy.fft.ifft(pulses_gpu, axis=0)[:chan.nsam * upsample, :]
            pulses = pulses_gpu.get()
            dbpulses = db(pulses)

            del pulses_gpu

            if max_rng < 0:
                max_rng = int(np.where(dbpulses[:, 0] == dbpulses[:, 0].max())[0][0])
            pmax = np.zeros(pulses.shape[1]).astype(int)
            for n in range(pulses.shape[1]):
                pmax[n] = np.where(
                    dbpulses[max_rng - 2:max_rng + 2, n] == dbpulses[max_rng - 2:max_rng + 2, n].max())[0][0] + max_rng - 2
                max_rng = pmax[n]
            for pidx, p in enumerate(pmax[1:]):
                if abs(p - pmax[pidx]) > 2:
                    pmax[pidx + 1] = np.where(db(pulses[pmax[pidx] - 2:pmax[pidx] + 2, pidx + 1]) == db(
                        pulses[pmax[pidx] - 2:pmax[pidx] + 2, pidx + 1]).max())[0][0] + pmax[pidx] - 2

            for idx in range(len(frames)):
                brngs[it * phase_chunk + idx] = rbins[pmax[idx]]
                phases[it * phase_chunk + idx] = np.angle(pulses[pmax[idx], idx])
            it += 1

        phases = np.unwrap(phases)
        offset = -np.angle(np.exp(-1j * wavenumber * brngs[0]))
        calc_rngs = brngs[0] - (phases + offset) / (2 * np.pi) * wavelength
        calc_rngs += brngs.mean() - calc_rngs.mean()
        calc_rngs = gaussian_filter(calc_rngs, 15)

        range_poly = np.poly1d(np.polyfit(np.arange(nf_todo), calc_rngs, 2))(np.arange(nf_todo))

        def minfunc(x):
            return np.linalg.norm(
                np.linalg.norm(
                    pvec * x[0] * (chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter] - chan.pulse_time[init_pulse])[:, None] +
                    np.array([es, ns, us]) - np.array([x[1], x[2], 0.]), axis=1) - calc_rngs)

        check = least_squares(minfunc, np.array([.33, 0, 0]), max_nfev=1e9, bounds=([.1, -20, -20], [1, 20, 20]))
        check_rng = np.linalg.norm(
                    pvec * check['x'][0] * (chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter] - chan.pulse_time[0])[:, None] +
                    np.array([es, ns, us]) - np.array([check['x'][1], check['x'][2], 0.]), axis=1)
        vel = .45
        delta_x = vel / chan.prf
        chirp_rate = chan.bw / chan.pulse_length_S

        dopp_f = int(
            calcDoppler(np.array([vel, 0, 0]), np.pi / 2, 0, chan.prf, chan) / chan.prf * dopp_fft_len)
        dopp_f = dopp_f if dopp_f % 2 == 0 else dopp_f + 1

        # OMEGA-K ALGORITHM
        for f in tqdm(np.arange(init_pulse, init_pulse + nf_todo, chunk_sz)):
            frames = f + np.arange(min(chunk_sz, init_pulse + nf_todo - f))
            Rt = calc_rngs[frames - f]
            Ro = Rt.min()
            fd, fr = np.meshgrid(dopp_freq, range_freq)
            kx = dopp_freq * 2 * np.pi / vel
            # kx = np.linspace(-np.pi / delta_x, np.pi / delta_x, len(frames))
            dkr = 2 * np.pi * range_freq / c0
            kr = 2 * np.pi * chan.fc / c0 + dkr
            ky0 = (kr.max() ** 2 - kx.max() ** 2) ** 0.5
            ky_delta = kr[1] - kr[0]  # Same spacing as kr to avoid aliasing during interpolation.
            # Ky axis after interpolation.
            ky_interp = np.arange(ky0, kr[-1], ky_delta)
            omega0 = 2 * np.pi * chan.fc
            azfilt = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), len(frames),
                                         chan.prf, 0, dopp_freq.max()), dopp_fft_len).conj()
            azfilt[:dopp_f // 2] *= taylor(dopp_f, sll=60)[-dopp_f // 2:]
            azfilt[-dopp_f // 2:] *= taylor(dopp_f, sll=60)[:dopp_f // 2]
            azfilt[dopp_f // 2:-dopp_f // 2] = 0
            azfilt = cupy.array(np.tile(azfilt, (chan.nsam, 1)), dtype=np.complex128)

            # Range error correction and range compression
            rpcomp = cupy.array(np.exp(1j * np.pi * np.outer(range_freq**2 / chirp_rate, np.ones(len(frames)))),
                               dtype=np.complex128)
            rcorr = cupy.array(np.exp(-2j * wavenumber * np.outer(np.ones(chan.nsam), Rt - check_rng)),
                               dtype=np.complex128)
            pulse_fft = cupy.fft.fft(cupy.array(sdr_f.getPulses(frames, ch), dtype=np.complex128),
                                     fft_len, axis=0) * mfilt[:, :len(frames)] * rpcomp
            pulse_fft = cupy.fft.ifft(pulse_fft, axis=0)[:chan.nsam, :] # * rcorr
            pulses = pulse_fft.get()
            cupy.get_default_memory_pool().free_all_blocks()

            # Take into the wavenumber domain
            pulse_fft = cupy.fft.fft2(pulse_fft, (fft_len, dopp_fft_len))

            # Range migration correction plus secondary range correction
            rmcm = cupy.array(np.exp(1j * Ro * c0 / 4 * kx**2 / omega0**2 * (2 * np.pi * fr - omega0)) *
                              np.exp(-1j * Ro * c0 / 4 * kx**2 / omega0**2 * (2 * np.pi * fr)**2 /
                                             (2 * np.pi * fr + omega0)),
                              dtype=np.complex128)

            # Stolt mapping
            cupy.get_default_memory_pool().free_all_blocks()

            # Final Azimuth IFFT
            pulse_fft = cupy.fft.ifft(pulse_fft * rmcm, axis=0)[:chan.nsam, :]
            pulse_fft = cupy.fft.ifft(pulse_fft * azfilt, axis=1)[:, :len(frames)]

            comp_pulses = np.fft.fftshift(pulse_fft.get(), axes=1)

            del pulse_fft
            del rmcm
            del azfilt
            del rcorr
            cupy.get_default_memory_pool().free_bytes()

        plt.figure(f'comp_{bg_file}_channel_{ch}')
        plt.imshow(db(comp_pulses))
        plt.colorbar()
        plt.clim(200, 250)
        plt.axis('tight')
        print(f'{vel} has a metric of {db(comp_pulses).max() - db(comp_pulses).mean()}')

        plt.figure(f'pulses_{bg_file}_channel_{ch}')
        plt.imshow(db(pulses))
        plt.colorbar()
        # plt.clim(200, 250)
        plt.axis('tight')

        plt.figure(f'phase_curve_{bg_file}_channel_{ch}')
        plt.plot(phases)
        plt.plot(np.unwrap(np.angle(comp_pulses[comp_pulses.shape[0] // 2, :])))

        plt.figure(f'cuts_{bg_file}_channel_{ch}')
        plt.subplot(2, 1, 1)
        plt.title('Azimuth')
        plt.plot(db(comp_pulses[comp_pulses.shape[0] // 2, :]))
        plt.subplot(2, 1, 2)
        plt.title('Range')
        plt.plot(db(comp_pulses[:, comp_pulses.shape[1] // 2]))

        plt.figure(f'calc_rngs_{bg_file}_channel_{ch}')
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], calc_rngs)
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], brngs)
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], range_poly)
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], check_rng)
        plt.legend(['calc_rngs', 'brngs', 'range_poly', 'check_rng'])
        # plt.plot(ptimes + chan.pulse_time[init_pulse], getRngs(ls_sol['x']))

        plt.figure(f'phase history_{bg_file}_channel_{ch}')
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], phases)
        plt.plot(chan.pulse_time[init_pulse:init_pulse + chunk_sz * n_iter], np.unwrap(np.angle(np.exp(-1j * wavenumber * calc_rngs))))


plt.show()

'''def doppminfunc(x):
    d = azelToVec(x[1], 0)
    l = azelToVec(x[2], x[3])
    res = np.array([[d[1] * l[2] * min_dopp_rng - x[0] * d[0] - xo[0]] for min_dopp_rng in min_rngs] +
                    [[-d[0] * l[2] * min_dopp_rng - x[0] * d[1] - xo[1]] for min_dopp_rng in min_rngs] +
                    [[(d[0] * l[1] - d[1] * l[0]) * min_dopp_rng - xo[2]] for min_dopp_rng in min_rngs] +
                    [[l.dot(d)]])
    return np.linalg.norm(res)

ls_sol = least_squares(doppminfunc, np.array([5, 259 * DTR, 259 * DTR - np.pi / 2, -45 * DTR, es, ns]), max_nfev=1e9)

pvec = azelToVec(ls_sol['x'][1], 0)

vels = np.zeros_like(calc_rngs)
vels[0] = 0

for idx in range(1, len(pvel)):
    mf = lambda x: (np.linalg.norm(xo + x * pvec) - calc_rngs[idx])**2
    vels[idx] = minimize_scalar(mf)['x']


def getRngs(x):
    ln = np.outer(ptimes, x[3] * azelToVec(x[0], 0)) + np.array([x[1], x[2], us])
    return np.sqrt((ln[:, 0] - x[4]) ** 2 + (ln[:, 1] - x[5]) ** 2 + (ln[:, 2] - 0.) ** 2)

plt.show()'''