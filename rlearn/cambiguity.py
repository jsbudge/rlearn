import numpy as np
import time
from scipy import signal
import numba
import multiprocessing
import itertools
import matplotlib.pyplot as plt
import os
from tftb.utils import nextpow2


@numba.jit(forceobj=True)
def xcor_numba(apple: np.ndarray, banana: np.ndarray) -> np.ndarray:
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)


def xcor(apple, banana):
    '''1D Cross-Correlation'''
    corr = signal.correlate(apple, banana, mode='same', method='fft')
    return np.abs(corr)


@numba.njit
def apply_fdoa_numba(ray: np.ndarray, fdoa: np.float64, samp_rate: np.float64) -> np.ndarray:
    precache = 2j * np.pi * fdoa / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray


def apply_fdoa(ray, fdoa, samp_rate):
    precache = 2j * np.pi * fdoa / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray


@numba.jit(forceobj=True)
def amb_surf_numba(needle: np.ndarray, haystack: np.ndarray, freqs_hz: np.float64, samp_rate: np.float64) -> np.ndarray:
    len_needle = len(needle)
    len_haystack = len(haystack)
    len_freqs = len(freqs_hz)
    assert len_needle == len_haystack
    surf = np.empty((len_freqs, len_needle))
    for fdx, freq_hz in enumerate(freqs_hz):
        shifted = apply_fdoa_numba(needle, freq_hz, samp_rate)
        surf[fdx] = xcor_numba(shifted, haystack)
    return surf


def amb_row_worker(args):
    needle, haystack, fdoa, samp_rate = args
    shifted = apply_fdoa(needle, fdoa, samp_rate)
    return xcor(shifted, haystack)


def amb_row_worker_numba(args):
    needle, haystack, fdoa, samp_rate = args
    shifted = apply_fdoa_numba(needle, fdoa, samp_rate)
    return xcor_numba(shifted, haystack)


def amb_surf_multiprocessing(needle, haystack, freqs_hz, samp_rate):
    len_needle = len(needle)
    len_haystack = len(haystack)
    len_freqs = len(freqs_hz)
    assert len_needle == len_haystack
    # surf = np.empty((len_freqs, len_needle))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        args = zip(
            itertools.repeat(needle),
            itertools.repeat(haystack),
            freqs_hz,
            itertools.repeat(samp_rate)
        )
        res = pool.map(amb_row_worker, args)
    return np.array(res)


def amb_surf_multiprocessing_numba(needle, haystack, freqs_hz, samp_rate):
    len_needle = len(needle)
    len_haystack = len(haystack)
    len_freqs = len(freqs_hz)
    assert len_needle == len_haystack
    # surf = np.empty((len_freqs, len_needle))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        args = zip(
            itertools.repeat(needle),
            itertools.repeat(haystack),
            freqs_hz,
            itertools.repeat(samp_rate)
        )
        res = pool.map(amb_row_worker_numba, args)
    return np.array(res)


def amb_surf(needle, haystack, freqs_hz, samp_rate):
    '''
    Returns the cross ambiguity function surface for a pair of signals.
    Parameters
    ----------
    needle : np.ndarray
        The signal of interest to localize within the haystack.
    haystack : np.ndarray
        The broader capture within which to localize the needle.
    freqs_hz : np.ndarray
        The frequency offsets to use in computing the CAF.
    samp_rate : float
        The sample rate for both the needle and the haystack.
    Returns
    -------
    surf : np.ndarray
        2D array of correlations of the needle in the haystack over frequency x lag.
    '''
    len_needle = len(needle)
    len_haystack = len(haystack)
    len_freqs = len(freqs_hz)
    assert len_needle == len_haystack
    surf = np.empty((len_freqs, len_needle))
    for fdx, freq_hz in enumerate(freqs_hz):
        shifted = apply_fdoa(needle, freq_hz, samp_rate)
        surf[fdx] = xcor(shifted, haystack)
    return surf


def narrow_band(s1, s2, lag=None, n_fbins=None):
    """Narrow band ambiguity function.

    :param signal: Signal to be analyzed.
    :param lag: vector of lag values.
    :param n_fbins: number of frequency bins
    :type signal: array-like
    :type lag: array-like
    :type n_fbins: int
    :return: Doppler lag representation
    :rtype: array-like
    """

    n = s1.shape[0]
    if lag is None:
        if n % 2 == 0:
            tau_start, tau_end = -n / 2 + 1, n / 2
        else:
            tau_start, tau_end = -(n - 1) / 2, (n + 1) / 2
        lag = np.arange(tau_start, tau_end)
    taucol = lag.shape[0]

    if n_fbins is None:
        n_fbins = s1.shape[0]

    naf = np.zeros((n_fbins, taucol), dtype=complex)
    for icol in range(taucol):
        taui = int(lag[icol])
        t = np.arange(abs(taui), n - abs(taui)).astype(int)
        naf[t, icol] = s1[t + taui] * np.conj(s2[t - taui])
    naf = np.fft.fft(naf, axis=0)

    _ix1 = np.arange((n_fbins + (n_fbins % 2)) // 2, n_fbins)
    _ix2 = np.arange((n_fbins + (n_fbins % 2)) // 2)

    _xi1 = -(n_fbins - (n_fbins % 2)) // 2
    _xi2 = ((n_fbins + (n_fbins % 2)) // 2 - 1)
    xi = np.arange(_xi1, _xi2 + 1, dtype=float) / n_fbins
    naf = naf[np.hstack((_ix1, _ix2)), :]
    return naf, lag, xi


def wide_band(s1, s2, fmin=None, fmax=None, N=None):
    if 1 in s1.shape:
        s1 = s1.ravel()
    elif s1.ndim != 1:
        raise ValueError("The input signal should be one dimensional.")
    s_ana = signal.hilbert(np.real(s1))
    s2_ana = signal.hilbert(np.real(s2))
    nx = s1.shape[0]
    m = int(np.round(nx / 2.0))
    t = np.arange(nx) - m
    tmin = 0
    tmax = nx - 1
    T = tmax - tmin

    # determine default values for fmin, fmax
    if (fmin is None) or (fmax is None):
        STF = np.fft.fftshift(s_ana)
        sp = np.abs(STF[:m]) ** 2
        maxsp = np.amax(sp)
        f = np.linspace(0, 0.5, m + 1)
        f = f[:m]
        indmin = np.nonzero(sp > maxsp / 100.0)[0].min()
        indmax = np.nonzero(sp > maxsp / 100.0)[0].max()
        if fmin is None:
            fmin = max([0.01, 0.05 * np.fix(f[indmin] / 0.05)])
        if fmax is None:
            fmax = 0.05 * np.ceil(f[indmax] / 0.05)
    B = fmax - fmin
    R = B / ((fmin + fmax) / 2.0)
    nq = np.ceil((B * T * (1 + 2.0 / R) * np.log((1 + R / 2.0) / (1 - R / 2.0))) / 2.0)
    nmin = nq - (nq % 2)
    if N is None:
        N = int(2 ** (nextpow2(nmin)))

    # geometric sampling for the analyzed spectrum
    k = np.arange(1, N + 1)
    q = (fmax / fmin) ** (1.0 / (N - 1))
    geo_f = fmin * (np.exp((k - 1) * np.log(q)))
    tfmatx = -2j * np.dot(t.reshape(-1, 1), geo_f.reshape(1, -1)) * np.pi
    tfmatx = np.exp(tfmatx)
    S = np.dot(s_ana.reshape(1, -1), tfmatx)
    S = np.tile(S, (nx, 1))
    Sb = np.dot(s2_ana.reshape(1, -1), tfmatx)
    Sb = np.tile(Sb, (nx, 1))
    Sb = Sb * tfmatx

    tau = t
    S = np.c_[S, np.zeros((nx, N))].T
    Sb = np.c_[Sb, np.zeros((nx, N))].T

    # mellin transform computation of the analyzed signal
    p = np.arange(2 * N)
    coef = np.exp(p / 2.0 * np.log(q))
    mellinS = np.fft.fftshift(np.fft.ifft(S[:, 0] * coef))
    mellinS = np.tile(mellinS, (nx, 1)).T

    mellinSb = np.zeros((2 * N, nx), dtype=complex)
    for i in range(nx):
        mellinSb[:, i] = np.fft.fftshift(np.fft.ifft(Sb[:, i] * coef))

    k = np.arange(1, 2 * N + 1)
    scale = np.logspace(np.log10(fmin / fmax), np.log10(fmax / fmin), N)
    theta = np.log(scale)
    mellinSSb = mellinS * np.conj(mellinSb)

    waf = np.fft.ifft(mellinSSb, N, axis=0)
    no2 = int((N + N % 2) / 2.0)
    waf = np.r_[waf[no2:(N + 1), :], waf[:no2, :]]

    # normalization
    s = np.real(s_ana)
    SP = np.fft.fft(signal.hilbert(s))
    indmin = int(1 + np.round(fmin * (nx - 2)))
    indmax = int(1 + np.round(fmax * (nx - 2)))
    sp_ana = SP[(indmin - 1):indmax]
    waf *= (np.linalg.norm(sp_ana) ** 2) / waf[no2 - 1, m - 1] / N

    return waf, tau, theta


if __name__ == '__main__':
    # FIXME: Ambiguity plot is left-right reversed at the moment

    # from mpl_toolkits.mplot3d import Axes3D

    data_dir = '../data'
    needle_filename = 'chirp_4_raw.c64'
    haystack_filename = 'chirp_4_T+70samp_F+82.89Hz.c64'
    print(haystack_filename)
    needle_samples = np.fromfile(os.path.join(data_dir, needle_filename), dtype=np.complex64)
    haystack_samples = np.fromfile(os.path.join(data_dir, haystack_filename), dtype=np.complex64)[0:4096]
    len_needle = len(needle_samples)

    samp_rate = 48e3
    freq_offsets = np.arange(-100, 100, 0.5)

    # benchmarks
    rounds = 3
    print('running {} rounds per function'.format(rounds))
    for func in [amb_surf, amb_surf_numba, amb_surf_multiprocessing, amb_surf_multiprocessing_numba]:
        start = time.time()
        for _ in range(rounds):
            surf = func(needle_samples, haystack_samples, freq_offsets, samp_rate)
        elap = (time.time()-start) / rounds
        fmax, tmax = np.unravel_index(surf.argmax(), surf.shape)
        tau_max = len(needle_samples)//2 - tmax
        freq_max = freq_offsets[fmax]
        print(func.__name__, surf.shape, surf.dtype, '->', tau_max, freq_max)
        print(func.__name__, 'elap {:.9f} s'.format(elap))

    # plotting
    extents = [
        -len_needle//2, len_needle//2,
        100, -100]
    plt.figure(dpi=150)
    plt.imshow(surf, aspect='auto', interpolation='nearest', extent=extents)
    plt.ylabel('Frequency offset [Hz]')
    plt.xlabel('Time offset [samples]')
    plt.gca().invert_yaxis()
    plt.plot(tau_max, freq_max, 'x', color='red', alpha=0.75)
    plt.show()

    print('Time lag: {:d} samples'.format(tau_max))
    print('Frequency offset: {:.2f} Hz'.format(freq_max))

    # fig = plt.figure(dpi=150)
    # ax = fig.add_subplot(111, projection='3d')
    # x = tau
    # y = freq_offsets
    # X, Y = np.meshgrid(x, y)
    # Z = surf.reshape(X.shape)
    #
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    #
    # ax.set_xlabel('Frequency offset [Hz]')
    # ax.set_ylabel('Lag [samples]')
    # ax.set_zlabel('Correlation')
    # plt.show()