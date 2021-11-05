import cmath
import math
import numpy as np

from numba import cuda, vectorize
# from numba.cuda import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

c0 = 299792458.0


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(device=True)
def raisedCosine(x, bw, a0):
    xf = x / bw + .5
    return a0 - (1 - a0) * math.cos(2 * np.pi * xf)


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], '
          'float64[:], float64[:], complex128[:], complex128[:], float64[:], float64[:], float64[:])')
def genRangeProfile(tx_path, rx_path, gx, gy, gz, gv, rx_pan, tx_pan, rx_tilt, tx_tilt, pd_r, pd_i, shift, p_rx, p_tx):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gx.size:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / p_tx[2]

        for ex_pt in range(shift.shape[0]):
            tmp_x = gx[samp_point] + shift[ex_pt, 0]
            tmp_y = gy[samp_point] + shift[ex_pt, 1]
            # Get LOS vector in XYZ and spherical coordinates at pulse time
            # Tx first
            tx_x = tmp_x - tx_path[0, tt]
            tx_y = tmp_y - tx_path[1, tt]
            tx_z = gz[samp_point] - tx_path[2, tt]
            tx_rng = math.sqrt(tx_x * tx_x + tx_y * tx_y + tx_z * tx_z)
            # Rx
            rx_x = tmp_x - rx_path[0, tt]
            rx_y = tmp_y - rx_path[1, tt]
            rx_z = gz[samp_point] - rx_path[2, tt]
            rx_rng = math.sqrt(rx_x * rx_x + rx_y * rx_y + rx_z * rx_z)
            rng_bin = ((tx_rng + rx_rng) / c0 - 2 * p_rx[6]) * p_rx[7]
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if n_samples > but > 0:
                el_tx = math.asin(-tx_z / tx_rng)
                az_tx = math.atan2(tx_x, tx_y)
                tx_eldiff = diff(el_tx, tx_tilt[tt])
                tx_azdiff = diff(az_tx, tx_pan[tt])
                tx_elpat = abs(math.sin(p_tx[0] * tx_eldiff) / (p_tx[0] * tx_eldiff)) if tx_eldiff != 0 else 1
                tx_azpat = abs(math.sin(p_tx[1] * tx_azdiff) / (p_tx[1] * tx_azdiff)) if tx_azdiff != 0 else 1
                att_tx = tx_elpat * tx_azpat
                el_rx = math.asin(-rx_z / rx_rng)
                az_rx = math.atan2(rx_x, rx_y)
                rx_eldiff = diff(el_rx, rx_tilt[tt])
                rx_azdiff = diff(az_rx, rx_pan[tt])
                rx_elpat = abs(math.sin(p_rx[0] * rx_eldiff) / (p_rx[0] * rx_eldiff)) if rx_eldiff != 0 else 1
                rx_azpat = abs(math.sin(p_rx[1] * rx_azdiff) / (p_rx[1] * rx_azdiff)) if rx_azdiff != 0 else 1
                att_rx = rx_elpat * rx_azpat
                acc_val = gv[samp_point] * att_tx * att_rx * cmath.exp(-1j * wavenumber * (tx_rng + rx_rng)) * \
                          1.0 / (tx_rng + rx_rng)
                cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
                cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)

                # Add as a small gaussian
                cuda.atomic.add(pd_r, (but - 1, np.uint64(tt)), acc_val.real * .6)
                cuda.atomic.add(pd_i, (but - 1, np.uint64(tt)), acc_val.imag * .6)
                cuda.atomic.add(pd_r, (but + 1, np.uint64(tt)), acc_val.real * .6)
                cuda.atomic.add(pd_i, (but + 1, np.uint64(tt)), acc_val.imag * .6)
                cuda.atomic.add(pd_r, (but - 2, np.uint64(tt)), acc_val.real * .2)
                cuda.atomic.add(pd_i, (but - 2, np.uint64(tt)), acc_val.imag * .2)
                cuda.atomic.add(pd_r, (but + 2, np.uint64(tt)), acc_val.real * .2)
                cuda.atomic.add(pd_i, (but + 2, np.uint64(tt)), acc_val.imag * .2)


@cuda.jit
def backproject(tx_path, rx_path, gx, gy, gz,
                rbins, rpan, rtilt, pulse_data, final_grid,
                params):
    px, py = cuda.grid(ndim=2)
    if px < gx.shape[0] and py < gx.shape[1]:
        # Load in all the parameters that don't change
        acc_val = 0
        nPulses = pulse_data.shape[1]
        n_samples = rbins.size
        att_el_c = params[0]
        att_az_c = params[1]
        k = 2 * np.pi / params[2]

        # Grab pulse data and sum up for this pixel
        for p in range(nPulses):
            cp = pulse_data[:, p]
            # Get LOS vector in XYZ and spherical coordinates at pulse time
            # Tx first
            tx_x = gx[px, py] - tx_path[0, p]
            tx_y = gy[px, py] - tx_path[1, p]
            tx_z = gz[px, py] - tx_path[2, p]
            tx_rng = math.sqrt(tx_x * tx_x + tx_y * tx_y + tx_z * tx_z)
            el_tx = math.asin(-tx_z / tx_rng)
            az_tx = math.atan2(tx_x, tx_y)
            tx_eldiff = diff(el_tx, rtilt[p])
            tx_azdiff = diff(az_tx, rpan[p])
            tx_elpat = abs(math.sin(att_el_c * tx_eldiff) / (att_el_c * tx_eldiff)) if tx_eldiff != 0 else 1
            tx_azpat = abs(math.sin(att_az_c * tx_azdiff) / (att_az_c * tx_azdiff)) if tx_azdiff != 0 else 1
            att_tx = tx_elpat * tx_azpat
            # Rx
            rx_x = gx[px, py] - rx_path[0, p]
            rx_y = gy[px, py] - rx_path[1, p]
            rx_z = gz[px, py] - rx_path[2, p]
            rx_rng = math.sqrt(rx_x * rx_x + rx_y * rx_y + rx_z * rx_z)

            rng_half = (tx_rng + rx_rng) / 2
            el_rx = math.asin(-rx_z / rx_rng)
            az_rx = math.atan2(rx_x, rx_y)
            rx_eldiff = diff(el_rx, rtilt[p])
            rx_azdiff = diff(az_rx, rpan[p])
            rx_elpat = abs(math.sin(att_el_c * rx_eldiff) / (att_el_c * rx_eldiff)) if rx_eldiff != 0 else 1
            rx_azpat = abs(math.sin(att_az_c * rx_azdiff) / (att_az_c * rx_azdiff)) if rx_azdiff != 0 else 1
            att_rx = rx_elpat * rx_azpat

            # Check to see if it's outside of our beam
            if (abs(rx_azdiff) > params[5]) or (abs(rx_eldiff) > params[4]):
                continue

            # Attenuation of beam in elevation and azimuth
            att = att_rx * att_tx

            # Azimuth window to reduce sidelobes
            # Gaussian window
            # az_win = math.exp(-azdiff*azdiff/(2*.001))
            # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
            az_win = raisedCosine(tx_azdiff, params[5], .5)
            # Flat top window
            # az_win = flattop3(azdiff, params[5])
            # Vorbis window
            # az_win = vorbis(azdiff, params[5])
            # Parzen window (not unit gain)
            # az_win = parzen(azdiff, params[5], 1, 2)
            # Avci-Nacaroglu Exponential window (not unit gain)
            # az_win = avcinac(azdiff, params[5], 3)
            # Knab (not unit gain)
            # az_win = knab(azdiff, params[5], 3)
            # az_win = 1

            # Get index into range compressed data
            rng_bin = ((tx_rng + rx_rng) / c0 - 2 * params[6]) * params[7]
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > n_samples:
                continue

            if rbins[but - 1] < rng_half < rbins[but]:
                bi0 = but - 1
                bi1 = but
            else:
                bi0 = but
                bi1 = but + 1
            if params[8] == 1:
                # Linear interpolation between bins (slower but more accurate)
                a = (cp[bi0] * (rbins[bi1] - rng_half) + cp[bi1] * (rng_half - rbins[bi0])) \
                    / (rbins[bi1] - rbins[bi0])
            elif params[8] == 0:
                # This is how APS does it (for reference, I guess)
                a = cp[bi0] if rng_half - rbins[bi0] < rbins[bi1] - rng_half else cp[bi1]
            else:
                # This is a lagrange polynomial interpolation of the specified order
                a = 0
                k = params[8] + 1
                ks = bi0 - (k - (k % 2)) / 2
                while ks < 0:
                    ks += 1
                ke = ks + k
                while ke > n_samples:
                    ke -= 1
                    ks -= 1
                for idx in range(ks, ke):
                    mm = 1
                    for jdx in range(ke - ks):
                        if jdx != idx - ks:
                            mm *= (rng_half - rbins[jdx]) / (rbins[idx] - rbins[jdx])
                    a += mm * cp[idx]

                    # Multiply by phase reference function, attenuation and azimuth window
            exp_phase = k * (tx_rng + rx_rng)
            acc_val += a * cmath.exp(1j * exp_phase) * att * az_win
        final_grid[px, py] = acc_val


@cuda.jit
def interpolate(params, pulses, interp_pulses):
    pulse, i_bin = cuda.grid(2)
    if pulse < interp_pulses.shape[1] and i_bin < params[0]:
        bi = i_bin * params[3]
        bi0 = max(int(math.floor(i_bin * params[3])), 0)
        bi1 = min(int(math.ceil(i_bin * params[3])), int(params[0]))
        pi0 = pulses[bi0, pulse]
        pi1 = pulses[bi1, pulse]
        interp_pulses[i_bin, pulse] = pi0 * (bi1 - bi) + pi1 * (bi - bi0)


@cuda.jit
def genDoppProfile(f_path, rpan, gx, gy, times, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.size and samp_point < gx.size:
        # Get LOS vector in XYZ and spherical coordinates at pulse time
        sh_x = f_path[0, tt] - gx[samp_point]
        sh_y = f_path[1, tt] - gy[samp_point]
        az_to_target = math.atan2(sh_x, sh_y)
        azdiff = diff(az_to_target, rpan[tt])
        fd = params[3] * math.cos(azdiff) / (params[2] * np.pi)
        val = cmath.exp(-1j * fd * times[tt])
        cuda.atomic.add(pd_r, np.uint64(tt), val.real)
        cuda.atomic.add(pd_i, np.uint64(tt), val.imag)
