import cmath
import math
import numpy as np

from numba import cuda, vectorize

c0 = 299792458.0


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(device=True)
def raisedCosine(x, bw, a0):
    xf = x / bw + .5
    return a0 - (1 - a0) * math.cos(2 * np.pi * xf)


@cuda.jit
def genRangeProfile(f_path, gx, gy, gz, gv, rpan, rtilt, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gx.size:
        # Load in all the parameters that don't change
        n_points = gx.size
        n_samples = pd_r.shape[0]
        att_el_c = params[0]
        att_az_c = params[1]
        wavenumber = 2 * np.pi / params[2]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        sh_x = gx[samp_point] - f_path[0, tt]
        sh_y = gy[samp_point] - f_path[1, tt]
        sh_z = gz[samp_point] - f_path[2, tt]
        rng_to_target = math.sqrt(sh_x * sh_x + sh_y * sh_y + sh_z * sh_z)
        rng_bin = (rng_to_target / c0 - params[6]) * params[7] * 2
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            el_to_target = math.asin(-sh_z / rng_to_target)
            az_to_target = math.atan2(sh_x, sh_y)
            eldiff = diff(el_to_target, rtilt[tt])
            azdiff = diff(az_to_target, rpan[tt])
            # Check to see if it's outside of our beam
            # if abs(azdiff) <= params[5] and abs(eldiff) <= params[4]:
            att = abs(math.sin(att_el_c * eldiff) / (att_el_c * eldiff)
                      * math.sin(att_az_c * azdiff) / (att_az_c * azdiff))
            fd = params[3] * math.sin(azdiff) / (params[2] * np.pi)
            acc_val = gv[samp_point] * cmath.exp(-1j * 2 * wavenumber * rng_to_target) * att
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit
def backproject(f_path, gx, gy, gz,
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
        exp_c = params[2]

        # Grab pulse data and sum up for this pixel
        for p in range(nPulses):
            cp = pulse_data[:, p]
            # Get LOS vector in XYZ and spherical coordinates at pulse time
            sh_x = gx[px, py] - f_path[0, p]
            sh_y = gy[px, py] - f_path[1, p]
            sh_z = gz[px, py] - f_path[2, p]
            rng_to_target = math.sqrt(sh_x * sh_x + sh_y * sh_y + sh_z * sh_z)
            el_to_target = math.asin(-sh_z / rng_to_target)
            az_to_target = math.atan2(sh_x, sh_y)

            r_heading = rpan[p]
            dep_ang = rtilt[p]
            eldiff = diff(el_to_target, dep_ang)
            azdiff = diff(az_to_target, r_heading)

            # Check to see if it's outside of our beam
            if (abs(azdiff) > params[5]) or (abs(eldiff) > params[4]):
                continue

            # Attenuation of beam in elevation and azimuth
            att = abs(math.sin(att_el_c * eldiff) / (att_el_c * eldiff)
                      * math.sin(att_az_c * azdiff) / (att_az_c * azdiff))
            # att = 1

            # Azimuth window to reduce sidelobes
            # Gaussian window
            # az_win = math.exp(-azdiff*azdiff/(2*.001))
            # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
            az_win = raisedCosine(azdiff, params[5], .5)
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
            rng_bin = (rng_to_target / c0 - params[6]) * params[7] * 2
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > n_samples:
                continue

            if rbins[but - 1] < rng_to_target < rbins[but]:
                bi0 = but - 1
                bi1 = but
            else:
                bi0 = but
                bi1 = but + 1
            if params[3] == 1:
                # Linear interpolation between bins (slower but more accurate)
                a = (cp[bi0] * (rbins[bi1] - rng_to_target) + cp[bi1] * (rng_to_target - rbins[bi0])) \
                    / (rbins[bi1] - rbins[bi0])
            elif params[3] == 0:
                # This is how APS does it (for reference, I guess)
                a = cp[bi0] if rng_to_target - rbins[bi0] < rbins[bi1] - rng_to_target else cp[bi1]
            else:
                # This is a lagrange polynomial interpolation of the specified order
                a = 0
                k = params[3] + 1
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
                            mm *= (rng_to_target - rbins[jdx]) / (rbins[idx] - rbins[jdx])
                    a += mm * cp[idx]

                    # Multiply by phase reference function, attenuation and azimuth window
            exp_phase = exp_c * rng_to_target
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
def genDoppProfile(f_path, gx, gy, gz, rpan, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.size and samp_point < gx.size:
        att_az_c = params[1]
        wavenumber = 2 * np.pi / params[2]
        # Get LOS vector in XYZ and spherical coordinates at pulse time
        sh_x = gx[samp_point] - f_path[0, tt]
        sh_y = gy[samp_point] - f_path[1, tt]
        sh_z = gz[samp_point] - f_path[2, tt]
        rng_to_target = math.sqrt(sh_x * sh_x + sh_y * sh_y + sh_z * sh_z)
        az_to_target = math.atan2(sh_x, sh_y)
        azdiff = diff(az_to_target, rpan[tt])
        fd = cmath.exp(-1j * params[3] * math.sin(azdiff) / (params[2] * np.pi))
        att = abs(math.sin(att_az_c * azdiff) / (att_az_c * azdiff))
        acc_val = fd * att
        cuda.atomic.add(pd_r, np.uint64(tt), acc_val.real)
        cuda.atomic.add(pd_i, np.uint64(tt), acc_val.imag)
