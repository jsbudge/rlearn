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
def genRangeProfile(f_path, gx, gy, gz, gv,
                    rbins, rpan, rtilt, pd_r, pd_i, times, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gx.size:
        # Load in all the parameters that don't change
        n_points = gx.size
        n_samples = rbins.size
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
            att = abs(math.sin(att_el_c * eldiff) / (att_el_c * eldiff)
                      * math.sin(att_az_c * azdiff) / (att_az_c * azdiff))
            acc_val = gv[samp_point] * cmath.exp(-1j * 2 * wavenumber * rng_to_target) * att
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)
