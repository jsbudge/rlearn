import cmath
import math
from numba import cuda, njit
import numpy as np

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def cpudiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(device=True)
def interp(x, y, tt, bg):
    # Simple 2d linear nearest neighbor interpolation
    x0 = int(x)
    y0 = int(y)
    x1 = int(x0 + 1 if x - x0 >= 0 else -1)
    y1 = int(y0 + 1 if y - y0 >= 0 else -1)
    xdiff = x - x0
    ydiff = y - y0
    return bg[tt, x1, y1].real * xdiff * ydiff + bg[tt, x1, y0].real * xdiff * (1 - ydiff) + bg[tt, x0, y1].real * \
         (1 - xdiff) * ydiff + bg[tt, x0, y0].real * (1 - xdiff) * (1 - ydiff)


@cuda.jit(device=True)
def applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, az, el, k, a_k, b_k):
    a = a_k / k * (2 * np.pi)
    b = b_k / k * (2 * np.pi)
    el_tx = math.asin(-s_tz / rngtx)
    az_tx = math.atan2(s_ty, s_tx)
    eldiff = diff(el_tx, el)
    azdiff = diff(az_tx, az)
    tx_pat = abs(math.sin(math.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) *
                 math.sin(np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi))) * \
             math.sqrt(math.sin(eldiff) * math.sin(eldiff) * math.cos(azdiff) * math.cos(azdiff) +
                     math.cos(eldiff) * math.cos(eldiff))
    el_rx = math.asin(-s_rz / rngrx)
    az_rx = math.atan2(s_rx, s_ry)
    eldiff = diff(el_rx, el)
    azdiff = diff(az_rx, az)
    rx_pat = abs(math.sin(math.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) *
                 math.sin(np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi))) * \
             math.sqrt(math.sin(eldiff) * math.sin(eldiff) * math.cos(azdiff) * math.cos(azdiff) +
                     math.cos(eldiff) * math.cos(eldiff))
    return tx_pat * rx_pat


# CPU version
def applyRadiationPatternCPU(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, az, el, k, a_k=3, b_k=.01):
    a = a_k / k * (2 * np.pi)
    b = b_k / k * (2 * np.pi)
    el_tx = np.arcsin(-s_tz / rngtx)
    az_tx = np.arctan2(s_ty, s_tx)
    eldiff = cpudiff(el_tx, el) if el_tx != el else 1e-9
    azdiff = cpudiff(az_tx, az) if az_tx != az else 1e-9
    tx_pat = abs(np.sin(np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) *
                 np.sin(np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi))) * \
             np.sqrt(np.sin(eldiff) * np.sin(eldiff) * np.cos(azdiff) * np.cos(azdiff) +
                       np.cos(eldiff) * np.cos(eldiff))
    el_rx = np.arcsin(-s_rz / rngrx)
    az_rx = np.arctan2(s_ry, s_rx)
    eldiff = cpudiff(el_rx, el) if el_rx != el else 1e-9
    azdiff = cpudiff(az_rx, az) if az_rx != az else 1e-9
    rx_pat = abs(np.sin(np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) *
                 np.sin(np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi))) * \
             np.sqrt(np.sin(eldiff) * np.sin(eldiff) * np.cos(azdiff) * np.cos(azdiff) +
                       np.cos(eldiff) * np.cos(eldiff))
    return tx_pat * rx_pat


@cuda.jit
def genRangeProfile(pathrx, pathtx, gp, pan, el, bg, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]

        # Calc out wave height
        x_i = tx % params[6] / params[6] * bg.shape[1]
        y_i = ty % params[7] / params[7] * bg.shape[2]
        tz = interp(x_i, y_i, tt, bg)

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = tx - pathtx[0, tt]
        s_ty = ty - pathtx[1, tt]
        s_tz = tz - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        s_rx = tx - pathrx[0, tt]
        s_ry = ty - pathrx[1, tt]
        s_rz = tz - pathrx[2, tt]
        rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
        rng = (rngtx + rngrx)
        rng_bin = (rng / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            # Cross product of dx and dy to get normal vector
            tz_v1 = interp(x_i + .01, y_i, tt, bg)
            tz_v2 = interp(x_i, y_i + .01, tt, bg)
            # l2 norm for normalization
            nnorm = math.sqrt((.01 * tz_v1) * (.01 * tz_v1) + (.01 * tz_v2) * (.01 * tz_v2) + .0001 * .0001)
            rnorm = (-.01 * tz_v1 * s_tx - .01 * tz_v2 * s_ty + .0001 * s_tz) / \
                    (rngtx * nnorm)
            # Reflection of wave
            ref_x = 2 * rnorm * -.01 * tz_v1 / nnorm - s_tx / rngtx
            ref_y = 2 * rnorm * -.01 * tz_v2 / nnorm - s_ty / rngtx
            ref_z = 2 * rnorm * .0001 / nnorm - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = abs(ref_x * s_rx / rngrx + ref_y * s_ry / rngrx + ref_z * s_rz / rngrx)
            att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, pan[tt], el[tt],
                                        wavenumber, params[9], params[10])
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit
def genSubProfile(pathrx, pathtx, subs, pan, el, bg, pd_r, pd_i, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        sub_x = subs[subnum, tt, 1]
        sub_y = subs[subnum, tt, 2]
        sub_cos = subs[subnum, tt, 3]
        sub_sin = subs[subnum, tt, 4]
        spow = subs[subnum, tt, 0] * 5

        # Calc out wave height
        x_i = sub_x % params[6] / params[6] * bg.shape[1]
        y_i = sub_y % params[7] / params[7] * bg.shape[2]
        tz = interp(x_i, y_i, tt, bg)
        sub_z = subs[subnum, tt, 0] + tz

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        xpts = 21
        ypts = 11
        sub_length = 10.0
        sub_width = 2.5
        for n in range(xpts):
            for m in range(ypts):
                shift_x = sub_x + (n - xpts // 2) / (xpts // 2) * sub_length * sub_sin
                shift_y = sub_y + sub_width / sub_length * math.sqrt(
                    sub_length * sub_length - ((n - xpts // 2) / (xpts // 2) * sub_length) *
                    ((n - xpts // 2) / (xpts // 2) * sub_length)) * sub_cos * (m - ypts // 2) / (ypts // 2)
                s_tx = shift_x - pathtx[0, tt]
                s_ty = shift_y - pathtx[1, tt]
                s_tz = sub_z - pathtx[2, tt]
                rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
                s_rx = shift_x - pathrx[0, tt]
                s_ry = shift_y - pathrx[1, tt]
                s_rz = sub_z - pathrx[2, tt]
                rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
                rng = (rngtx + rngrx)
                rng_bin = (rng / c0 - 2 * params[3]) * params[4]
                but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
                if n_samples > but > 0:
                    att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx,
                                                pan[tt], el[tt], wavenumber, params[9], params[10])
                    acc_val = spow * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
                    cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
                    cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit()
def getDetectionCheck(pathtx, subs, pd_r, pd_i, pan, el, det_spread, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        wavenumber = 2 * np.pi / params[2]

        sub_x = subs[subnum, tt, 1]
        sub_y = subs[subnum, tt, 2]
        sub_z = subs[subnum, tt, 0]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = sub_x - pathtx[0, tt]
        s_ty = sub_y - pathtx[1, tt]
        s_tz = sub_z - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_tx, s_ty, s_tz, rngtx, pan[tt], el[tt],
                                    wavenumber, params[9], params[10])
        acc_val = att * cmath.exp(1j * wavenumber * rngtx) * 1 / (rngtx * rngtx)

        # Find exact spot in detection chunk to place pulse
        ptt = int((params[6] + rngtx / c0) * params[4])
        pt_s = int(ptt // params[7])
        pt_f = int(ptt % params[7])
        cuda.atomic.add(pd_r, (pt_f, pt_s), acc_val.real)
        cuda.atomic.add(pd_i, (pt_f, pt_s), acc_val.imag)
        det_spread[pt_s] = 1