import cmath
import math
from numba import cuda, njit
from numba.cuda.random import xoroshiro128p_uniform_float64
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


'''@cuda.jit(device=True)
def applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, az_rx, el_rx, az_tx, el_tx, k, a_k, b_k):
    a = a_k / k * (2 * np.pi)
    b = b_k / k * (2 * np.pi)
    el_c = math.asin(-s_tz / rngtx)
    az_c = math.atan2(s_ty, s_tx)
    eldiff = diff(el_c, el_tx)
    azdiff = diff(az_c, az_tx)
    tx_pat = abs(math.sin(math.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) *
                 math.sin(np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi))) * \
             math.sqrt(math.sin(eldiff) * math.sin(eldiff) * math.cos(azdiff) * math.cos(azdiff) +
                       math.cos(eldiff) * math.cos(eldiff))
    el_c = math.asin(-s_rz / rngrx)
    az_c = math.atan2(s_rx, s_ry)
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    rx_pat = abs(math.sin(math.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) *
                 math.sin(np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi))) * \
             math.sqrt(math.sin(eldiff) * math.sin(eldiff) * math.cos(azdiff) * math.cos(azdiff) +
                       math.cos(eldiff) * math.cos(eldiff))
    return tx_pat * rx_pat'''


@cuda.jit(device=True)
def applyRadiationPattern(el_c, az_c, az_rx, el_rx, az_tx, el_tx, bw_az, bw_el):
    a = np.pi / bw_az 
    b = np.pi / bw_el
    eldiff = diff(el_c, el_tx)
    azdiff = diff(az_c, az_tx)
    txaz = abs(math.sin(a * azdiff) / (a * azdiff)) if azdiff != 0 else 1.
    txaz = txaz if azdiff <= bw_az * 2 else 0.
    txel = abs(math.sin(b * eldiff) / (b * eldiff)) if eldiff != 0 else 1.
    txel = txel if eldiff <= bw_el * 2 else 0.
    tx_pat = txaz * txel
    # tx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    rxaz = abs(math.sin(a * azdiff) / (a * azdiff)) if azdiff != 0 else 1.
    rxaz = rxaz if azdiff <= bw_az * 2 else 0.
    rxel = abs(math.sin(b * eldiff) / (b * eldiff)) if eldiff != 0 else 1.
    rxel = rxel if eldiff <= bw_el * 2 else 0.
    rx_pat = rxaz * rxel
    # rx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    return tx_pat * tx_pat * rx_pat * rx_pat


@cuda.jit(device=True)
def applyOneWayRadPat(el_c_tx, az_c_tx, az_tx, el_tx, k, a_k, b_k):
    a = a_k / k * (2 * np.pi)
    b = b_k / k * (2 * np.pi)
    eldiff = diff(el_c_tx, el_tx)
    azdiff = diff(az_c_tx, az_tx)
    return abs(math.sin(math.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) /
               (np.pi * a * k * math.cos(azdiff) * math.cos(eldiff) / (2 * np.pi)) *
               math.sin(np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi)) /
               (np.pi * b * k * math.cos(eldiff) * math.sin(azdiff) / (2 * np.pi))) * \
           math.sqrt(math.sin(eldiff) * math.sin(eldiff) * math.cos(azdiff) * math.cos(azdiff) +
                     math.cos(eldiff) * math.cos(eldiff))


# CPU version
def applyRadiationPatternCPU(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, az_rx, el_rx, az_tx, el_tx, k, a_k=3,
                             b_k=.01):
    a = a_k / k * (2 * np.pi)
    b = b_k / k * (2 * np.pi)
    el_c = np.arcsin(-s_tz / rngtx)
    az_c = np.arctan2(s_ty, s_tx)
    eldiff = cpudiff(el_c, el_tx) if el_tx != el_c else 1e-9
    azdiff = cpudiff(az_c, az_tx) if az_tx != az_c else 1e-9
    tx_pat = abs(np.sin(np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) *
                 np.sin(np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi))) * \
             np.sqrt(np.sin(eldiff) * np.sin(eldiff) * np.cos(azdiff) * np.cos(azdiff) +
                     np.cos(eldiff) * np.cos(eldiff))
    el_c = np.arcsin(-s_rz / rngrx)
    az_c = np.arctan2(s_ry, s_rx)
    eldiff = cpudiff(el_c, el_rx) if el_rx != el_c else 1e-9
    azdiff = cpudiff(az_c, az_rx) if az_rx != az_c else 1e-9
    rx_pat = abs(np.sin(np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) /
                 (np.pi * a * k * np.cos(azdiff) * np.cos(eldiff) / (2 * np.pi)) *
                 np.sin(np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi)) /
                 (np.pi * b * k * np.cos(eldiff) * np.sin(azdiff) / (2 * np.pi))) * \
             np.sqrt(np.sin(eldiff) * np.sin(eldiff) * np.cos(azdiff) * np.cos(azdiff) +
                     np.cos(eldiff) * np.cos(eldiff))
    return tx_pat * rx_pat


@cuda.jit
def genRangeProfile(pathrx, pathtx, gp, norms, panrx, elrx, pantx, eltx, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]
        tz = gp[samp_point, 2]

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
            rnorm = norms[samp_point, 0] * s_tx + norms[samp_point, 1] * s_ty + norms[samp_point, 2] * s_tz
            # Reflection of wave
            ref_x = 2 * rnorm * norms[samp_point, 0] - s_tx / rngtx
            ref_y = 2 * rnorm * norms[samp_point, 1] - s_ty / rngtx
            ref_z = 2 * rnorm * norms[samp_point, 2] - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = abs(ref_x * s_rx / rngrx + ref_y * s_ry / rngrx + ref_z * s_rz / rngrx) * 10.
            att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, panrx[tt], elrx[tt],
                                        pantx[tt], eltx[tt], wavenumber, params[9], params[10]) * 1. * 1 / (rng * rng)
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit
def genRangeData(pathrx, pathtx, gp, norms, panrx, elrx, pantx, eltx, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]
        tz = gp[samp_point, 2]

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
            rnorm = norms[samp_point, 0] * s_tx + norms[samp_point, 1] * s_ty + norms[samp_point, 2] * s_tz
            # Reflection of wave
            ref_x = 2 * rnorm * norms[samp_point, 0] - s_tx / rngtx
            ref_y = 2 * rnorm * norms[samp_point, 1] - s_ty / rngtx
            ref_z = 2 * rnorm * norms[samp_point, 2] - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = abs(ref_x * s_rx / rngrx + ref_y * s_ry / rngrx + ref_z * s_rz / rngrx) * 10.
            att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, panrx[tt], elrx[tt],
                                        pantx[tt], eltx[tt], wavenumber, params[9], params[10]) * 1. * 1 / (rng * rng)
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit
def getAngleBlock(sp, data, range_bin, az, el, rx, fc):
    p, _az, _el = cuda.grid(ndim=3)
    if _az < len(az) and _el < len(el) and p < len(range_bin):
        val = 0.
        f_val = 0.
        for tt in range(data.shape[1]):
            for r in range(rx.shape[0]):
                ah = rx[r, 0] * math.cos(az[_az]) * math.sin(np.pi / 2 + el[_el]) + \
                     rx[r, 1] * math.sin(az[_az]) * math.sin(np.pi / 2 + el[_el]) + \
                     rx[r, 2] * math.cos(np.pi / 2 + el[_el])
                pc = cmath.exp(-1j * 2 * np.pi * fc / c0 * ah)
                val += data[p, tt, r] * pc
            f_val = max(f_val, abs(val))
        sp[p, _az, _el] = f_val


@cuda.jit
def calcBounceFromMesh(tri_vert_indices, vert_xyz, vert_norms, vert_reflectivity, source_xyz, bounce_uv, return_xyz,
                       return_pow, bounce_xyz):
    tri, tt = cuda.grid(ndim=2)
    if tri < tri_vert_indices.shape[0] and tt < source_xyz.shape[1]:
        tv1 = tri_vert_indices[tri, 0]
        tv2 = tri_vert_indices[tri, 1]
        tv3 = tri_vert_indices[tri, 2]

        for n in range(bounce_uv.shape[0]):
            u = bounce_uv[n, 0]
            v = bounce_uv[n, 1]
            w = 1 - (u + v)
            # Get barycentric coordinates for bounce points
            bar_x = vert_xyz[tv1, 0] * u + vert_xyz[tv2, 0] * v + vert_xyz[tv3, 0] * w
            bar_y = vert_xyz[tv1, 1] * u + vert_xyz[tv2, 1] * v + vert_xyz[tv3, 1] * w
            bar_z = vert_xyz[tv1, 2] * u + vert_xyz[tv2, 2] * v + vert_xyz[tv3, 2] * w

            norm_x = vert_norms[tv1, 0] * u + vert_norms[tv2, 0] * v + vert_norms[tv3, 0] * w
            norm_y = vert_norms[tv1, 1] * u + vert_norms[tv2, 1] * v + vert_norms[tv3, 1] * w
            norm_z = vert_norms[tv1, 2] * u + vert_norms[tv2, 2] * v + vert_norms[tv3, 2] * w

            # Calculate out the angles in azimuth and elevation for the bounce
            s_tx = bar_x - source_xyz[0, tt]
            s_ty = bar_y - source_xyz[1, tt]
            s_tz = bar_z - source_xyz[2, tt]
            rng = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz)

            # Calculate out the bounce angles
            rnorm = norm_x * s_tx / rng + norm_y * s_ty / rng + norm_z * s_tz / rng
            b_y = -(2 * rnorm * norm_x - s_tx / rng)
            b_x = -(2 * rnorm * norm_y - s_ty / rng)
            b_z = -(2 * rnorm * norm_z - s_tz / rng)

            # Calc power multiplier based on range, reflectivity
            pt_idx = tri * bounce_uv.shape[0] + n - 1
            pow_mult = vert_reflectivity[tv1] * u + vert_reflectivity[tv2] * v + vert_reflectivity[tv3] * w
            return_xyz[pt_idx, tt, 0] = s_tx
            return_xyz[pt_idx, tt, 1] = s_ty
            return_xyz[pt_idx, tt, 2] = s_tz

            bounce_xyz[pt_idx, tt, 0] = b_x
            bounce_xyz[pt_idx, tt, 1] = b_y
            bounce_xyz[pt_idx, tt, 2] = b_z

            return_pow[pt_idx, tt] = pow_mult


@cuda.jit
def checkIntersection(return_sph, is_block):
    pt_idx, tt = cuda.grid(ndim=2)
    if pt_idx < return_sph.shape[0] and tt < return_sph.shape[1]:
        for idx in range(return_sph.shape[0]):
            if idx == pt_idx:
                continue
            ang_diff = math.sqrt(diff(return_sph[pt_idx, tt, 0], return_sph[idx, tt, 0]) *
                                 diff(return_sph[pt_idx, tt, 0], return_sph[idx, tt, 0]) +
                                 diff(return_sph[pt_idx, tt, 1], return_sph[idx, tt, 1]) *
                                 diff(return_sph[pt_idx, tt, 1], return_sph[idx, tt, 1]))
            if ang_diff < 0.01 and return_sph[pt_idx, tt, 2] > return_sph[idx, tt, 2]:
                is_block[pt_idx, tt] = True


@cuda.jit
def genRangeProfileFromMesh(ret_xyz, bounce_xyz, receive_xyz, return_pow, is_blocked, panrx, elrx, pantx,
                            eltx, pd_r, pd_i, debug_att, wavelength, near_range_s, source_fs, bw_az, bw_el):
    pt_idx, tt = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and pt_idx < ret_xyz.shape[0]:
        if is_blocked[pt_idx, tt]:
            return
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # Convert point to Cartesian coords and get receiver angles
        tx = ret_xyz[pt_idx, tt, 0]
        ty = ret_xyz[pt_idx, tt, 1]
        tz = ret_xyz[pt_idx, tt, 2]

        rx = tx - receive_xyz[0, tt]
        ry = ty - receive_xyz[1, tt]
        rz = tz - receive_xyz[2, tt]
        r_rng = math.sqrt(rx * rx + ry * ry + rz * rz)
        r_el = math.acos(rz / r_rng)
        r_az = math.atan2(rx, ry)
        two_way_rng = math.sqrt(tx * tx + ty * ty + tz * tz) + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            # Get bounce vector dot product
            a = bounce_xyz[pt_idx, tt, 0] * rx / r_rng + bounce_xyz[pt_idx, tt, 1] * ry / r_rng + \
                         bounce_xyz[pt_idx, tt, 2] * rz / r_rng
            # Run scattering coefficient through scaling to simulate real-world scattering
            P = return_pow[pt_idx, tt]
            sigma = .04 * P
            reflectivity = math.exp(-math.pow((a * a / (2 * sigma)), P))
            att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt], bw_az, bw_el) * \
                  1 / (two_way_rng * two_way_rng) * reflectivity
            debug_att[pt_idx, tt] = att
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * 1e1
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)
            
            
@cuda.jit
def genRangeWithoutIntersection(rng_states, tri_vert_indices, vert_xyz, vert_norms, vert_scattering, vert_reflectivity,
                                source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, calc_pts, calc_angs,
                                wavelength, near_range_s, source_fs, bw_az, bw_el, pts_per_tri, debug_flag):
    # sourcery no-metrics
    tri, tt = cuda.grid(ndim=2)
    
    if tri < tri_vert_indices.shape[0] and tt < source_xyz.shape[1]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        tv1 = tri_vert_indices[tri, 0]
        tv2 = tri_vert_indices[tri, 1]
        tv3 = tri_vert_indices[tri, 2]

        for n in range(pts_per_tri):
            u = xoroshiro128p_uniform_float64(rng_states, tri)
            v = xoroshiro128p_uniform_float64(rng_states, tri)
            if u + v > 1:
                u /= 2
                v /= 2
            w = 1 - (u + v)
            # Get barycentric coordinates for bounce points
            bar_x = vert_xyz[tv1, 0] * u + vert_xyz[tv2, 0] * v + vert_xyz[tv3, 0] * w
            bar_y = vert_xyz[tv1, 1] * u + vert_xyz[tv2, 1] * v + vert_xyz[tv3, 1] * w
            bar_z = vert_xyz[tv1, 2] * u + vert_xyz[tv2, 2] * v + vert_xyz[tv3, 2] * w

            norm_x = vert_norms[tv1, 0] * u + vert_norms[tv2, 0] * v + vert_norms[tv3, 0] * w
            norm_y = vert_norms[tv1, 1] * u + vert_norms[tv2, 1] * v + vert_norms[tv3, 1] * w
            norm_z = vert_norms[tv1, 2] * u + vert_norms[tv2, 2] * v + vert_norms[tv3, 2] * w

            # Calculate out the angles in azimuth and elevation for the bounce
            tx = bar_x - source_xyz[0, tt]
            ty = bar_y - source_xyz[1, tt]
            tz = bar_z - source_xyz[2, tt]
            rng = math.sqrt(tx * tx + ty * ty + tz * tz)
            if debug_flag and tt == 0:
                calc_pts[tri, 0] = tx
                calc_pts[tri, 1] = ty
                calc_pts[tri, 2] = tz
                calc_angs[tri, 0] = math.acos(tz / rng)
                calc_angs[tri, 1] = math.atan2(tx, ty)

            # Calculate out the bounce angles
            rnorm = norm_x * tx / rng + norm_y * ty / rng + norm_z * tz / rng
            b_y = -(2 * rnorm * norm_x - tx / rng)
            b_x = -(2 * rnorm * norm_y - ty / rng)
            b_z = -(2 * rnorm * norm_z - tz / rng)

            # Calc power multiplier based on range, reflectivity
            scat_ref = vert_reflectivity[tv1] * u + vert_reflectivity[tv2] * v + vert_reflectivity[tv3] * w
            scat_pow = vert_scattering[tv1] * u + vert_scattering[tv2] * v + vert_scattering[tv3] * w

            rx = tx - receive_xyz[0, tt]
            ry = ty - receive_xyz[1, tt]
            rz = tz - receive_xyz[2, tt]
            r_rng = math.sqrt(rx * rx + ry * ry + rz * rz)
            r_el = math.acos(rz / r_rng)
            r_az = math.atan2(rx, ry)
            two_way_rng = math.sqrt(tx * tx + ty * ty + tz * tz) + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if n_samples > but > 0:
                a = b_x * rx / r_rng + b_y * ry / r_rng + \
                    b_z * rz / r_rng
                sigma = .04 * scat_ref
                reflectivity = scat_ref * a # math.exp(-math.pow((a * a / (2 * sigma)), scat_ref))
                att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt], bw_az, bw_el) * \
                      1 / (two_way_rng * two_way_rng) * reflectivity * 1e2
                acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * scat_pow
                cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
                cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@njit
def apply_shift(ray: np.ndarray, freq_shift: np.float64, samp_rate: np.float64) -> np.ndarray:
    # apply frequency shift
    precache = 2j * np.pi * freq_shift / samp_rate
    new_ray = np.empty_like(ray)
    for idx, val in enumerate(ray):
        new_ray[idx] = val * np.exp(precache * idx)
    return new_ray


def ambiguity(s1, s2, prf, dopp_bins, mag=True, normalize=True):
    fdopp = np.linspace(-prf / 2, prf / 2, dopp_bins)
    fft_sz = findPowerOf2(len(s1)) * 2
    s1f = np.fft.fft(s1, fft_sz).conj().T
    shift_grid = np.zeros((len(s2), dopp_bins), dtype=np.complex64)
    for n in range(dopp_bins):
        shift_grid[:, n] = apply_shift(s2, fdopp[n], fs)
    s2f = np.fft.fft(shift_grid, n=fft_sz, axis=0)
    A = np.fft.fftshift(np.fft.ifft(s2f * s1f[:, None], axis=0, n=fft_sz * 2),
                        axes=0)[fft_sz - dopp_bins // 2: fft_sz + dopp_bins // 2]
    if normalize:
        A = A / abs(A).max()
    if mag:
        return abs(A) ** 2, fdopp, np.linspace(-len(s1) / 2 / fs, len(s1) / 2 / fs, len(s1))
    else:
        return A, fdopp, np.linspace(-dopp_bins / 2 * fs / c0, dopp_bins / 2 * fs / c0, dopp_bins)


def getMaxThreads():
    gpuDevice = cuda.get_current_device()
    maxThreads = gpuDevice.MAX_THREADS_PER_BLOCK // 3
    sqrtMaxThreads = int(np.sqrt(maxThreads))
    return sqrtMaxThreads, sqrtMaxThreads


