"""
MAIN SCRIPT
This is a script to simulate the effect of
helicopter blades on a radar output.

Created by Jeff Budge 04/29/2021
"""

import numpy as np
import pandas as pd
import pylab as plab
import matplotlib.pyplot as plt


def pow2(x):
    return 1 << x.bit_length()


def db(x):
    return 10 * np.log10(abs(x))


# Get the difference between two angles, smallest angle in the circle
def adiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


# Get the range given our geometry
def rng(x, om_t, d_r):
    return np.sqrt(d_r ** 2 + x ** 2 * np.sin(om_t) ** 2)


# Get the attenuation in x and wt
def att(x, om_t, phi_el, phi_az, beta):
    az_diff = adiff(om_t, 0)
    el_diff = adiff(x, beta)
    return abs(np.sin(np.pi / phi_az * az_diff) * np.sin(np.pi / phi_el * el_diff) * phi_az * phi_el) / \
           (np.pi ** 2 * el_diff * az_diff)


def rad_vel(x, om_t, d_r, om):
    return om * x**2 * np.sin(om_t) * np.cos(om_t) / (np.sqrt(d_r**2 + x**2 * np.sin(om_t)**2))


def integrand(x, t, om_t, d_r, phi_el, phi_az, beta, sigma, f_o):
    dopp = np.exp(1j * 2 * np.pi * (f_o - 2 * rad_vel(x, om_t, d_r, om_t / t) * f_o / c0) * t)
    loc_att = att(x, om_t, phi_el, phi_az, beta)
    return sigma * dopp * loc_att


num_blades = 4
blade_len = 1.524
omega = 2 * 2 * np.pi  # To get the speed in rad/s
fc = 10e9
fs = 500e6
prf = 2172
tp = 0.000004528
c0 = 299792458
plt.close('all')

nsam = int(fs * tp)
nsam_pow2 = pow2(nsam) * 4

t = np.arange(prf) / prf
rngs =
