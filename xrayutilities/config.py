# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2010-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to parse xrayutilities user-specific config file
the parsed values are provide as global constants for the use
in other parts of xrayutilities. The config file with the default constants
is found in the python installation path of xrayutilities. It is however not
recommended to change things there, instead the user-specific config file
~/.xrayutilities.conf or the local xrayutilities.conf file should be used.
"""

from ast import literal_eval
import os.path
import numpy
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from . import __path__
from . import utilities_noconf

# so far parsed config variables are
#
# wavelength
# energy
# verbosity
# nthreads
# dynlow
# dynhigh
# epsilon
# database filename for atomic structure factors
# kappa_plane and kappa_angle

xuParser = configparser.ConfigParser()
xuParser.optionxform = str

# read global default values for configuration variables
with open(os.path.join(__path__[0], "xrayutilities_default.conf")) as conffile:
    xuParser.readfp(conffile)

# read user configuration and local configuration if available
cfiles = xuParser.read([
    os.path.expanduser(os.path.join("~", ".xrayutilities.conf")),
    "xrayutilities.conf"])

# set global variables according to configuration
sxu = xuParser["xrayutilities"]
INFO_LOW = sxu.getint("info_low")
INFO_ALL = sxu.getint("info_all")
DEBUG = sxu.getint("debug")

VERBOSITY = sxu.getint("verbosity")
try:
    WAVELENGTH = sxu.getfloat("wavelength")
except:
    WAVELENGTH = numpy.nan
if numpy.isnan(WAVELENGTH):
    WAVELENGTH = sxu.get("wavelength")

try:
    ENERGY = sxu.getfloat("energy")
except:
    ENERGY = numpy.nan
if numpy.isnan(ENERGY):
    ENERGY = sxu.get("energy")
if ENERGY == 'NaN':
    ENERGY = utilities_noconf.lam2en(utilities_noconf.wavelength(WAVELENGTH))
else:  # energy was given and wavelength is calculated from given energy
    WAVELENGTH = utilities_noconf.en2lam(utilities_noconf.energy(ENERGY))

# number of threads in parallel section of c-code
NTHREADS = sxu.getint("nthreads")

# default parameters for the maplog function
DYNLOW = sxu.getfloat("dynlow")
DYNHIGH = sxu.getfloat("dynhigh")

# small number needed for error checks
EPSILON = sxu.getfloat("epsilon")

# name of the database with atomic scattering factors
DBNAME = sxu.get("dbname")

# kappa goniometer specific config parameters
KAPPA_PLANE = sxu.get("kappa_plane")
KAPPA_ANGLE = sxu.getfloat("kappa_angle")

# parser Powder profile related variables
POWDER = dict()

subsec = 'classoptions'
POWDER[subsec] = dict(xuParser["powder"])
for k in ('oversampling',):
    POWDER[subsec][k] = int(POWDER[subsec][k])
for k in ('output_gaussian_smoother_bins_sigma', 'window_width'):
    POWDER[subsec][k] = float(POWDER[subsec][k])

subsec = 'global'
POWDER[subsec] = dict(xuParser["powder.global"])
for k in ('diffractometer_radius', 'equatorial_divergence_deg'):
    POWDER[subsec][k] = float(POWDER[subsec][k])

subsec = 'emission'
POWDER[subsec] = dict(xuParser["powder.emission"])
for k in ('crystallite_size_gauss', 'crystallite_size_lor'):
    POWDER[subsec][k] = float(POWDER[subsec][k])
for k in ('emiss_wavelengths', 'emiss_intensities',
          'emiss_gauss_widths', 'emiss_lor_widths'):
    POWDER[subsec][k] = numpy.asarray(literal_eval(POWDER[subsec][k]))
POWDER[subsec]['emiss_wavelengths'] = numpy.asarray(tuple(
    utilities_noconf.wavelength(wl) * 1e-10
    for wl in POWDER[subsec]['emiss_wavelengths']))

subsec = 'axial'
POWDER[subsec] = dict(xuParser["powder.axial"])
for k in ('n_integral_points',):
    POWDER[subsec][k] = int(POWDER[subsec][k])
for k in ('slit_length_source', 'slit_length_target', 'length_sample',
          'angI_deg', 'angD_deg'):
    POWDER[subsec][k] = float(POWDER[subsec][k])

subsec = 'absorption'
POWDER[subsec] = dict(xuParser["powder.absorption"])
for k in ('absorption_coefficient', 'sample_thickness'):
    POWDER[subsec][k] = float(POWDER[subsec][k])

subsec = 'si_psd'
POWDER[subsec] = dict(xuParser["powder.psd"])
for k in ('si_psd_window_bounds',):
    POWDER[subsec][k] = literal_eval(POWDER[subsec][k])

subsec = 'receiver_slit'
POWDER[subsec] = dict(xuParser["powder.receiver_slit"])
for k in ('slit_width', ):
    POWDER[subsec][k] = float(POWDER[subsec][k])

subsec = 'tube_tails'
POWDER[subsec] = dict(xuParser["powder.tube_tails"])
for k in ('main_width', 'tail_left', 'tail_right', 'tail_intens'):
    POWDER[subsec][k] = float(POWDER[subsec][k])

if VERBOSITY >= DEBUG:
    print("XU.config: xrayutilities configuration files: %s" % repr(cfiles))
    print("xrayutilities configuration:")
    for (name, value) in xuParser.items("xrayutilities"):
        print("%s: %s" % (name, value))
    print("---")
