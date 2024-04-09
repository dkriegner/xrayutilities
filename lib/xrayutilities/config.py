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
# Copyright (C) 2010-2023 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to parse xrayutilities user-specific config file
the parsed values are provide as global constants for the use
in other parts of xrayutilities. The config file with the default constants
is found in the python installation path of xrayutilities. It is however not
recommended to change things there, instead the user-specific config file
~/.xrayutilities.conf or the local xrayutilities.conf file should be used.
"""

import configparser
import math
import os.path
from ast import literal_eval

from . import __path__, utilities_noconf

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


def trytomake(obj, key, typefunc):
    try:
        obj[key] = typefunc(obj[key])
    except KeyError:
        pass


# read global default values for configuration variables
with open(os.path.join(__path__[0], "xrayutilities_default.conf")) as conffile:
    xuParser.read_file(conffile)

# read user configuration and local configuration if available
cfiles = xuParser.read([
    os.path.expanduser(os.path.join("~", ".xrayutilities.conf")),
    "xrayutilities.conf"])

# set global variables according to configuration
sect = "xrayutilities"
INFO_LOW = xuParser.getint(sect, "info_low")
INFO_ALL = xuParser.getint(sect, "info_all")
DEBUG = xuParser.getint(sect, "debug")

VERBOSITY = xuParser.getint(sect, "verbosity")
try:
    WAVELENGTH = xuParser.getfloat(sect, "wavelength")
except ValueError:
    WAVELENGTH = xuParser.get(sect, "wavelength")
except configparser.NoOptionError:
    pass

try:
    ENERGY = xuParser.getfloat(sect, "energy")
except ValueError:
    ENERGY = xuParser.get(sect, "energy")
except configparser.NoOptionError:
    ENERGY = utilities_noconf.lam2en(utilities_noconf.wavelength(WAVELENGTH))
WAVELENGTH = utilities_noconf.en2lam(utilities_noconf.energy(ENERGY))

# number of threads in parallel section of c-code
NTHREADS = xuParser.getint(sect, "nthreads")

# default parameters for the maplog function
DYNLOW = xuParser.getfloat(sect, "dynlow")
DYNHIGH = xuParser.getfloat(sect, "dynhigh")

# small number needed for error checks
EPSILON = xuParser.getfloat(sect, "epsilon")
DIGITS = int(abs(math.log10(EPSILON)))

# name of the database with atomic scattering factors
DBNAME = xuParser.get(sect, "dbname")

# kappa goniometer specific config parameters
KAPPA_PLANE = xuParser.get(sect, "kappa_plane")
KAPPA_ANGLE = xuParser.getfloat(sect, "kappa_angle")

# parser Powder profile related variables
POWDER = dict()

subsec = 'classoptions'
POWDER[subsec] = dict(xuParser.items("powder"))
trytomake(POWDER[subsec], 'oversampling', int)
for k in ('gaussian_smoother_bins_sigma', 'window_width'):
    trytomake(POWDER[subsec], k, float)

subsec = 'global'
POWDER[subsec] = dict(xuParser.items("powder.global"))
for k in ('diffractometer_radius', 'equatorial_divergence_deg'):
    trytomake(POWDER[subsec], k, float)
trytomake(POWDER[subsec], 'geometry_incidence_angle', literal_eval)

subsec = 'emission'
POWDER[subsec] = dict(xuParser.items("powder.emission"))
for k in ('crystallite_size_gauss', 'crystallite_size_lor',
          'strain_lor', 'strain_gauss'):
    trytomake(POWDER[subsec], k, float)
for k in ('emiss_wavelengths', 'emiss_intensities',
          'emiss_gauss_widths', 'emiss_lor_widths'):
    trytomake(POWDER[subsec], k, literal_eval)
if 'emiss_wavelengths' in POWDER[subsec]:
    POWDER[subsec]['emiss_wavelengths'] = tuple(
        utilities_noconf.wavelength(wl) * 1e-10
        for wl in POWDER[subsec]['emiss_wavelengths'])

subsec = 'axial'
POWDER[subsec] = dict(xuParser.items("powder.axial"))
trytomake(POWDER[subsec], 'n_integral_points', int)
for k in ('slit_length_source', 'slit_length_target', 'length_sample',
          'angI_deg', 'angD_deg'):
    trytomake(POWDER[subsec], k, float)

subsec = 'absorption'
POWDER[subsec] = dict(xuParser.items("powder.absorption"))
for k in ('absorption_coefficient', 'sample_thickness'):
    trytomake(POWDER[subsec], k, float)

subsec = 'si_psd'
POWDER[subsec] = dict(xuParser.items("powder.si_psd"))
trytomake(POWDER[subsec], 'si_psd_window_bounds', literal_eval)

subsec = 'receiver_slit'
POWDER[subsec] = dict(xuParser.items("powder.receiver_slit"))
trytomake(POWDER[subsec], 'slit_width', float)

subsec = 'tube_tails'
POWDER[subsec] = dict(xuParser.items("powder.tube_tails"))
for k in ('main_width', 'tail_left', 'tail_right', 'tail_intens'):
    trytomake(POWDER[subsec], k, float)

if VERBOSITY >= DEBUG:
    print(f"XU.config: xrayutilities configuration files: {repr(cfiles)}")
    print("xrayutilities configuration:")
    for (name, value) in xuParser.items("xrayutilities"):
        print(f"{name}: {value}")
    print("---")
