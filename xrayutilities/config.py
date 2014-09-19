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
# Copyright (C) 2010,2012 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to parse xrayutilities user-specific config file
the parsed values are provide as global constants for the use
in other parts of xrayutilities. The config file with the default constants
is found in the python installation path of xrayutilities. It is however not
recommended to change things there, instead the user-specific config file
~/.xrayutilities.conf or the local xrayutilities.conf file should be used.
"""

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

#read global default values for configuration variables
with open(os.path.join(__path__[0],"xrayutilities_default.conf")) as gconffile:
    xuParser.readfp(gconffile)

# read user configuration and local configuration if available
cfiles = xuParser.read([os.path.expanduser(os.path.join("~",".xrayutilities.conf")), \
              "xrayutilities.conf"])

# set global variables according to configuration
INFO_LOW = xuParser.getint("xrayutilities","info_low")
INFO_ALL = xuParser.getint("xrayutilities","info_all")
DEBUG = xuParser.getint("xrayutilities","debug")

VERBOSITY = xuParser.getint("xrayutilities","verbosity")
try: WAVELENGTH = xuParser.getfloat("xrayutilities","wavelength")
except: WAVELENGTH = numpy.nan
if numpy.isnan(WAVELENGTH):
    WAVELENGTH = xuParser.get("xrayutilities","wavelength")

try: ENERGY = xuParser.getfloat("xrayutilities","energy")
except: ENERGY=numpy.nan
if numpy.isnan(ENERGY):
    ENERGY = xuParser.get("xrayutilities","energy")
if ENERGY=='NaN':
    ENERGY = utilities_noconf.lam2en(utilities_noconf.wavelength(WAVELENGTH))
else: # energy was given and wavelength is calculated from given energy
    WAVELENGTH = utilities_noconf.en2lam(utilities_noconf.energy(ENERGY))

# number of threads in parallel section of c-code
NTHREADS = xuParser.getint("xrayutilities","nthreads")

# default parameters for the maplog function
DYNLOW = xuParser.getfloat("xrayutilities","dynlow")
DYNHIGH = xuParser.getfloat("xrayutilities","dynhigh")

# small number needed for error checks
EPSILON = xuParser.getfloat("xrayutilities","epsilon")

# name of the database with atomic scattering factors
DBNAME = xuParser.get("xrayutilities","dbname")

# kappa goniometer specific config parameters
KAPPA_PLANE = xuParser.get("xrayutilities","kappa_plane")
KAPPA_ANGLE = xuParser.getfloat("xrayutilities","kappa_angle")

if VERBOSITY >= DEBUG:
    print("XU.config: xrayutilities configuration files: %s" %repr(cfiles))
    print("xrayutilities configuration:")
    for (name, value) in xuParser.items("xrayutilities"):
        print("%s: %s" %(name,value))
    print ("---")
