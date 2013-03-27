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
# dynlow
# dynhigh
# epsilon
# clib_path
# database filename for atomic structure factors
# kappa_plane and kappa_angle

xrutilsParser = configparser.ConfigParser()

#read global default values for configuration variables
with open(os.path.join(__path__[0],"xrayutilities_default.conf")) as gconffile:
    xrutilsParser.readfp(gconffile)

# read user configuration and local configuration if available
cfiles = xrutilsParser.read([os.path.join(__path__[0],"clib_path.conf"), \
              os.path.expanduser(os.path.join("~",".xrayutilities.conf")), \
              "xrayutilities.conf"])

# set global variables according to configuration
INFO_LOW = xrutilsParser.getint("xrayutilities","info_low")
INFO_ALL = xrutilsParser.getint("xrayutilities","info_all")
DEBUG = xrutilsParser.getint("xrayutilities","debug")

VERBOSITY = xrutilsParser.getint("xrayutilities","verbosity")
try: WAVELENGTH = xrutilsParser.getfloat("xrayutilities","wavelength")
except: WAVELENGTH = numpy.nan
if numpy.isnan(WAVELENGTH):
    WAVELENGTH = xrutilsParser.get("xrayutilities","wavelength")

try: ENERGY = xrutilsParser.getfloat("xrayutilities","energy")
except: ENERGY=numpy.nan
if numpy.isnan(ENERGY):
    ENERGY = xrutilsParser.get("xrayutilities","energy")
if ENERGY=='NaN':
    ENERGY = utilities_noconf.lam2en(utilities_noconf.wavelength(WAVELENGTH))
else: # energy was given and wavelength is calculated from given energy
    WAVELENGTH = utilities_noconf.lam2en(utilities_noconf.energy(ENERGY))

DYNLOW = xrutilsParser.getfloat("xrayutilities","dynlow")
DYNHIGH = xrutilsParser.getfloat("xrayutilities","dynhigh")

# small number needed for error checks
EPSILON = xrutilsParser.getfloat("xrayutilities","epsilon")

# name of the database with atomic scattering factors
DBNAME = xrutilsParser.get("xrayutilities","dbname")

# kappa goniometer specific config parameters
KAPPA_PLANE = xrutilsParser.get("xrayutilities","kappa_plane")
KAPPA_ANGLE = xrutilsParser.getfloat("xrayutilities","kappa_angle")

try:
    CLIB_PATH = xrutilsParser.get("xrayutilities","clib_path")
except NoOptionError:
    print("Config option clib_path not found indicating that you did not proper install xrayutilities!\n Look at the README.txt file for installation instructions")

if VERBOSITY >= DEBUG:
    print("XU.config: xrayutilities configuration files: %s" %repr(cfiles))
    print("xrayutilities configuration:")
    for (name, value) in xrutilsParser.items("xrayutilities"):
        print("%s: %s" %(name,value))
    print ("---")
