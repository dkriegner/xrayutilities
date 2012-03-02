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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@aol.at>

"""
module to parse xrutils user-specific config file
the parsed values are provide as global constants for the use
in other parts of xrutils. The config file with the default constants
is found in the python installation path of xrutils. It is however not 
recommended to change things there, instead the user-specific config file
~/.xrutils.conf or the local xrutils.conf file should be used.
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
# clib_path NotImplemented yet

xrutilsParser = configparser.ConfigParser()

#read global default values for configuration variables 
with open(os.path.join(__path__[0],"xrutils_default.conf")) as gconffile:
    xrutilsParser.readfp(gconffile)

# read user configuration and local configuration if available 
cfiles = xrutilsParser.read([os.path.join(__path__[0],"clib_path.conf"), \
              os.path.expanduser(os.path.join("~",".xrutils.conf")), \
              "xrutils.conf"])

# set global variables according to configuration
INFO_LOW = xrutilsParser.getint("xrutils","info_low")
INFO_ALL = xrutilsParser.getint("xrutils","info_all")
DEBUG = xrutilsParser.getint("xrutils","debug")

VERBOSITY = xrutilsParser.getint("xrutils","verbosity")
WAVELENGTH = xrutilsParser.getfloat("xrutils","wavelength")
ENERGY = xrutilsParser.getfloat("xrutils","energy")
if numpy.isnan(ENERGY): 
    ENERGY = utilities_noconf.lam2en(WAVELENGTH)
else: # energy was given and wavelength is calculated from given energy
    WAVELENGTH = utilities_noconf.lam2en(ENERGY)

DYNLOW = xrutilsParser.getfloat("xrutils","dynlow")
DYNHIGH = xrutilsParser.getfloat("xrutils","dynhigh")



try:
    CLIB_PATH = xrutilsParser.get("xrutils","clib_path")
except NoOptionError:
    print("Config option clib_path not found indicating that you did not proper install xrutils!\n Look at the README.txt file for installation instructions")

if VERBOSITY >= DEBUG:
    print("XU.config: xrutils configuration files: %s" %repr(cfiles))
    print("xrutils configuration:")
    for (name, value) in xrutilsParser.items("xrutils"):
        print("%s: %s" %(name,value))
    print ("---")
