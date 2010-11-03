"""
module to parse xrutils user-specific config file
the parsed values are provide as global constants for the use
in other parts of xrutils. The config file with the default constants
is 
"""

import os.path
import numpy 
import ConfigParser

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

xrutilsParser = ConfigParser.ConfigParser()

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
