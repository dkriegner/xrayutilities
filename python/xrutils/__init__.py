"""
xrutils is a package for assisting with x-ray diffraction experiments

It helps with planning experiments as well as analyzing the data.
"""

from . import math
from . import io
from . import materials

#the vis module is meanwhile deactivated - needs to 
#be rewritten.
try:
    from . import vis
except:
    print "Visualization module cannot be imported!"
    print "an will therefore not be available!"

from .experiment import Experiment
from .experiment import HXRD
from .experiment import GID
from .experiment import GID_ID10B
from .experiment import GISAXS
from .experiment import Powder

from .normalize import blockAverage1D
from .normalize import blockAverage2D
from .normalize import blockAveragePSD
from .normalize import IntensityNormalizer

from .gridder import Gridder2D
from .gridder import Gridder3D

from .utilities import maplog
from .utilities import lam2en
