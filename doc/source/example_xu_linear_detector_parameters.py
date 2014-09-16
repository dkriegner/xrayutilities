"""
example script to show how the detector parameters
such as pixel width, center channel and detector tilt
can be determined for a linear detector.
"""

import xrayutilities as xu
import os

# load any data file with with the detector spectra of a reference scan
# in the primary beam, here I use spectra measured with a Seifert XRD
# diffractometer
dfile = os.path.join("data","primarybeam_alignment20130403_2_dis350.nja")
s = xu.io.SeifertScan(dfile)

ang = s.axispos["T"] # detector angles during the scan
spectra = s.data[:,:,1] # detector spectra aquired

# determine detector parameters
# this function accepts some optional arguments to describe the goniometer
# see the API documentation
pwidth,cch,tilt = xu.analysis.linear_detector_calib(ang,spectra,usetilt=True)

