"""
example script to show the detector parameter determination for area detectors from images recorded in the primary beam
"""

import xrayutilities as xu
import os

en=10300.0 #eV
datadir = os.path.join("data","wire_") # data path for CCD files
filetmp = os.path.join(datadir,"wire_12_%05d.edf.gz") # template for the CCD file names

## manually selected images 
# select images which have the primary beam fully on the CCD
imagenrs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]

images = []
ang1 = []
ang2 = []

# read images and angular positions from the data file
# this might differ for data taken at different beamlines since
# they way how motor positions are stored is not always consistent
for imgnr in imagenrs:
    filename = filetmp%imgnr
    edf = xu.io.EDFFile(filename)
    images.append(edf.data)
    ang1.append(float(edf.header['ESRF_ID01_PSIC_NANO_NU']))
    ang2.append(float(edf.header['ESRF_ID01_PSIC_NANO_DEL']))


# call the fit for the detector parameters 
# detector arm rotations and primary beam direction need to be given 
# in total 8 parameters are fitted, however the 4 misalignment parameters can be fixed
# they are the detector tilt azimuth, the detector tilt angle, the detector rotation around the primary beam and the outer angle offset
param,eps = xu.analysis.sample_align.area_detector_calib(ang1,ang2,images,['z+','y-'],'x+',start=(45,0,-0.7,0),fix=(False,False,False,False),wl=xu.en2lam(en))

