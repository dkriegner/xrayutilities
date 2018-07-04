"""
Example script to show how to use xrayutilities to read and plot
reciprocal space map scans from a spec file created at the ESRF/ID10B

for details about the measurement see:
    D Kriegner et al. Nanotechnology 22 425704 (2011)
    http://dx.doi.org/10.1088/0957-4484/22/42/425704
"""

import os

import matplotlib.pyplot as plt
import numpy
import xrayutilities as xu

# global setting for the experiment
sample = "test"  # sample name used also as file name for the data file
energy = 8042.5  # x-ray energy in eV
center_ch = 715.9  # center channel of the linear detector
chpdeg = 345.28  # channels per degree of the linear detector
roi = [100, 1340]  # region of interest of the detector
nchannel = 1500  # number of channels of the detector

# intensity normalizer function responsible for count time and absorber
# correction
normalizer_detcorr = xu.IntensityNormalizer(
    "MCA",
    mon="Monitor",
    time="Seconds",
    absfun=lambda d: d["detcorr"] / d["psd2"].astype(numpy.float))

# substrate material used for Bragg peak calculation to correct for
# experimental offsets
InP = xu.materials.InP

# initialize experimental class to specify the reference directions of your
# crystal
# 11-2: inplane reference
# 111: surface normal
hxrd = xu.HXRD(InP.Q(1, 1, -2), InP.Q(1, 1, 1), en=energy)

# configure linear detector
# detector direction + parameters need to be given
# mounted along z direction, which corresponds to twotheta
hxrd.Ang2Q.init_linear('z-', center_ch, nchannel, chpdeg=chpdeg, roi=roi)

# read spec file and save to HDF5-file
# since reading is much faster from HDF5 once the data are transformed
h5file = os.path.join("data", sample + ".h5")
try:
    s  # try if spec file object already exist ("run -i" in ipython)
except NameError:
    s = xu.io.SPECFile(sample + ".spec", path="data")
else:
    s.Update()
s.Save2HDF5(h5file)

#################################
# InP (333) reciprocal space map
omalign = 43.0529  # experimental aligned values
ttalign = 86.0733
[omnominal, dummy, dummy, ttnominal] = hxrd.Q2Ang(
    InP.Q(3, 3, 3))  # nominal values of the substrate peak

# read the data from the HDF5 file
# scan number:36, names of motors in spec file: omega= sample rocking, gamma =
# twotheta
[om, tt], MAP = xu.io.geth5_scan(h5file, 36, 'omega', 'gamma')
# normalize the intensity values (absorber and count time corrections)
psdraw = normalizer_detcorr(MAP)
# remove unusable detector channels/regions (no averaging of detector channels)
psd = xu.blockAveragePSD(psdraw, 1, roi=roi)

# convert angular coordinates to reciprocal space + correct for offsets
[qx, qy, qz] = hxrd.Ang2Q.linear(
    om, tt,
    delta=[omalign - omnominal, ttalign - ttnominal])

# calculate data on a regular grid of 200x201 points
gridder = xu.Gridder2D(200, 201)
gridder(qy, qz, psd)
# maplog function limits the shown dynamic range to 8 orders of magnitude
# from the maxium
INT = xu.maplog(gridder.data.T, 8., 0)

# plot the intensity as contour plot using matplotlib
plt.figure()
cf = plt.contourf(gridder.xaxis, gridder.yaxis, INT, 100, extend='min')
plt.xlabel(r'$Q_{[11\bar2]}$ ($\AA^{-1}$)')
plt.ylabel(r'$Q_{[\bar1\bar1\bar1]}$ ($\AA^{-1}$)')
cb = plt.colorbar(cf)
cb.set_label(r"$\log($Int$)$ (cps)")
