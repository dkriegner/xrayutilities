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
# Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>

import os

import xrayutilities as xu

en = 10300.0  # eV
datadir = os.path.join("data", "wire_")  # data path for CCD files
# template for the CCD file names
filetmp = os.path.join(datadir, "wire_12_%05d.edf.gz")

# manually selected images
# select images which have the primary beam fully on the CCD
imagenrs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

images = []
ang1 = []
ang2 = []

# read images and angular positions from the data file
for imgnr in imagenrs:
    filename = filetmp % imgnr
    edf = xu.io.EDFFile(filename)
    images.append(edf.data)
    ang1.append(float(edf.header['ESRF_ID01_PSIC_NANO_NU']))
    ang2.append(float(edf.header['ESRF_ID01_PSIC_NANO_DEL']))
    # or for newer EDF files (recorded in year >~2013)
    # ang1.append(edf.motors['nu'])
    # ang2.append(edf.motors['del'])

# call the fit for the detector parameters
# detector arm rotations and primary beam direction need to be given
param, eps = xu.analysis.sample_align.area_detector_calib(
    ang1, ang2, images, ['z+', 'y-'], 'x+',
    start=(None, None, 1.0, 45, 0, -0.7, 0),
    fix=(False, False, True, False, False, False, False),
    wl=xu.en2lam(en))
