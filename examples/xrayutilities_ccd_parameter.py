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
# Copyright (C) 2018 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrayutilities_id01_functions.py

"""
example script for determining ESRF/ID01 detector parameters using a specfile
from the second half of 2017
"""

import os
import re

import xrayutilities as xu

import xrayutilities_id01_functions as id01

s = xu.io.SPECFile(specfile)
specscan = s.scan3
en = id01.getmono_energy(specscan)
# template for the CCD file names
filetmp = id01.getmpx4_filetmp(specscan)

images = []
ang1 = specscan.data['nu']
ang2 = specscan.data['del']

# read images and angular positions from the data file
for imgnr in imagenrs:
    filename = filetmp % imgnr
    edf = xu.io.EDFFile(filename)
    images.append(edf.data)

# call the fit for the detector parameters
# detector arm rotations and primary beam direction need to be given
param, eps = xu.analysis.sample_align.area_detector_calib(
    ang1, ang2, images, ['z-', 'y-'], 'x+',
    start=(55e-6, 55e-6, 0.5, 45, 0, -0.7, 0),
    fix=(True, True, False, False, False, False, True),
    wl=xu.en2lam(en))
