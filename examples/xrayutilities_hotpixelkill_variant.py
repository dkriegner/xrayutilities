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
# Copyright (C) 2013 Raphael Grifone <raphael.grifone@esrf.fr>
# Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
The function below is a modification of the hotpixelkill function included in
the xrayutilities_id01_functions.py file.

Instead of manually specifying the list of hot pixel they are automatically
determined from a series of dark images. The idea is as follows: If in the
average data of several dark images (no intentional x-ray radiations during
acquisition) a signal above a certain threshold remains it is expected to arise
from a hot pixel and such pixels should be removed to avoid spikes in the data.
"""

import glob

import numpy
import xrayutilities as xu


def hotpixelkill(ccd):
    """
    function to remove hot pixels from CCD frames
    """
    ccd[hotpixelnumbers] = 0
    # one could also use NaN here, since NaN values are
    # ignored by the gridder
    return ccd


# identify hot pixels numbers from a series of dark images
# open a series of 2D detector frames
ccdavg = numpy.empty(0)
n = 0
for f in glob.glob(r"G:\data\dark*.edf"):
    e = xu.io.EDFFile(f)
    ccdraw = e.data
    if len(ccdavg) == 0:
        ccdavg = ccdraw.astype(numpy.float)
    else:
        ccdavg += ccdraw
    n += 1

ccdavg /= float(n)

# adjust treshold value: either as absolute value or in relation to the maximum
threshold = 10  # counts
# or
# threshold = ccdraw.max()*0.1 #take 10% of maximum intensity

# determine hot pixels by comparison with the threshold value
hotpixelnumbers = numpy.where(ccdraw > threshold)
