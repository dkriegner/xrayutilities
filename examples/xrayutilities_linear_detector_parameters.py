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
# Copyright (c) 2013, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
script to show how the detector parameters (like pixel width, center channel
and detector tilt can be determined for a linear detector.
"""

import os

import xrayutilities as xu

en = xu.utilities.energies["CuKa1"]  # eV
dfile = os.path.join("data", "primarybeam_alignment20130403_2_dis350.nja")

# read seifert file
s = xu.io.SeifertScan(dfile)
ang = s.axispos["T"]
spectra = s.data[:, :, 1]

pwidth, cch, tilt = xu.analysis.linear_detector_calib(ang, spectra)
