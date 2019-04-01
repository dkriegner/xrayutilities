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
# Copyright (C) 2011-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities.analysis is a package for assisting with the analysis of
x-ray diffraction data, mainly reciprocal space maps

Routines for obtaining line cuts from gridded reciprocal space maps are
offered, with the ability to integrate the intensity perpendicular to the
line cut direction.
"""

from .line_cuts import (get_omega_scan, get_qx_scan, get_qy_scan, get_qz_scan,
                        get_radial_scan, get_ttheta_scan)
from .misc import coplanar_intensity, getangles, getunitvector
from .sample_align import (area_detector_calib, area_detector_calib_hkl,
                           fit_bragg_peak, linear_detector_calib, miscut_calc,
                           psd_chdeg, psd_refl_align)
