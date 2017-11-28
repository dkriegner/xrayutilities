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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy
import xrayutilities as xu

# config for kappa geometry
# set it either in the script or include it in on of the config files

xu.config.KAPPA_PLANE = 'zy'
xu.config.KAPPA_ANGLE = -60

# kappa goniometer as shown in
# http://en.wikipedia.org/wiki/File:Kappa_goniometer_animation.ogg
qconv = xu.experiment.QConversion(['z+', 'k+', 'z+'], ['z+'], (1, 0, 0))

print(qconv)

print("angles: 0, 0, 0, 90")
(qx, qy, qz) = qconv(0, 0, 0, 90)
print("Q= %6.3f %6.3f %6.3f (Abs: %6.3f)"
      % (qx, qy, qz, numpy.linalg.norm((qx, qy, qz))))

print("angles: 90, 0, 0, 90")
(qx, qy, qz) = qconv(90, 0, 0, 90)
print("Q= %6.3f %6.3f %6.3f (Abs: %6.3f)"
      % (qx, qy, qz, numpy.linalg.norm((qx, qy, qz))))

print("angles: 0, 90, 0, 90")
(qx, qy, qz) = qconv(0, 90, 0, 90)
print("Q= %6.3f %6.3f %6.3f (Abs: %6.3f)"
      % (qx, qy, qz, numpy.linalg.norm((qx, qy, qz))))
