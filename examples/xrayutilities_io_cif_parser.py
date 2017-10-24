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

import os

import numpy
import xrayutilities as xu

# import matplotlib.pyplot as plt

# create material
Calcite = xu.materials.Crystal.fromCIF(os.path.join("data", "Calcite.cif"))

# experiment class with some weird directions
expcal = xu.HXRD(Calcite.Q(-2, 1, 9), Calcite.Q(1, -1, 4))

powder_cal = xu.simpack.PowderDiffraction(Calcite)
print(powder_cal)
