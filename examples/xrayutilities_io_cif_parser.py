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

import xrayutilities as xu
import numpy
import os
#import matplotlib.pyplot as plt

# parse cif file to get unit cell structure
cif_cal = xu.materials.CIFFile(os.path.join("data","Calcite.cif"))

#create material
Calcite = xu.materials.Material("Calcite",cif_cal.Lattice())

#experiment class with some weird directions
expcal = xu.HXRD(Calcite.Q(-2,1,9),Calcite.Q(1,-1,4))

powder_cal = xu.Powder(Calcite)
powder_cal.PowderIntensity()
th,inte = powder_cal.Convolute(0.002,0.02)

print(powder_cal)

#plt.figure()
#plt.clf()
#plt.bar(powder_cal.ang*2,powder_cal.data,align='center')
#plt.plot(th*2,inte,'k-',lw=2)
#plt.xlabel("2Theta (deg)")
#plt.ylabel("Intensity")

