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
import matplotlib.pyplot as plt

energy = (2*8048 + 8028)/3. # copper k alpha 1,2

# creating Indium powder
In_powder = xu.Powder(xu.materials.In,en=energy)
# calculating the reflection strength for the powder
In_powder.PowderIntensity()

# convoluting the peaks with a gaussian in q-space
peak_width = 0.01 # in q-space
resolution = 0.0005 # resolution in q-space
In_th,In_int = In_powder.Convolute(resolution,peak_width)

plt.figure(); plt.clf()
ax1 = plt.subplot(111)
plt.xlabel(r"2Theta (deg)"); plt.ylabel(r"Intensity")
# plot the convoluted signal
plt.plot(In_th*2,In_int/In_int.max(),'k-',label="Indium powder convolution")
# plot each peak in a bar plot
plt.bar(In_powder.ang*2, In_powder.data/In_powder.data.max(), width=0.3, bottom=0, linewidth=0, color='r',align='center', orientation='vertical',label="Indium powder bar plot")

plt.legend(); ax1.set_xlim(15,100); plt.grid()
