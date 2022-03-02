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
# Copyright (C) 2020-2021 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2020 Mike Moron <mike.moron@tu-dortmund.de>

import matplotlib.pyplot as plt
import numpy as np

import xrayutilities as xu

# the script below currently only works for Amorphous materials.
# file an issue on github if you need/want this for Crystal objects

# create a fictitious LayerStack with thin, rough Layers to illustrate the
# difference between the slicing approach and the usual layered approach
Si = xu.materials.Amorphous('Si', 2285)
SiO2 = xu.materials.Amorphous('SiO2', 1000)
C = xu.materials.Amorphous('C', 800)

s = xu.simpack.Layer(Si, np.inf, roughness=20)
l1 = xu.simpack.Layer(SiO2, 35, roughness=15)
l2 = xu.simpack.Layer(C, 15, roughness=8)

ls = s + l1 + l2

# conventional X-ray reflectivity modelling
m = xu.simpack.SpecularReflectivityModel(ls)
pos, eldens, layer_eldens = m.densityprofile(500, individual_layers=True)


# slice the layerstack into an Amorphous sublayer at every 0.1 angstrom.
# at the top a vacuum layer is added
sls = xu.simpack.effectiveDensitySlicing(ls, 0.1)
ms = xu.simpack.SpecularReflectivityModel(sls)
spos, seldens = ms.densityprofile(500)

# perform simulation and plot simulation and density profile
alpha = np.linspace(0., 5., num=500)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(alpha, m.simulate(alpha), label='conventional XRR')
ax.semilogy(alpha, ms.simulate(alpha), label='sliced XRR')
ax.set_xlabel(r'incidence angle')
ax.set_ylabel(r'reflectivity')
ax.legend()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pos, eldens, '.-', label='conventional')
for i in range(len(layer_eldens)):
    ax.plot(pos, layer_eldens[i], ':')
ax.plot(spos, seldens, '.-', label='sliced')  # arbitrary shift for vis.
ax.legend()
ax.set_xlabel(r'z-position')
ax.set_ylabel(r'electron density')

plt.show()
