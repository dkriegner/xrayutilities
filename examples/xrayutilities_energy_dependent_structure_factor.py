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

import matplotlib.pyplot as plt
import numpy
import xrayutilities as xu

# defining material and experimental setup
InAs = xu.materials.InAs
energy = numpy.linspace(500, 20000, 5000)  # 500 - 20000 eV

F = InAs.StructureFactorForEnergy(InAs.Q(1, 1, 1), energy)

plt.figure()
plt.clf()
plt.plot(energy, F.real, 'k-', label='Re(F)')
plt.plot(energy, F.imag, 'r-', label='Imag(F)')
plt.xlabel("Energy (eV)")
plt.ylabel("F")
plt.legend()
