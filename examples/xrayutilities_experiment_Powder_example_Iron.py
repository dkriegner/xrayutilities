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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable

import xrayutilities as xu


def main():
    """dummy main function to enable multiprocessing on windows"""
    cryst_size = 40e-9  # meter

    # create Fe BCC (space group nr. 229 Im3m) with a = 2.87 angstrom although
    # this is already predefined as xu.materials.Fe we will repeat here for
    # educational purposes
    FeBCC = xu.materials.Crystal(
        "Fe", xu.materials.SGLattice(229, 2.87,
                                     atoms=[xu.materials.elements.Fe, ],
                                     pos=['2a', ]))

    print("Creating Fe powder ...")
    Fe_powder = xu.simpack.Powder(FeBCC, 1, crystallite_size_gauss=cryst_size)
    pm = xu.simpack.PowderModel(Fe_powder)
    tt = numpy.arange(5, 120, 0.01)
    inte = pm.simulate(tt)

    print(pm)

    # # to create a mixed powder sample one would use
    # Co_powder = xu.simpack.Powder(xu.materials.Co, 5)  # 5 times more Co
    # pmix = xu.simpack.PowderModel(Fe_powder + Co_powder, I0=100)
    # inte = pmix.simulate(tt)
    # pmix.close()  # after end-of-use for cleanup

    ax = pm.plot(tt)
    ax.set_xlim(5, 120)
    pm.close()


if __name__ == '__main__':
    freeze_support()
    main()
