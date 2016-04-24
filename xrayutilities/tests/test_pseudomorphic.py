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

import unittest

import xrayutilities as xu
import numpy


class TestPseudomorphic(unittest.TestCase):
    mA = xu.materials.Si
    mB = xu.materials.Ge
    relaxation = numpy.random.rand()

    def test_pseudomoprhic001(self):
        mBpseudo = xu.materials.PseudomorphicMaterial(self.mA, self.mB,
                                                      self.relaxation)

        # calc lattice for mBpseudo
        asub = (self.mA.lattice.a + self.mA.lattice.b) / 2.
        abulk = (self.mB.lattice.a + self.mB.lattice.b) / 2.
        apar = asub + (abulk - asub) * self.relaxation
        epar = (apar - abulk) / abulk
        eperp = -2 * self.mB.c12 / self.mB.c11 * epar
        aperp = abulk * (1 + eperp)

        # check that angles are 90deg
        mBpl = mBpseudo.lattice
        for ang in (mBpl.alpha, mBpl.beta, mBpl.gamma):
            self.assertAlmostEqual(ang, 90, places=12)

        # check lattice parameters
        self.assertAlmostEqual(mBpl.a, apar, places=12)
        self.assertAlmostEqual(mBpl.b, apar, places=12)
        self.assertAlmostEqual(mBpl.c, aperp, places=12)

    def test_pseudomorphic111(self):
        t = xu.HXRD(self.mA.Q([1, 1, -2]), self.mA.Q([1, 1, 1]))._transform
        mBpseudo = xu.materials.PseudomorphicMaterial(self.mA, self.mB,
                                                      self.relaxation, trans=t)

        # calc lattice for mBpseudo
        def get_inplane111(l):
            """determine inplane lattice parameter for (111) surfaces"""
            return (xu.math.VecNorm(l.GetPoint(1, 1, -2)) / numpy.sqrt(6) +
                    xu.math.VecNorm(l.GetPoint(1, -1, 0)) / numpy.sqrt(2)) / 2
        asub = get_inplane111(self.mA.lattice)
        abulk = get_inplane111(self.mB.lattice)
        apar = asub + (abulk - asub) * self.relaxation
        epar = (apar - abulk) / abulk
        eperp = -epar * (2*self.mB.c11 + 4*self.mB.c12 - 4*self.mB.c44) /\
                        (self.mB.c11 + 2*self.mB.c12 + 4*self.mB.c44)
        eps = (eperp - epar + 3 * numpy.identity(3) * epar) / 3.

        # check that angles lattice spacings are correct
        self.assertAlmostEqual(xu.math.VecNorm(mBpseudo.Q(1, 1, 1)),
                               xu.math.VecNorm(self.mB.Q(1, 1, 1)) / (1+eperp),
                               places=12)
        self.assertAlmostEqual(xu.math.VecNorm(mBpseudo.Q(1, 1, -2)),
                               xu.math.VecNorm(self.mB.Q(1, 1, -2)) / (1+epar),
                               places=12)


if __name__ == '__main__':
    unittest.main()
