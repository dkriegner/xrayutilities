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
# Copyright (C) 2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 0
testfile = 'rg5041sup1.cif'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestMAT_CIF(unittest.TestCase):
    sg = '129:2'
    mata = 3.820
    matc = 6.318

    @classmethod
    def setUpClass(cls):
        cls.mat = xu.materials.Crystal.fromCIF(fullfilename)

    def test_spacegroup(self):
        self.assertEqual(self.mat.lattice.space_group, self.sg)

    def test_latticeparameters(self):
        self.assertAlmostEqual(self.mat.lattice.a, self.mata, places=6)
        self.assertAlmostEqual(self.mat.lattice.c, self.matc, places=6)

    def test_structurefactor(self):
        Q = self.mat.Q(numpy.random.randint(-3, 4, size=3))
        self.assertAlmostEqual(self.mat.StructureFactor(Q),
                               xu.materials.CuMnAs.StructureFactor(Q),
                               places=3)


if __name__ == '__main__':
    unittest.main()
