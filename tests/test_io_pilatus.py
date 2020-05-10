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
# Copyright (C) 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu

testfile = 'pilatus100K.tif'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_Pilatus(unittest.TestCase):
    dshape = (195, 487)
    dmax = 1.0
    dmin = -2.0
    tpos = (75, 281)
    dtpos = 1.0

    @classmethod
    def setUpClass(cls):
        cls.data = xu.io.get_tiff(testfile, path=datadir)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.dmax, self.data.max(), places=10)
        self.assertAlmostEqual(self.dmin, self.data.min(), places=10)
        self.assertAlmostEqual(self.dtpos,
                               self.data[self.tpos[0], self.tpos[1]],
                               places=10)


if __name__ == '__main__':
    unittest.main()
