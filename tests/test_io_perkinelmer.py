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
# Copyright (C) 2015-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu

testfile = 'perkinelmer.tif.bz2'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_PerkinElmer(unittest.TestCase):
    dshape = (2048, 2048)
    dmax = 173359.0
    dmin = -113.0
    tpos = (500, 500)
    dtpos = 2929.0

    @classmethod
    def setUpClass(cls):
        imgreader = xu.io.PerkinElmer()
        cls.data = imgreader.readImage(testfile, path=datadir)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.dmax, self.data.max(), places=10)
        self.assertAlmostEqual(self.dmin, self.data.min(), places=10)
        self.assertAlmostEqual(self.dtpos,
                               self.data[self.tpos[0], self.tpos[1]],
                               places=10)

    def test_tiffread(self):
        t = xu.io.TIFFRead(testfile, path=datadir)
        self.assertTrue(numpy.all(t.data == self.data))


if __name__ == '__main__':
    unittest.main()
