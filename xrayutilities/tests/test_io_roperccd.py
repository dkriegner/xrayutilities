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

import unittest
import os.path

import xrayutilities as xu

testfile = 'roperccd.bin.gz'
datadir = 'data'
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_RoperCCD(unittest.TestCase):
    dshape = (4096, 4096)
    dmax = 2285.0
    dmin = 273.0
    tpos = (500, 500)
    dtpos = 304.0

    @classmethod
    def setUpClass(cls):
        imgreader = xu.io.RoperCCD()
        cls.data = imgreader.readImage(testfile, path=datadir)

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
