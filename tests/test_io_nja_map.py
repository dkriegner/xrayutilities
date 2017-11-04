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

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'seifert_map.nja.bz2'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_map_NJA(unittest.TestCase):
    dshape = (601601,)
    dmax = 12.8161590
    dmin = 0.0
    motmin = 77.0
    motmax = 80.0
    dtpos = 1.8286480
    tpos = 294840

    @classmethod
    def setUpClass(cls):
        cls.mot, cls.mot2, cls.data = xu.io.getSeifert_map(testfile,
                                                           path=datadir)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.motmax, self.mot.max(), places=6)
        self.assertAlmostEqual(self.motmin, self.mot.min(), places=6)
        self.assertAlmostEqual(self.dmax, self.data.max(), places=6)
        self.assertAlmostEqual(self.dmin, self.data.min(), places=6)
        self.assertAlmostEqual(self.dtpos, self.data[self.tpos], places=6)


if __name__ == '__main__':
    unittest.main()
