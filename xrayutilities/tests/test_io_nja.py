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

import xrayutilities as xu
import numpy

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'seifert.nja.bz2'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_NJA(unittest.TestCase):
    dshape = (501, 2)
    dmax = 1344706.25
    dmin = 90.0
    motmin = -0.05
    motmax = 0.15
    axisT = 1.5
    dtpos = [5.00000000e-02, 5.64250000e+03]
    tpos = 250

    @classmethod
    def setUpClass(cls):
        cls.njafile = xu.io.SeifertScan(testfile, path=datadir)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.njafile.data.shape)

    def test_headervalue(self):
        self.assertAlmostEqual(self.axisT,
                               self.njafile.axispos['T'][0], places=6)

    def test_datavalues(self):
        motor = self.njafile.data[:, 0]
        counter = self.njafile.data[:, 1]
        self.assertAlmostEqual(self.motmax, motor.max(), places=6)
        self.assertAlmostEqual(self.motmin, motor.min(), places=6)
        self.assertAlmostEqual(self.dmax, counter.max(), places=6)
        self.assertAlmostEqual(self.dmin, counter.min(), places=6)
        self.assertTrue(numpy.all(self.dtpos == self.njafile.data[self.tpos]))


if __name__ == '__main__':
    unittest.main()
