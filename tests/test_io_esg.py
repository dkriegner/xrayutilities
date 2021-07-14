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

import xrayutilities as xu

testfile = 'sc.esg'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_ESG(unittest.TestCase):
    dshape = (602, 4096)
    dmax = 69282.0
    dmin = 0.0
    motorname = '_pd_meas_angle_omega'
    motmin = 2.0
    motmax = 20.0
    dtpos = 9.0
    tpos = (200, 600)

    @classmethod
    def setUpClass(cls):
        cls.esgfile = xu.io.pdESG(fullfilename)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.esgfile.data.shape)

    def test_datavalues(self):
        motor = self.esgfile.fileheader[self.motorname]
        counter = self.esgfile.data
        self.assertAlmostEqual(self.motmax, motor.max(), places=6)
        self.assertAlmostEqual(self.motmin, motor.min(), places=6)
        self.assertAlmostEqual(self.dmax, counter.max(), places=6)
        self.assertAlmostEqual(self.dmin, counter.min(), places=6)
        self.assertAlmostEqual(counter[self.tpos[0], self.tpos[1]], self.dtpos)


if __name__ == '__main__':
    unittest.main()
