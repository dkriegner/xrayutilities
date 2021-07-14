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

xu.config.VERBOSITY = 0
testfile = 'NISI.cif'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_pdCIF(unittest.TestCase):
    dshape = (4495, )
    # dmax =  # needs update of pdCIF to handle value + error bar
    # dmin =
    motorname = '_pd_meas_time_of_flight'
    motmin = 1000.0
    motmax = 8190.4
    # dtpos =
    # tpos =

    @classmethod
    def setUpClass(cls):
        cls.dfile = xu.io.pdCIF(fullfilename)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.dfile.data.shape)

    def test_datavalues(self):
        motor = self.dfile.data[self.motorname]
        # counter = self.dfile.data
        self.assertAlmostEqual(self.motmax, motor.max(), places=6)
        self.assertAlmostEqual(self.motmin, motor.min(), places=6)
        # self.assertAlmostEqual(self.dmax, counter.max(), places=6)
        # self.assertAlmostEqual(self.dmin, counter.min(), places=6)
        # self.assertAlmostEqual(counter[self.tpos[0], self.tpos[1]],
        #                        self.dtpos)


if __name__ == '__main__':
    unittest.main()
