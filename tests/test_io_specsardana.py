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
# Copyright (C) 2019-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import math
import os.path
import unittest

import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'sardana2.8.spec'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_SPEC_Sardana(unittest.TestCase):
    scannrs = [4, 6, 8, 10]
    s4_init_motor_pos_len = 6
    s8_init_motor_pos_len = 2
    s8_init_mopo_mot12 = 20.0
    s8_dshape = (29,)
    s8_ncol = 9
    s8_dtmax = 10.7484171391
    s8_dtmin = 1.35413002968
    s8_mot15max = -4.4
    s8_mot15min = -10.0
    s10_pos = 119
    s10_dpos = 119
    countername = 'Pt_No'

    @classmethod
    def setUpClass(cls):
        cls.specfile = xu.io.SPECFile(testfile, path=datadir)
        cls.scans = dict()
        for nr in cls.scannrs:
            cls.scans[nr] = getattr(cls.specfile, 'scan%d' % nr)
            cls.scans[nr].ReadData()

    def test_init_mopo(self):
        self.assertEqual(self.s4_init_motor_pos_len,
                         len(self.scans[4].init_motor_pos))
        self.assertEqual(self.s8_init_motor_pos_len,
                         len(self.scans[8].init_motor_pos))
        self.assertAlmostEqual(
            self.s8_init_mopo_mot12,
            self.scans[8].init_motor_pos['INIT_MOPO_mot12'], places=6)
        with self.assertRaises(KeyError):
            # test not existing initial motor position
            self.scans[8].init_motor_pos['INIT_MOPO_offset03']

    def test_datashape_aborted(self):
        self.assertEqual(self.s8_dshape, self.scans[8].data.shape)
        self.assertEqual(self.s8_ncol, len(self.scans[8].colnames))

    def test_datavalues(self):
        self.assertTrue(math.isnan(self.scans[6].data['ct13'][29]))
        s8d = self.scans[8].data
        self.assertAlmostEqual(self.s8_dtmax, s8d['dt'].max(), places=6)
        self.assertAlmostEqual(self.s8_dtmin, s8d['dt'].min(), places=6)
        self.assertAlmostEqual(self.s8_mot15max, s8d['mot15'].max(), places=6)
        self.assertAlmostEqual(self.s8_mot15min, s8d['mot15'].min(), places=6)
        s10d = self.scans[10].data
        self.assertAlmostEqual(self.s10_dpos,
                               s10d[self.countername][self.s10_pos], places=6)


if __name__ == '__main__':
    unittest.main()
