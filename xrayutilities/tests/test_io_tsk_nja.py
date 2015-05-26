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
testfile = 'seifert_tsk.nja.gz'
datadir = 'data'
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_task_NJA(unittest.TestCase):
    dshape = (4001, 1280)
    dmax = 143563.531
    dmin = 0.0
    motmin = 7.5
    motmax = 57.5
    dtpos = 1.026
    tpos = (1515, 640)

    @classmethod
    def setUpClass(cls):
        cls.mot, cls.mot2, cls.data = xu.io.getSeifert_map(testfile,
                                                           path=datadir,
                                                           scantype='tsk')

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.motmax, self.mot.max(), places=6)
        self.assertAlmostEqual(self.motmin, self.mot.min(), places=6)
        self.assertAlmostEqual(self.dmax, self.data.max(), places=6)
        self.assertAlmostEqual(self.dmin, self.data.min(), places=6)
        self.assertTrue(numpy.all(self.dtpos == self.data[self.tpos[0],
                                                          self.tpos[1]]))


if __name__ == '__main__':
    unittest.main()
