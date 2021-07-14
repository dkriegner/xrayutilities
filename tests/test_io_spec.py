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
import tempfile
import unittest

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'specmca.spec.gz'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_SPEC(unittest.TestCase):
    dshape = (4001,)
    dmax = 2567926.75
    dmin = 1.0
    motmax = 95.407753
    motmin = 15.40775
    date = 'Mon Nov 04 21:18:05 2013'
    tpos = 2400
    dtpos = 6.0
    scannr = 43
    motorname = 'Nu'
    countername = 'PSDCORR'

    @classmethod
    def setUpClass(cls):
        cls.specfile = xu.io.SPECFile(testfile, path=datadir)
        cls.specfile.Update()  # this should be a noop
        cls.specscan = getattr(cls.specfile, 'scan%d' % cls.scannr)
        cls.specscan.ReadData()
        cls.sdata = cls.specscan.data
        cls.motor, cls.inte = xu.io.getspec_scan(cls.specfile, cls.scannr,
                                                 cls.motorname,
                                                 cls.countername)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.sdata.shape)
        self.assertEqual(self.dshape, self.inte.shape)
        self.assertEqual(self.dshape, self.motor.shape)

    def test_getheader(self):
        self.assertEqual(self.date, self.specscan.getheader_element('D'))

    def test_datavalues(self):
        self.assertAlmostEqual(self.motmax, self.motor.max(), places=6)
        self.assertAlmostEqual(self.motmin, self.motor.min(), places=6)
        self.assertAlmostEqual(self.dmax, self.inte.max(), places=6)
        self.assertAlmostEqual(self.dmin, self.inte.min(), places=6)
        self.assertAlmostEqual(self.dtpos, self.inte[self.tpos], places=6)

    def test_datamethods(self):
        self.assertTrue(numpy.all(self.sdata[self.countername] == self.inte))

    def test_hdf5file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'tmp.h5')
            self.specfile.Save2HDF5(fname)
            h5d = xu.io.geth5_scan(fname, self.scannr)
            self.assertTrue(numpy.all(self.inte == h5d[self.countername]))


if __name__ == '__main__':
    unittest.main()
