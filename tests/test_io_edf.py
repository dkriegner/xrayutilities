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
# Copyright (C) 2015,2019 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import tempfile
import unittest

import h5py
import numpy
import xrayutilities as xu

testfile = 'esrf.edf.gz'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_EDF(unittest.TestCase):
    dshape = (2048, 2048)
    dmax = 60134
    dmin = 72
    tpos = (500, 500)
    dtpos = 2349

    @classmethod
    def setUpClass(cls):
        cls.edffile = xu.io.EDFFile(testfile, path=datadir)
        cls.data = cls.edffile.data

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.dmax, self.data.max(), places=10)
        self.assertAlmostEqual(self.dmin, self.data.min(), places=10)
        self.assertAlmostEqual(self.dtpos,
                               self.data[self.tpos[0], self.tpos[1]],
                               places=10)

    def test_savehdf5(self):
        with tempfile.NamedTemporaryFile(mode='w') as fid:
            self.edffile.Save2HDF5(fid.name)
            with h5py.File(fid.name, 'r') as h5f:
                h5d = h5f[list(h5f.keys())[0]]
                h5d = numpy.asarray(h5d)
                self.assertTrue(numpy.all(h5d == self.data))

    def test_EDFDirectory(self):
        with tempfile.NamedTemporaryFile(mode='w') as fid:
            ed = xu.io.EDFDirectory(datadir, 'edf.gz')
            ed.Save2HDF5(fid.name)
            with h5py.File(fid.name, 'r') as h5f:
                h5g = h5f[os.path.split(datadir)[-1]]
                h5d = h5g[list(h5g.keys())[0]]
                h5d = numpy.asarray(h5d)
                self.assertTrue(numpy.all(h5d == self.data))


if __name__ == '__main__':
    unittest.main()
