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
# Copyright (C) 2026 Davis Rens

import os.path
import unittest

import numpy
import xrayutilities as xu

testfile = "pilatus100K.tif"
datadir = os.path.join(os.path.dirname(__file__), "data")
fullfilename = os.path.join(datadir, testfile)


class TestPointNormalizer(unittest.TestCase):
    def test_smoothed_linear_monitor(self):
        monitor = numpy.arange(1.0, 22.0)
        data = numpy.rec.fromarrays(
            [10 * monitor, monitor, numpy.ones_like(monitor)],
            names=["counts", "monitor", "time"],
        )
        for window in (3, 5, 7):
            with self.subTest(window=window):
                normalizer = xu.IntensityNormalizer(
                    "counts",
                    mon="monitor",
                    time="time",
                    av_mon=1.0,
                    smoothmon=window,
                )
                corrected = normalizer(data)
                margin = window // 2
                numpy.testing.assert_allclose(
                    corrected[margin:-margin], 10.0, rtol=0, atol=1e-12
                )


@unittest.skipIf(
    not os.path.isfile(fullfilename),
    "additional test data needed (http://xrayutilities.sf.io)",
)
class TestCCD_Normalizer(unittest.TestCase):
    dshape = (2, 195, 487)
    dmax = 0.1
    dmin = -0.2
    nmax = 21

    @classmethod
    def setUpClass(cls):
        imgreader = xu.io.Pilatus100K()
        cls.img = imgreader.readImage(testfile, path=datadir)
        cls.ccddata = numpy.asarray((cls.img, 2 * cls.img))
        cls.fakedata = numpy.array(
            [(1.0, 10.0), (2.0, 21.0)],
            dtype=[("time", float), ("moni", float)],
        )
        cls.normalizer = xu.IntensityNormalizer(
            mon="moni", time="time", av_mon=1.0
        )
        cls.normdata = cls.normalizer(cls.fakedata, ccd=cls.ccddata)

    def test_datashape(self):
        self.assertTrue(numpy.all(self.dshape == self.normdata.shape))

    def test_datavalues(self):
        self.assertAlmostEqual(self.dmax, self.normdata.max(), places=10)
        self.assertAlmostEqual(self.dmin, self.normdata.min(), places=10)
        testhist, _binedges = numpy.histogram(self.normdata, bins=30)
        self.assertEqual(self.nmax, testhist[-1])


if __name__ == "__main__":
    unittest.main()
