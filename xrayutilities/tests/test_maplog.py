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

import xrayutilities as xu
import numpy


class Test_maplog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dmin = xu.config.DYNLOW
        cls.dmax = xu.config.DYNHIGH
                
    def test_maplog(self):
        d = numpy.logspace(1,6,100)
        # make noop
        dm = xu.maplog(d, numpy.inf, -numpy.inf)
        self.assertTrue(numpy.all(numpy.log10(d) == dm))
        # cut bottom
        dl = 3
        dm = xu.maplog(d, dl, -numpy.inf)
        dl = min(dl, self.dmin)
        self.assertAlmostEqual(d.max(), 10.0**dm.max(), places=10)
        self.assertAlmostEqual(d.max()/10.0**dl, 10**dm.min(), places=10)
        # cut top
        dt = 2
        dm = xu.maplog(d, numpy.inf, dt)
        dt = max(dt, self.dmax)
        self.assertAlmostEqual(d.min(), 10.0**dm.min(), places=10)
        self.assertAlmostEqual(d.max()/10.0**dt, 10.0**dm.max(), places=10)

    def test_maplogzero(self):
        d = numpy.array((0, 1))
        # make function call with a zero and negative number
        dm = xu.maplog(d)
        self.assertAlmostEqual(d.max()/10.0**self.dmax,
                               10.0**dm.max(), places=10)
        self.assertAlmostEqual(d.max()/10.0**self.dmin,
                               10**dm.min(), places=10)
        # call with negative number
        d[0] = -1
        dm = xu.maplog(d)
        self.assertAlmostEqual(d.max()/10.0**self.dmax,
                               10.0**dm.max(), places=10)
        self.assertAlmostEqual(d.max()/10.0**self.dmin,
                               10**dm.min(), places=10)
 


if __name__ == '__main__':
    unittest.main()
