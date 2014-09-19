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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy
import unittest

class TestQ2Ang_HXRD(unittest.TestCase):

    def setUp(self):
        self.mat = xu.materials.GeTe
        qconv = xu.QConversion(['x+','y+','z-'],'x+',[0,1,0])
        inp = numpy.cross(self.mat.Q(1,-1,0),self.mat.Q(1,1,1))
        self.hxrd = xu.HXRD(inp,self.mat.Q(1,1,1),qconv=qconv)
        self.hkltest = (1,3,2)

    def test_Q2Ang_hxrd_point(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hkltest))
        qout = self.hxrd.Ang2HKL(*ang,mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hkltest[i], places=10)

    def test_Q2Ang_hxrd_geom_hi_lo(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hkltest),geometry='hi_lo')
        qout = self.hxrd.Ang2HKL(*ang,mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hkltest[i], places=10)

    def test_Q2Ang_hxrd_geom_lo_hi(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hkltest),geometry='lo_hi')
        qout = self.hxrd.Ang2HKL(*ang,mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hkltest[i], places=10)

    def test_Q2Ang_hxrd_geom_real(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hkltest),geometry='real')
        qout = self.hxrd.Ang2HKL(*ang,mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hkltest[i], places=10)

    def test_Q2Ang_hxrd_geom_realTilt(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hkltest),geometry='realTilt')
        qout = self.hxrd.Ang2HKL(*ang,mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hkltest[i], places=10)

if __name__ == '__main__':
        unittest.main()
