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

import numpy
import xrayutilities as xu


class TestQ2AngFit(unittest.TestCase):
    energy = 15000

    @classmethod
    def setUpClass(cls):
        cls.qconv = xu.experiment.QConversion(['z+', 'y-', 'z-'],
                                              ['z+', 'y-'],
                                              [1, 0, 0])
        cls.hxrd = xu.HXRD((1, 0, 0), (0, 0, 1), en=cls.energy,
                           qconv=cls.qconv)
        cls.bounds = (0, (-180, 180), 0, (-1, 90), (-1, 90))
        qz = numpy.random.rand()
        cls.qvec = numpy.array(((numpy.random.rand()-0.5)*qz, 0, qz))

    def test_q2angfit(self):
        ang, qerror, errcode = xu.Q2AngFit(self.qvec, self.hxrd, self.bounds)
        qout = self.hxrd.Ang2Q(*ang)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.qvec[i], places=5)


if __name__ == '__main__':
    unittest.main()
