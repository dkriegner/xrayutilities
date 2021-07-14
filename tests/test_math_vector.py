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
# Copyright (C) 2012-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu


class TestMathVector(unittest.TestCase):
    def test_distance(self):
        vec = (3, 1, 1)
        point = (-4, -5, -1)
        x, y, z = (-6, 1, 21)
        d = xu.math.vector.distance(numpy.array((x, x)),
                                    numpy.array((y, y)),
                                    numpy.array((z, z)),
                                    point, vec)
        self.assertEqual(d.size, 2)
        self.assertAlmostEqual(d[0], 4*numpy.sqrt(30), places=10)
        self.assertAlmostEqual(d[1], 4*numpy.sqrt(30), places=10)


if __name__ == '__main__':
    unittest.main()
