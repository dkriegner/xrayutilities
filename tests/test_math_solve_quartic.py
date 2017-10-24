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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu

digits = 10


class TestSolveQuartic(unittest.TestCase):
    z = [(-2.1120821587966274-2.945550430762482j),
         (-2.1120821587966274+2.945550430762482j),
         0.6655848599174902,
         2.0585794576757652]

    def test_solver(self):
        roots = xu.math.solve_quartic(2, 3, 6, -60, 36)
        roots = numpy.sort(roots)

        for i, root in enumerate(roots):
            self.assertAlmostEqual(root, self.z[i], places=digits)


if __name__ == '__main__':
    unittest.main()
