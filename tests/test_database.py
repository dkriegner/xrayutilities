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
import math

import xrayutilities as xu
import numpy


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.el = xu.materials.elements.dummy

    def test_db_f0(self):
        f0 = self.el.f0(0)
        self.assertAlmostEqual(f0, 1.0, places=10)

    def test_db_f1_neg(self):
        f1 = self.el.f1(-1)
        self.assertTrue(math.isnan(f1))

    def test_db_f1(self):
        f1 = self.el.f1(1000)
        self.assertAlmostEqual(f1, 0.0, places=10)

    def test_db_f2_neg(self):
        f2 = self.el.f2(-1)
        self.assertTrue(math.isnan(f2))

    def test_db_f2(self):
        f2 = self.el.f2(1000)
        self.assertAlmostEqual(f2, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
