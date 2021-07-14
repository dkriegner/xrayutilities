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
# Copyright (C) 2012-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import xrayutilities as xu


class Test_maplog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = xu.materials.Amorphous('Ir0.2Mn0.8', 10130)

    def test_elements(self):
        self.assertEqual(self.a.base[0][0], xu.materials.elements.Ir)
        self.assertEqual(self.a.base[1][0], xu.materials.elements.Mn)

    def test_composition(self):
        self.assertAlmostEqual(self.a.base[0][1], 0.2, places=10)
        self.assertAlmostEqual(self.a.base[1][1], 0.8, places=10)


if __name__ == '__main__':
    unittest.main()
