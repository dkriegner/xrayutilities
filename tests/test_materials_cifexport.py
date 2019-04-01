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
# Copyright (C) 2018-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

import os
import tempfile
import unittest

import xrayutilities as xu

xu.config.VERBOSITY = 0


class Test_CIF_export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.materials = []
        for name, obj in xu.materials.predefined_materials.__dict__.items():
            if isinstance(obj, xu.materials.Crystal):
                cls.materials.append(obj)

    def test_export(self):
        for m in self.materials:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as fid:
                filename = fid.name
            m.toCIF(filename)
            c = xu.materials.Crystal.fromCIF(filename)
            self.assertEqual(m, c)
            os.remove(filename)


if __name__ == '__main__':
    unittest.main()
