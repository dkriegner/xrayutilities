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
# Copyright (C) 2014-2020 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2026 Davis Rens

import math
import os
import tempfile
import unittest

import numpy

import xrayutilities as xu


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.el = xu.materials.elements.Dummy
        # test creation of database
        fid, cls.dbfilename = tempfile.mkstemp(suffix=".db")
        os.close(fid)
        xu.materials.database.createAndFillDatabase(cls.dbfilename)
        cls.db = xu.materials.DataBase(cls.dbfilename)
        cls.db.Open()
        cls.db.SetMaterial(cls.el.name)

    @classmethod
    def tearDownClass(cls):
        cls.db.Close()
        try:
            os.remove(cls.dbfilename)
        except OSError:
            print("OSError during cleaning up of temporary database file!")

    def test_db_f0(self):
        f0 = self.el.f0(0)
        self.assertAlmostEqual(f0, 1.0, places=10)

    def test_owndb_f0(self):
        f0 = self.db.GetF0(0)
        self.assertAlmostEqual(f0, 1.0, places=10)

    def test_db_f1_neg(self):
        f1 = self.el.f1(-1)
        self.assertTrue(math.isnan(f1))

    def test_owndb_f1_neg(self):
        f1 = self.db.GetF1(-1)
        self.assertTrue(math.isnan(f1))

    def test_db_f1(self):
        f1 = self.el.f1(1000)
        self.assertAlmostEqual(f1, 0.0, places=10)

    def test_owndb_f1(self):
        f1 = self.db.GetF1(1000)
        self.assertAlmostEqual(f1, 0.0, places=10)

    def test_db_f2_neg(self):
        f2 = self.el.f2(-1)
        self.assertTrue(math.isnan(f2))

    def test_owndb_f2_neg(self):
        f2 = self.db.GetF2(-1)
        self.assertTrue(math.isnan(f2))

    def test_db_f2(self):
        f2 = self.el.f2(1000)
        self.assertAlmostEqual(f2, 0.0, places=10)

    def test_owndb_f2(self):
        f2 = self.db.GetF2(1000)
        self.assertAlmostEqual(f2, 0.0, places=10)

    def test_atomic_array_shape_cache(self):
        for method in ("f0", "f", "f1", "f2"):
            values = numpy.linspace(
                0 if method in ("f0", "f") else 7000,
                3 if method in ("f0", "f") else 10000,
                6,
            )
            grid = values.reshape(2, 3)
            for first, second in ((values, grid), (grid, values)):
                with self.subTest(method=method, shape=second.shape):
                    atom = xu.materials.Atom("Si", 14)
                    scatter = getattr(atom, method)
                    expected = numpy.array(
                        [scatter(float(value)) for value in values]
                    ).reshape(second.shape)
                    scatter(first)
                    actual = scatter(second)
                    self.assertEqual(actual.shape, second.shape)
                    numpy.testing.assert_allclose(
                        actual, expected, rtol=1e-6, atol=1e-9
                    )

    def test_atomic_noncontiguous_arrays(self):
        for method, start, stop in (("f0", 0, 3), ("f1", 7000, 10000)):
            values = numpy.linspace(start, stop, 6)
            grid = values.reshape(2, 3)
            for view in (values[::2], grid.T, numpy.asfortranarray(grid)):
                with self.subTest(method=method, shape=view.shape):
                    atom = xu.materials.Atom("Si", 14)
                    scatter = getattr(atom, method)
                    expected = numpy.array(
                        [scatter(float(value)) for value in view.flat]
                    ).reshape(view.shape)
                    numpy.testing.assert_allclose(
                        scatter(view), expected, rtol=1e-6, atol=1e-9
                    )


if __name__ == "__main__":
    unittest.main()
