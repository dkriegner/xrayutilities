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

import unittest

import numpy

import xrayutilities as xu


class TestStructureFactor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = xu.materials.elements.Dummy
        cls.mat = xu.materials.Crystal(
            "test",
            xu.materials.SGLattice(
                "227:1",
                4,
                atoms=[
                    cls.at,
                ],
                pos=[
                    "8a",
                ],
            ),
        )

    def test_StructureFactor(self):
        f = self.mat.StructureFactor(self.mat.Q(1, 3, 1))
        self.assertAlmostEqual(f, 4 - 4j, places=10)
        f = self.mat.StructureFactor(self.mat.Q(0, 4, 0))
        self.assertAlmostEqual(f, 8, places=10)
        f = self.mat.StructureFactor(self.mat.Q(1, 2, 1))
        self.assertAlmostEqual(f, 0, places=10)

    def test_StructureFactorHKL(self):
        f = self.mat.StructureFactorForHKL((1, 3, 1))
        self.assertAlmostEqual(f, 4 - 4j, places=10)
        f = self.mat.StructureFactorForHKL([0, 4, 0])
        self.assertAlmostEqual(f, 8, places=10)
        f = self.mat.StructureFactorForHKL((1, 2, 1))
        self.assertAlmostEqual(f, 0, places=10)

    def test_StructureFactorQ(self):
        q = (self.mat.Q(1, 1, 1), self.mat.Q(0, 4, 0), self.mat.Q(1, 2, 1))
        f = self.mat.StructureFactorForQ(q)
        for i in range(3):
            self.assertAlmostEqual(f[i], (4 + 4j, 8, 0)[i], places=10)

    def test_StructureFactorE(self):
        q = self.mat.Q(1, 1, 1)
        f = self.mat.StructureFactorForEnergy(q, (1000, 2000, 3000))
        for i in range(3):
            self.assertAlmostEqual(f[i], 4 + 4j, places=10)

    def test_mixed_valence_structure_factors(self):
        sites = [
            (xu.materials.elements.Fe2p, (0, 0, 0), 0.4, 0.5),
            (xu.materials.elements.Fe3p, (0.5, 0, 0), 0.75, 1.2),
        ]
        for ordered_sites in (sites, sites[::-1]):
            with self.subTest(first_atom=ordered_sites[0][0].name):
                atoms, positions, occupations, factors = zip(*ordered_sites)
                mat = xu.materials.Crystal(
                    "mixed-valence iron",
                    xu.materials.SGLattice(
                        1,
                        4,
                        4,
                        4,
                        90,
                        90,
                        90,
                        atoms=atoms,
                        pos=[("1a", position) for position in positions],
                        occ=occupations,
                        b=factors,
                    ),
                )
                q = numpy.array([mat.Q(*h) for h in ((0, 0, 0), (1, 0, 0))])
                energies = numpy.array([6000.0, 8000.0, 10000.0])

                def site_sum(momentum, energy):
                    qnorm = numpy.linalg.norm(momentum, axis=-1)
                    return sum(
                        atom.f(qnorm, energy)
                        * occupancy
                        * numpy.exp(-1j * (momentum @ mat.lattice.GetPoint(p)))
                        * numpy.exp(-factor * qnorm**2 / (4 * numpy.pi) ** 2)
                        for atom, p, occupancy, factor in ordered_sites
                    )

                numpy.testing.assert_allclose(
                    [mat.StructureFactor(v, en=8000) for v in q],
                    [site_sum(v, 8000) for v in q],
                    rtol=1e-6,
                    atol=1e-9,
                )
                numpy.testing.assert_allclose(
                    mat.StructureFactorForQ(q, en0=8000),
                    site_sum(q, 8000),
                    rtol=1e-6,
                    atol=1e-9,
                )
                numpy.testing.assert_allclose(
                    mat.StructureFactorForEnergy(q[1], energies),
                    site_sum(q[1], energies),
                    rtol=1e-6,
                    atol=1e-9,
                )


if __name__ == "__main__":
    unittest.main()
