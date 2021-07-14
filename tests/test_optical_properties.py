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
# Copyright (C) 2016-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu

digits = 11


class TestOpticalProperties(unittest.TestCase):
    en = 'CuKa1'
    n = numpy.complex128(1 - 2.3236677077820289e-05 + 3.1809448794498657e-06j)
    cn = 1 - 1.4554622905893488e-05 + 4.3128498279470241e-07j
    rho_cf = 0.5*8900 + 0.5*7874
    mat = xu.materials.Amorphous('CoFe', rho_cf, [('Co', 0.5), ('Fe', 0.5)])
    cmat = xu.materials.GaAs

    def test_idx_refraction(self):
        idx = self.mat.idx_refraction(en=self.en)
        self.assertAlmostEqual(idx, self.n, places=digits)
        idx = self.cmat.idx_refraction(en=self.en)
        self.assertAlmostEqual(idx, self.cn, places=digits)

    def test_delta_beta(self):
        n2 = 1 - self.mat.delta(en=self.en) + 1j * self.mat.ibeta(en=self.en)
        self.assertAlmostEqual(n2, self.n, places=digits)
        n2 = 1 - self.cmat.delta(en=self.en) + 1j * self.cmat.ibeta(en=self.en)
        self.assertAlmostEqual(n2, self.cn, places=digits)

    def test_chi0(self):
        n3 = 1 + self.mat.chi0(en=self.en) / 2.
        self.assertAlmostEqual(n3, numpy.complex128(self.n), places=digits)
        n3 = 1 + self.cmat.chi0(en=self.en) / 2.
        self.assertAlmostEqual(n3, self.cn, places=digits)


if __name__ == '__main__':
    unittest.main()
