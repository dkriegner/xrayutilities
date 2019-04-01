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
# Copyright (C) 2019 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy

import xrayutilities as xu

try:
    import lmfit
except ImportError:
    lmfit = None


@unittest.skipIf(lmfit is None, "the lmfit Python package is needed")
class Test_SpecularReflectivityModel(unittest.TestCase):
    # define used layer stack
    lSiO2 = xu.simpack.Layer(xu.materials.SiO2, numpy.inf, roughness=3.0)
    lRu = xu.simpack.Layer(xu.materials.Ru, 50, roughness=2.5)
    rho_cf = 0.5*8900 + 0.5*7874
    mat_cf = xu.materials.Amorphous('CoFe', rho_cf)
    lCoFe = xu.simpack.Layer(mat_cf, 30, roughness=5.2)

    # simulation parameters
    kwargs = dict(I0=1e6, background=2, sample_width=10, beam_width=0.2,
                  energy='MoKa1')

    @classmethod
    def setUpClass(cls):
        cls.model = xu.simpack.SpecularReflectivityModel(cls.lSiO2, cls.lRu,
                                                         cls.lCoFe,
                                                         **cls.kwargs)
        cls.fm = xu.simpack.FitModel(cls.model)
        cls.ai = numpy.arange(0.1, 3, 0.005)

    def test_Calculation(self):
        sim = self.model.simulate(self.ai)
        self.assertEqual(len(sim), len(self.ai))
        self.assertTrue(numpy.all(sim >= self.kwargs['background']))
        self.assertTrue(numpy.all(sim <= self.kwargs['I0']))

    def test_FitModel_eval(self):
        sim1 = self.model.simulate(self.ai)
        p = self.fm.make_params()
        sim2 = self.fm.eval(p, x=self.ai)
        for v1, v2 in zip(sim1, sim2):
            self.assertAlmostEqual(v1, v2, places=10)

    def test_densityprofile(self):
        N = 123
        z, d = self.model.densityprofile(N)
        self.assertEqual(len(z), N)
        self.assertEqual(len(d), N)


if __name__ == '__main__':
    unittest.main()
