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


class Test_KinematicalModel(unittest.TestCase):
    # define used layer stack
    sub = xu.simpack.Layer(xu.materials.GaAs, numpy.inf)
    lay = xu.simpack.Layer(xu.materials.AlGaAs(0.75), 995.64, relaxation=0.0)
    pls = xu.simpack.PseudomorphicStack001('AlGaAs on GaAs', sub, lay)
    hkl = (0, 0, 4)

    # simulation parameters
    kwargs = dict(I0=1e6, background=0, resolution_width=0.0001)

    @classmethod
    def setUpClass(cls):
        cls.kinmod = xu.simpack.KinematicalModel(cls.pls, **cls.kwargs)
        cls.kinmul = xu.simpack.KinematicalMultiBeamModel(cls.pls,
                                                          **cls.kwargs)
        cls.fm = xu.simpack.FitModel(cls.kinmod)
        cls.fmm = xu.simpack.FitModel(cls.kinmul)
        cls.qz = numpy.linspace(4.40, 4.50, 2000)

    def test_Calculation(self):
        sim = self.kinmod.simulate(self.qz, hkl=self.hkl, refraction=True)
        self.assertEqual(len(sim), len(self.qz))
        self.assertTrue(numpy.all(sim >= self.kwargs['background']))
        # Next line is actually not True for the kinematic model!
        # self.assertTrue(numpy.all(sim <= self.kwargs['I0']))

    def test_FitModel_eval(self):
        sim1 = self.kinmod.simulate(self.qz, hkl=self.hkl, refraction=True)
        p = self.fm.make_params()
        sim2 = self.fm.eval(p, x=self.qz, hkl=self.hkl, refraction=True)
        for v1, v2 in zip(sim1, sim2):
            self.assertAlmostEqual(v1, v2, places=10)

    def test_FitModel_eval_mult(self):
        sim1 = self.kinmul.simulate(self.qz, hkl=self.hkl, refraction=True)
        p = self.fmm.make_params()
        sim2 = self.fmm.eval(p, x=self.qz, hkl=self.hkl, refraction=True)
        for v1, v2 in zip(sim1, sim2):
            self.assertAlmostEqual(v1, v2, places=10)

    def test_Consistency(self):
        sim1 = self.kinmod.simulate(self.qz, hkl=self.hkl, refraction=True)
        sim2 = self.kinmul.simulate(self.qz, hkl=self.hkl, refraction=True)
        self.assertEqual(len(sim1), len(sim2))
        self.assertEqual(numpy.argmax(sim1), numpy.argmax(sim2))
        self.assertAlmostEqual(xu.math.fwhm_exp(self.qz, sim1),
                               xu.math.fwhm_exp(self.qz, sim2), places=6)
        self.assertTrue(xu.math.fwhm_exp(self.qz, sim1) >=
                        self.kwargs['resolution_width'])


if __name__ == '__main__':
    unittest.main()
