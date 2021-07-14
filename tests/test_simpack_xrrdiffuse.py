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
# Copyright (C) 2018-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu


class Test_DiffuseReflectivityModel(unittest.TestCase):
    dmax = 2e-6
    # define used layer stack
    sub = xu.simpack.Layer(xu.materials.Si, numpy.inf, roughness=1,
                           lat_correl=100)
    lay1 = xu.simpack.Layer(xu.materials.Si, 200, roughness=1, lat_correl=200)
    lay2 = xu.simpack.Layer(xu.materials.Ge, 70, roughness=3, lat_correl=50)

    ls = xu.simpack.LayerStack('SL 5', sub+5*(lay2+lay1))

    # simulation parameters
    kwargs = dict(sample_width=10, beam_width=1, energy='CuKa1',
                  vert_correl=1000, vert_nu=0, H=1, method=1, vert_int=0)

    @classmethod
    def setUpClass(cls):
        cls.m1 = xu.simpack.DiffuseReflectivityModel(cls.ls, **cls.kwargs)
        cls.kwargs['H'] = cls.kwargs['H'] - xu.config.EPSILON
        cls.m2 = xu.simpack.DiffuseReflectivityModel(cls.ls, **cls.kwargs)
        cls.kwargs['H'] = 1
        cls.kwargs['method'] = 2
        cls.m3 = xu.simpack.DiffuseReflectivityModel(cls.ls, **cls.kwargs)
        cls.ai = numpy.arange(0.3, 2, 0.005)

    def test_Calculation(self):
        sim = self.m1.simulate(self.ai)
        self.assertEqual(len(sim), len(self.ai))
        self.assertTrue(numpy.all(sim >= 0))

    def test_Consistency(self):
        # calc models
        sim1 = self.m1.simulate(self.ai)
        sim2 = self.m2.simulate(self.ai)
        sim3 = self.m3.simulate(self.ai)

        self.assertTrue(abs(numpy.mean(sim1) - numpy.mean(sim2)) < self.dmax)
        self.assertTrue(abs(numpy.mean(sim1) - numpy.mean(sim3)) < self.dmax)


if __name__ == '__main__':
    unittest.main()
