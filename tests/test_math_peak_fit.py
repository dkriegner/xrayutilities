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

digits = 3


class TestPeakFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sl, cls.back = numpy.random.rand(2) * 0.1
        cls.width = numpy.random.rand() * 0.05 + 0.025
        cls.amp = numpy.random.rand() + 0.7
        cls.pos = numpy.random.rand() * 0.3 + 0.3
        cls.x = numpy.linspace(-0.1, 0.9, 1000)
        p = [cls.pos, cls.width, cls.amp, cls.back]
        cls.fg = xu.math.Gauss1d(cls.x, *p)
        cls.fl = xu.math.Lorentz1d(cls.x, *p)
        cls.eta = numpy.random.rand()
        p += [cls.eta]
        cls.fv = xu.math.PseudoVoigt1d(cls.x, *p)

    def test_gaussfit(self):
        params, sd_params, itlim = xu.math.gauss_fit(self.x, self.fg)
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)

    def test_gaussfit_linear(self):
        f = self.fg + self.sl * self.x
        params, sd_params, itlim, ffunc = xu.math.peak_fit(
            self.x, f, peaktype='Gauss', background='linear', func_out=True)
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)
        self.assertAlmostEqual(params[4], self.sl, places=digits)

    def test_lorentzfit(self):
        params, sd_params, itlim = xu.math.peak_fit(
            self.x, self.fl, peaktype='Lorentz', background='constant')
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)

    def test_lorentzfit_linear(self):
        iparam = numpy.asarray([self.pos, self.width, self.amp, self.back,
                                self.sl])
        iparam += (numpy.random.rand(5) - 0.5) * 0.05
        f = self.fl + self.sl * self.x
        params, sd_params, itlim = xu.math.peak_fit(
            self.x, f, peaktype='Lorentz', background='linear', iparams=iparam)
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)
        self.assertAlmostEqual(params[4], self.sl, places=digits)

    def test_pvoigtfit(self):
        params, sd_params, itlim = xu.math.peak_fit(
            self.x, self.fv, peaktype='PseudoVoigt', background='constant')
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)
        self.assertAlmostEqual(params[4], self.eta, places=digits)

    def test_pvoigtfit_linear(self):
        f = self.fv + self.sl * self.x
        params, sd_params, itlim = xu.math.peak_fit(
            self.x, f, peaktype='PseudoVoigt', background='linear')
        self.assertAlmostEqual(params[0], self.pos, places=digits)
        self.assertAlmostEqual(abs(params[1]), self.width, places=digits)
        self.assertAlmostEqual(params[2], self.amp, places=digits)
        self.assertAlmostEqual(params[3], self.back, places=digits)
        self.assertAlmostEqual(params[4], self.eta, places=digits)
        self.assertAlmostEqual(params[5], self.sl, places=digits)


if __name__ == '__main__':
    unittest.main()
