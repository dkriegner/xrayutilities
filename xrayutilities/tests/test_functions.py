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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy
import scipy.integrate
import unittest

class TestMathFunctions(unittest.TestCase):

    def setUp(self):
        amp = numpy.random.rand(1)[0]
        fwhm = numpy.random.rand(1)[0]*1.5
        self.x = numpy.arange(-3,3,0.0003)
        self.p = [0.,fwhm,amp,0.]
        self.p2d = [0.,0.,fwhm,fwhm,amp,0.,2*numpy.pi*numpy.random.rand(1)]
        self.sigma = fwhm/(2*numpy.sqrt(2*numpy.log(2)))

    def test_gauss1dwidth(self):
        p = numpy.copy(self.p)
        p[1] = self.sigma
        f = xu.math.Gauss1d(self.x,*p)
        fwhm = xu.analysis.fwhm_exp(self.x,f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_lorentz1dwidth(self):
        p = numpy.copy(self.p)
        f = xu.math.Lorentz1d(self.x,*p)
        fwhm = xu.analysis.fwhm_exp(self.x,f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_pvoigt1dwidth(self):
        p = list(numpy.copy(self.p))
        p += [numpy.random.rand(1),]
        f = xu.math.PseudoVoigt1d(self.x,*p)
        fwhm = xu.analysis.fwhm_exp(self.x,f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_gauss1darea(self):
        p = numpy.copy(self.p)
        p[1] = self.sigma
        area = xu.math.Gauss1dArea(*p)
        (numarea,err) = scipy.integrate.quad(xu.math.Gauss1d,-numpy.inf,numpy.inf,args=tuple(p))
        digits = int(numpy.abs(numpy.log10(err)))-1
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_lorentz1darea(self):
        p = numpy.copy(self.p)
        area = xu.math.Lorentz1dArea(*p)
        (numarea,err) = scipy.integrate.quad(xu.math.Lorentz1d,-numpy.inf,numpy.inf,args=tuple(p))
        digits = int(numpy.abs(numpy.log10(err)))-1
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_pvoigt1darea(self):
        p = list(numpy.copy(self.p))
        p += [numpy.random.rand(1)[0],]
        area = xu.math.PseudoVoigt1dArea(*p)
        (numarea,err) = scipy.integrate.quad(xu.math.PseudoVoigt1d,-numpy.inf,numpy.inf,args=tuple(p))
        digits = int(numpy.abs(numpy.log10(err)))-1
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_gauss2darea(self):
        p = numpy.copy(self.p2d)
        p[2] = self.sigma; p[3] = numpy.random.rand(1)*self.sigma
        area = xu.math.Gauss2dArea(*p)
        (numarea,err) = scipy.integrate.dblquad(xu.math.Gauss2d,-numpy.inf,numpy.inf,lambda x:-numpy.inf,lambda x: numpy.inf,args=tuple(p))
        digits = int(numpy.abs(numpy.log10(err)))-1
        self.assertAlmostEqual(area, numarea, places=digits)

if __name__ == '__main__':
        unittest.main()
