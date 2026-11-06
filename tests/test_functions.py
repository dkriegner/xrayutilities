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
# Copyright (C) 2026 Davis Rens <166565828+davrens101@users.noreply.github.com>

import copy
import unittest

import numpy
import xrayutilities as xu
from scipy.integrate import nquad, quad


class TestMathFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        amp = numpy.random.rand() + 0.1
        fwhm = numpy.random.rand() * 1.5 + 0.1
        cls.x = numpy.arange(-3, 3, 0.0003)
        cls.p = [0.0, fwhm, amp, 0.0]
        cls.p2d = [
            0.0,
            0.0,
            fwhm,
            fwhm,
            amp,
            0.0,
            2 * numpy.pi * numpy.random.rand(),
        ]
        cls.sigma = fwhm / (2 * numpy.sqrt(2 * numpy.log(2)))

    def test_gauss1dwidth(self):
        p = numpy.copy(self.p)
        p[1] = self.sigma
        f = xu.math.Gauss1d(self.x, *p)
        fwhm = xu.math.fwhm_exp(self.x, f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_lorentz1dwidth(self):
        p = numpy.copy(self.p)
        f = xu.math.Lorentz1d(self.x, *p)
        fwhm = xu.math.fwhm_exp(self.x, f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_pvoigt1dwidth(self):
        p = list(numpy.copy(self.p))
        p += [
            numpy.random.rand(),
        ]
        f = xu.math.PseudoVoigt1d(self.x, *p)
        fwhm = xu.math.fwhm_exp(self.x, f)
        self.assertAlmostEqual(fwhm, self.p[1], places=4)

    def test_normedgauss1d(self):
        p = numpy.copy(self.p)
        p[1] = self.sigma
        f = xu.math.Gauss1d(self.x, *p) / xu.math.Gauss1dArea(*p)
        norm = xu.math.NormGauss1d(self.x, p[0], p[1])
        self.assertAlmostEqual(numpy.sum(numpy.abs(f - norm)), 0, places=6)

    def test_gauss1darea(self):
        p = numpy.copy(self.p)
        p[1] = self.sigma
        area = xu.math.Gauss1dArea(*p)
        (numarea, err) = quad(
            xu.math.Gauss1d, -numpy.inf, numpy.inf, args=tuple(p)
        )
        digits = int(numpy.abs(numpy.log10(err))) - 3
        self.assertTrue(digits >= 3)
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_lorentz1darea(self):
        p = numpy.copy(self.p)
        area = xu.math.Lorentz1dArea(*p)
        (numarea, err) = quad(
            xu.math.Lorentz1d, -numpy.inf, numpy.inf, args=tuple(p)
        )
        digits = int(numpy.abs(numpy.log10(err))) - 3
        self.assertTrue(digits >= 3)
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_pvoigt1darea(self):
        p = list(numpy.copy(self.p))
        p += [
            numpy.random.rand(),
        ]
        area = xu.math.PseudoVoigt1dArea(*p)
        (numarea, err) = quad(
            xu.math.PseudoVoigt1d, -numpy.inf, numpy.inf, args=tuple(p)
        )
        digits = int(numpy.abs(numpy.log10(err))) - 3
        self.assertTrue(digits >= 3)
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_gauss2darea(self):
        p = numpy.copy(self.p2d)
        p[2] = self.sigma
        p[3] = (numpy.random.rand() + 0.1) * self.sigma
        area = xu.math.Gauss2dArea(*p)
        (numarea, err) = nquad(
            xu.math.Gauss2d,
            [[-numpy.inf, numpy.inf], [-numpy.inf, numpy.inf]],
            args=tuple(p),
        )
        digits = int(numpy.abs(numpy.log10(err))) - 3
        self.assertTrue(digits >= 3)
        self.assertAlmostEqual(area, numarea, places=digits)

    def test_twogauss2d_shared_background(self):
        x, y = numpy.meshgrid(
            numpy.linspace(-4, 4, 13), numpy.linspace(-4, 4, 11)
        )
        p = [
            1,
            -0.5,
            0.6,
            1.2,
            7,
            0.35,
            -1,
            1,
            0.9,
            0.4,
            4,
            -0.6,
            3.5,
        ]
        expected = numpy.full_like(x, p[12])
        for start in (0, 6):
            cx, cy, sx, sy, amp, angle = p[start : start + 6]
            c, s = numpy.cos(angle), numpy.sin(angle)
            rotation = numpy.array([[c, -s], [s, c]])
            precision = rotation.T @ numpy.diag([sx**-2, sy**-2]) @ rotation
            delta = numpy.stack((x - cx, y - cy), axis=-1)
            exponent = numpy.einsum(
                "...i,ij,...j->...", delta, precision, delta
            )
            expected += amp * numpy.exp(-0.5 * exponent)
        numpy.testing.assert_allclose(
            xu.math.TwoGauss2d(x, y, *p), expected, rtol=1e-13, atol=1e-13
        )
        self.assertEqual(xu.math.TwoGauss2d(100.0, 100.0, *p), p[12])

    def test_derivatives(self):
        eps = 1e-9
        # generate test input parameters (valid for Gauss, Lorentz, and Voigt)
        p = list(numpy.copy(self.p))
        p[1] = self.sigma
        p[3] = numpy.random.rand()
        p.append(numpy.random.rand())
        params = [
            self.x,
        ] + p

        functions = [
            (xu.math.Gauss1d, xu.math.Gauss1d_der_x, xu.math.Gauss1d_der_p),
            (
                xu.math.Lorentz1d,
                xu.math.Lorentz1d_der_x,
                xu.math.Lorentz1d_der_p,
            ),
            (
                xu.math.PseudoVoigt1d,
                xu.math.PseudoVoigt1d_der_x,
                xu.math.PseudoVoigt1d_der_p,
            ),
        ]

        # test all derivates by benchmarking against a simple finite difference
        # calculation
        for f, fdx, fdp in functions:
            deriv = numpy.vstack((fdx(*params), fdp(*params)))
            for argidx in range(len(deriv)):
                pmeps = copy.copy(params)
                ppeps = copy.copy(params)
                pmeps[argidx] = pmeps[argidx] - eps / 2
                ppeps[argidx] = ppeps[argidx] + eps / 2
                findiff = (f(*ppeps) - f(*pmeps)) / eps
                diff = numpy.abs(deriv[argidx] - findiff)
                errinfo = (
                    f"maximum difference, idx/npoints: "
                    f"{numpy.max(diff)}, {numpy.argmax(diff)}/{len(diff)}\n"
                    f"parameters: {p!s}"
                )
                self.assertTrue(
                    numpy.allclose(deriv[argidx], findiff, atol=1e3 * eps),
                    f"{f!s}, {argidx}, derivatives not close "
                    f"to numerical approximation ({errinfo})",
                )


if __name__ == "__main__":
    unittest.main()
