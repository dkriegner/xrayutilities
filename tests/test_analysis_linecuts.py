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
# Copyright (C) 2018 Dominik Kriegner <dominik.kriegner@gmail.com>

import math
import unittest

import numpy
import xrayutilities as xu


class Test_analysis_linecuts(unittest.TestCase):
    exp = xu.HXRD([1, 0, 0], [0, 0, 1])
    qmax = (2 * exp.k0) / math.sqrt(2) - 0.1
    qyp, qzp = numpy.sort(numpy.random.rand(2) * (qmax-2) + 2)
    width1 = 0.015
    width2 = 0.002
    qy = numpy.linspace(qyp-0.1, qyp+0.1, 601)
    qz = numpy.linspace(qzp-0.1, qzp+0.1, 617)
    Ncut = 450
    QY, QZ = numpy.meshgrid(qy, qz)
    omp, _, _, ttp = exp.Q2Ang(0, qyp, qzp, trans=False)

    def test_radial_cut(self):
        omegaang = math.radians(self.omp-self.ttp/2.)
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, omegaang)
        x, d, m = xu.analysis.get_radial_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='2theta')
        x2, d2, m2 = xu.analysis.get_radial_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='omega')
        self.assertEqual(x.size, self.Ncut)
        self.assertTrue(xu.math.fwhm_exp(x, d) > xu.math.fwhm_exp(x2, d2))

    def test_omega_cut(self):
        radialang = math.radians((self.omp-self.ttp/2.) - 90)
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, radialang)
        x, d, m = xu.analysis.get_omega_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='2theta')
        x2, d2, m2 = xu.analysis.get_omega_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='radial')
        self.assertEqual(x.size, self.Ncut)
        self.assertTrue(xu.math.fwhm_exp(x, d) > xu.math.fwhm_exp(x2, d2))

    def test_ttheta_cut(self):
        omegaang = math.radians(self.omp-self.ttp/2.)
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, omegaang)
        x, d, m = xu.analysis.get_ttheta_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='radial')
        x2, d2, m2 = xu.analysis.get_ttheta_scan(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), self.Ncut, 1,
            intdir='omega')
        self.assertEqual(x.size, self.Ncut)
        self.assertTrue(xu.math.fwhm_exp(x, d) > xu.math.fwhm_exp(x2, d2))

    def test_qz_cut(self):
        omegaang = math.radians(self.omp-self.ttp/2.)
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, omegaang)
        x, d, m = xu.analysis.get_qz_scan(
            [self.QY, self.QZ], data, self.qyp, self.Ncut, 1, intdir='2theta')
        x2, d2, m2 = xu.analysis.get_qz_scan(
            [self.QY, self.QZ], data, self.qyp, self.Ncut, 1, intdir='omega')
        self.assertEqual(x.size, self.Ncut)
        self.assertTrue(xu.math.fwhm_exp(x, d) > xu.math.fwhm_exp(x2, d2))

    def test_qy_cut(self):
        omegaang = math.radians(self.omp-self.ttp/2.)
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, omegaang)
        x, d, m = xu.analysis.get_qy_scan(
            [self.QY, self.QZ], data, self.qzp, self.Ncut, 1, intdir='2theta')
        x2, d2, m2 = xu.analysis.get_qy_scan(
            [self.QY, self.QZ], data, self.qzp, self.Ncut, 1, intdir='omega')
        self.assertTrue(xu.math.fwhm_exp(x, d) > xu.math.fwhm_exp(x2, d2))

    def test_qcut_width(self):
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, 0)
        x, d, m = xu.analysis.get_qz_scan(
            [self.QY, self.QZ], data, self.qyp, self.Ncut, 0.1,
            intdir='q')
        self.assertEqual(x.size, self.Ncut)
        self.assertAlmostEqual(
            xu.math.fwhm_exp(x, d) / (2*math.sqrt(2*math.log(2))),
            self.width2, places=4)
        x, d, m = xu.analysis.get_qy_scan(
            [self.QY, self.QZ], data, self.qzp, self.Ncut, 0.1,
            intdir='q')
        self.assertAlmostEqual(
            xu.math.fwhm_exp(x, d) / (2*math.sqrt(2*math.log(2))),
            self.width1, places=4)

    def test_arbitrary_cut(self):
        data = xu.math.Gauss2d(self.QY, self.QZ, self.qyp, self.qzp,
                               self.width1, self.width2, 1, 0, 0)
        x, d, m = xu.analysis.get_arbitrary_line(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), (1, 0), self.Ncut,
            0.01)
        x2, d2, m2 = xu.analysis.get_arbitrary_line(
            [self.QY, self.QZ], data, (self.qyp, self.qzp), (0, 1), self.Ncut,
            0.01)

        self.assertEqual(x.size, self.Ncut)
        self.assertAlmostEqual(x[numpy.argmax(d)], self.qyp, places=2)
        self.assertAlmostEqual(x2[numpy.argmax(d2)], self.qzp, places=2)
        self.assertAlmostEqual(
            xu.math.fwhm_exp(x, d) / (2*math.sqrt(2*math.log(2))),
            self.width1, places=4)
        self.assertAlmostEqual(
            xu.math.fwhm_exp(x2, d2) / (2*math.sqrt(2*math.log(2))),
            self.width2, places=4)


if __name__ == '__main__':
    unittest.main()
