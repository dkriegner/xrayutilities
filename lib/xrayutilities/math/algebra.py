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
# Copyright (C) 2016-2021 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module providing analytic algebraic functions not implemented in scipy or any
other dependency of xrayutilities. In particular the analytic solution of a
quartic equation which is needed for the solution of the dynamic scattering
equations.
"""

import numpy


def solve_quartic(a4, a3, a2, a1, a0):
    """
    analytic solution [1]_ of the general quartic equation. The solved equation
    takes the form :math:`a_4 z^4 + a_3 z^3 + a_2 z^2 + a_1 z + a_0`

    Returns
    -------
    tuple
        tuple of the four (complex) solutions of aboves equation.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/QuarticEquation.html
    """
    a4 = numpy.asarray(a4)
    a3 = numpy.asarray(a3)
    a2 = numpy.asarray(a2)
    a1 = numpy.asarray(a1)
    a0 = numpy.asarray(a0)

    t01 = 1. / a4
    t02 = a3**2
    t04 = t01 * t01
    t05 = t04 * t01
    t09 = a1 * t01
    t10 = (a3 * t02 * t05) / 8
    t11 = (a3 * a2 * t04) / 2
    t03 = t09 + t10 - t11
    t06 = a2 * t01
    t19 = (3 * t02 * t04) / 8
    t07 = t06 - t19
    t08 = t07**2
    t12 = t03**2
    t13 = a0 * t01
    t14 = t05 * t01
    t15 = t02**2
    t16 = (a2 * t02 * t05) / 16
    t20 = (3 * t14 * t15) / 256
    t21 = (a3 * a1 * t04) / 4
    t17 = t13 + t16 - t20 - t21
    t18 = t17**2
    t22 = t12 / 2.
    t23 = (t07 * t08) / 27
    t24 = 3.**(1./2)
    t25 = t12**2
    t26 = 27 * t25
    t27 = t08**2
    t28 = 4 * t07 * t08 * t12
    t29 = 128 * t08 * t18
    t35 = 256 * t17 * t18
    t36 = 16 * t17 * t27
    t37 = 144 * t07 * t12 * t17
    t30 = t26 + t28 + t29 - t35 - t36 - t37
    t31 = t30.astype(complex)**(1./2)
    t32 = (t24 * t31) / 18
    t34 = (4 * t07 * t17) / 3
    t33 = t22 + t23 + t32 - t34
    t38 = 1 / t33.astype(complex)**(1./6)
    t39 = t33**(1./3)
    t40 = 12 * a0 * t01
    t41 = t33**(2./3)
    t42 = 9 * t41
    t43 = (3 * a2 * t02 * t05) / 4
    t46 = 6 * t07 * t39
    t47 = (9 * t14 * t15) / 64
    t48 = 3 * a3 * a1 * t04
    t44 = t08 + t40 + t42 + t43 - t46 - t47 - t48
    t45 = t44.astype(complex)**(1./2)
    t49 = 6.**(1./2)
    t50 = 27 * t12
    t51 = 2 * t07 * t08
    t52 = 3 * t24 * t31
    t62 = 72 * t07 * t17
    t53 = t50 + t51 + t52 - t62
    t54 = t53.astype(complex)**(1./2)
    t55 = 3 * t03 * t49 * t54
    t59 = t08 * t45
    t60 = 12 * t17 * t45
    t61 = 9 * t41 * t45
    t63 = 12 * t07 * t39 * t45
    t56 = t55 - t59 - t60 - t61 - t63
    t57 = t56.astype(complex)**(1./2)
    t58 = 1. / t44.astype(complex)**(1./4)
    t64 = (t38 * t45) / 6
    t65 = - t55 - t59 - t60 - t61 - t63
    t66 = t65.astype(complex)**(1./2)

    return (-(a3 * t01) / 4 - (t38 * t45) / 6 - (t38 * t57 * t58) / 6,
            (t38 * t57 * t58) / 6 - (t38 * t45) / 6 - (a3 * t01) / 4,
            t64 - (a3 * t01) / 4 - (t38 * t58 * t66) / 6,
            t64 - (a3 * t01) / 4 + (t38 * t58 * t66) / 6)
