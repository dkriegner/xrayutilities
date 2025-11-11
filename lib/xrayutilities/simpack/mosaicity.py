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
# Copyright (C) 2019-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import math

import numpy


def mosaic_analytic(qx, qz, RL, RV, Delta, hx, hz, shape):
    """
    simulation of the coplanar reciprocal space map of a single mosaic layer
    using a simple analytic approximation

    Parameters
    ----------
    qx :    array-like
        vector of the qx values (offset from the Bragg peak)
    qz :    array-like
        vector of the qz values (offset from the Bragg peak)
    RL :    float
        lateral block radius in angstrom
    RV :    float
        vertical block radius in angstrom
    Delta : float
        root mean square misorientation of the grains in degree
    hx :    float
        lateral component of the diffraction vector
    hz :    float
        vertical component of the diffraction vector
    shape :  float
        shape factor (1..Gaussian)

    Returns
    -------
    array-like
    2D array with calculated intensities
    """
    QX, QZ = numpy.meshgrid(qx, qz)
    QX = QX.T
    QZ = QZ.T
    DD = numpy.radians(Delta)
    tmp = 6 + DD**2 * ((hz * RL) ** 2 + (hx * RV) ** 2)
    F = (
        (
            (DD * RL * RV) ** 2 * (QZ * hz + QX * hx) ** 2
            + 6 * ((RL * QX) ** 2 + (RV * QZ) ** 2)
        )
        / 4
        / tmp
    )
    return (
        math.pi
        * math.sqrt(6)
        * RL
        * RV
        / math.sqrt(tmp)
        * numpy.exp(-(F**shape))
    )
