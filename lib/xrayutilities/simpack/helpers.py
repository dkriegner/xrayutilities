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

import numpy

from .. import config, utilities


def coplanar_alphai(qx, qz, en="config", nan_Laue=False):
    """
    calculate coplanar incidence angle from knowledge of the qx (inplane) and
    qz (out of plane) coordinates

    Parameters
    ----------
    qx :    array-like
        inplane momentum transfer component
    qz :    array-like
        out of plane momentum transfer component
    en :    float or str, optional
        x-ray energy (eV). By default the value from the config is used.
    nan_Laue : bool, optional
        set output points inside the Laue zone to 'nan'. default: False

    Returns
    -------
    alphai : array-like
        the incidence angle in degree
    """
    if isinstance(en, str) and en == "config":
        en = utilities.energy(config.ENERGY)
    k = 2 * numpy.pi / utilities.en2lam(en)
    th = numpy.arcsin(numpy.sqrt(qx**2 + qz**2) / (2 * k))
    ai = numpy.arctan2(qx, qz) + th
    if nan_Laue:
        if isinstance(ai, numpy.ndarray):  # remove positions in Laue zone
            ai[qz < numpy.sqrt(2 * qx * k - qx**2)] = numpy.nan
        else:
            if qz < numpy.sqrt(2 * qx * k - qx**2):
                ai = numpy.nan
    return numpy.degrees(ai)


def coplanar_alphaf(qx, qz, en="config", nan_Laue=False):
    """
    calculate coplanar exit angle from knowledge of the qx (inplane) and qz
    (out of plane) coordinates

    Parameters
    ----------
    qx :    array-like
        inplane momentum transfer component
    qz :    array-like
        out of plane momentum transfer component
    en :    float or str, optional
        x-ray energy (eV). By default the value from the config is used.
    nan_Laue : bool, optional
        set output points inside the Laue zone to 'nan'. default: False

    Returns
    -------
    alphaf : array-like
        the exit angle in degree. points in the Laue zone are set to
        'nan'.
    """
    if isinstance(en, str) and en == "config":
        en = utilities.energy(config.ENERGY)
    k = 2 * numpy.pi / utilities.en2lam(en)
    th = numpy.arcsin(numpy.sqrt(qx**2 + qz**2) / (2 * k))
    af = th - numpy.arctan2(qx, qz)
    if nan_Laue:
        if isinstance(af, numpy.ndarray):  # remove positions in Laue zone
            af[qz < numpy.sqrt(2 * qx * k - qx**2)] = numpy.nan
        else:
            if qz < numpy.sqrt(2 * qx * k - qx**2):
                af = numpy.nan
    return numpy.degrees(af)


def get_qz(qx, alphai, en="config"):
    """
    calculate the qz position from the qx position and the incidence angle for
    a coplanar diffraction geometry

    Parameters
    ----------
    qx :        array-like
        inplane momentum transfer component
    alphai :    array-like
        incidence angle (deg)
    en :        float or str, optional
        x-ray energy (eV). By default the value from the config is used.

    Returns
    -------
    array-like
        the qz position for the given incidence angle. The output is nan in
        case the inplane momentum transfer can not be reached.
    """
    if isinstance(en, str) and en == "config":
        en = utilities.energy(config.ENERGY)
    k = 2 * numpy.pi / utilities.en2lam(en)
    ai = numpy.radians(alphai)
    qz = numpy.sqrt(k**2 - (qx + k * numpy.cos(ai)) ** 2) + k * numpy.sin(ai)
    return qz
