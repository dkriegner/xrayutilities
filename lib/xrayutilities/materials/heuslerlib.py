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

"""
implement convenience functions to define Heusler materials.
"""

from . import elements
from .material import Crystal
from .spacegrouplattice import SGLattice

__all__ = ['FullHeuslerCubic225', 'FullHeuslerCubic225_A2',
           'FullHeuslerCubic225_B2', 'FullHeuslerCubic225_DO3',
           'HeuslerHexagonal194', 'HeuslerTetragonal119',
           'HeuslerTetragonal139', 'InverseHeuslerCubic216']


def _check_elements(*elem):
    ret = []
    for el in elem:
        if isinstance(el, str):
            ret.append(getattr(elements, el))
        else:
            ret.append(el)
    return ret


def FullHeuslerCubic225(X, Y, Z, a, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Full Heusler structure with formula X2YZ.
    Strukturberichte symbol L2_1; space group Fm-3m (225)

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a :         float
        cubic lattice parameter in angstrom
    biso :      list of floats, optional
        Debye Waller factors for X, Y, Z elements
    occ :       list of floats, optional
        occupation numbers for the elements X, Y, Z

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(225, a, atoms=[x, y, z], pos=['8c', '4a', '4b'],
                             b=biso, occ=occ))


def FullHeuslerCubic225_B2(X, Y, Z, a, b2dis, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Full Heusler structure with formula X2YZ.
    Strukturberichte symbol L2_1; space group Fm-3m (225) with B2-type (CsCl)
    disorder

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a :         float
        cubic lattice parameter in angstrom
    b2dis :     float
        amount of B2-type disorder (0: fully ordered, 1: fully disordered)
    biso :      list of floats, optional
        Debye Waller factors for X, Y, Z elements
    occ :       list of floats, optional
        occupation numbers for the elements X, Y, Z

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(225, a,
                             atoms=[x, y, z, y, z],
                             pos=['8c', '4a', '4b', '4b', '4a'],
                             occ=[1*occ[0], (1-b2dis/2.)*occ[1],
                                  (1-b2dis/2.)*occ[2], b2dis/2.*occ[1],
                                  b2dis/2.*occ[2]],
                             b=biso + [biso[1], biso[2]]))


def FullHeuslerCubic225_A2(X, Y, Z, a, a2dis, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Full Heusler structure with formula X2YZ.
    Strukturberichte symbol L2_1; space group Fm-3m (225) with A2-type (W)
    disorder

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a :         float
        cubic lattice parameter in angstrom
    a2dis :     float
        amount of A2-type disorder (0: fully ordered, 1: fully disordered)
    biso :      list of floats, optional
        Debye Waller factors for X, Y, Z elements
    occ :       list of floats, optional
        occupation numbers for the elements X, Y, Z

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(225, a,
                             atoms=[x, x, x, y, y, y, z, z, z],
                             pos=['8c', '4a', '4b',
                                  '8c', '4a', '4b',
                                  '8c', '4a', '4b'],
                             occ=[(1-a2dis/2.)*occ[0], a2dis/2.*occ[0],
                                  a2dis/2.*occ[0], a2dis/4.*occ[1],
                                  (1-a2dis*3./4.)*occ[1], a2dis/4.*occ[1],
                                  a2dis/4.*occ[2], a2dis/4.*occ[2],
                                  (1-a2dis*3./4.)*occ[2]],
                             b=[biso[0], ]*3 + [biso[1], ]*3 + [biso[2], ]*3))


def FullHeuslerCubic225_DO3(X, Y, Z, a, do3disxy, do3disxz, biso=[0, 0, 0],
                            occ=[1, 1, 1]):
    """
    Full Heusler structure with formula X2YZ.
    Strukturberichte symbol L2_1; space group Fm-3m (225) with DO_3-type (BiF3)
    disorder, either between atoms X <-> Y or X <-> Z.

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a :         float
        cubic lattice parameter in angstrom
    do3disxy :  float
        amount of DO_3-type disorder between X and Y atoms (0: fully ordered,
        1: fully disordered)
    do3disxz :  float
        amount of DO_3-type disorder between X and Z atoms (0: fully ordered,
        1: fully disordered)
    biso :      list of floats, optional
        Debye Waller factors for X, Y, Z elements
    occ :       list of floats, optional
        occupation numbers for the elements X, Y, Z

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(225, a,
                             atoms=[x, y, z,
                                    x, y,
                                    x, z],
                             pos=['8c', '4a', '4b',
                                  '4a', '8c',
                                  '4b', '8c'],
                             occ=[(1-do3disxy/3.-do3disxz/3.)*occ[0],
                                  (1-do3disxy*2/3.)*occ[1],
                                  (1-do3disxz*2/3.)*occ[2],
                                  do3disxy*2/3.*occ[0], do3disxy*1/3.*occ[1],
                                  do3disxz*2/3.*occ[0], do3disxz*1/3.*occ[2]],
                             b=biso + [biso[0], biso[1]] + [biso[0], biso[2]]))


def InverseHeuslerCubic216(X, Y, Z, a, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Full Heusler structure with formula (XY)X'Z structure;
    space group F-43m (216)

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a :         float
        cubic lattice parameter in angstrom

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('(%s%s)%s\'%s' % (x.basename, y.basename,
                                     x.basename, z.basename),
                   SGLattice(216, a, atoms=[x, x, y, z],
                             pos=['4a', '4d', '4b', '4c'],
                             b=[biso[0], ] + biso,
                             occ=[occ[0], ] + occ))


def HeuslerTetragonal139(X, Y, Z, a, c, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Tetragonal Heusler structure with formula X2YZ
    space group I4/mmm (139)

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a, c :      float
        tetragonal lattice parameters in angstrom

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(139, a, c,
                             atoms=[x, y, z],
                             pos=['4d', '2b', '2a'],
                             b=biso, occ=occ))


def HeuslerTetragonal119(X, Y, Z, a, c, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Tetragonal Heusler structure with formula X2YZ
    space group I-4m2 (119)

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a, c :      float
        tetragonal lattice parameters in angstrom

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s2%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(119, a, c,
                             atoms=[x, x, y, z],
                             pos=['2b', '2c', '2d', '2a'],
                             b=[biso[0], ] + biso,
                             occ=[occ[0], ] + occ))


def HeuslerHexagonal194(X, Y, Z, a, c, biso=[0, 0, 0], occ=[1, 1, 1]):
    """
    Hexagonal Heusler structure with formula XYZ
    space group P63/mmc (194)

    Parameters
    ----------
    X, Y, Z :   str or Element
        elements
    a, c :      float
        hexagonal lattice parameters in angstrom

    Returns
    -------
    Crystal
        Crystal describing the Heusler material
    """
    x, y, z = _check_elements(X, Y, Z)
    return Crystal('%s%s%s' % (x.basename, y.basename, z.basename),
                   SGLattice(194, a, c,
                             atoms=[x, y, z],
                             pos=['2a', '2c', '2d'],
                             b=biso, occ=occ))
