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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities utilities contains a conglomeration of useful functions
this part of utilities does not need the config class
"""

import numbers
import numpy
import scipy.constants
import os

from .exception import InputError

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str

energies = {
    'CuKa1': 8047.82310,
    'CuKa2': 8027.9117,
    'CuKa12': 8041.18,
    'CuKb': 8905.337,
    'MoKa1': 17479.374,
    'CoKa1': 6930.32,
    'CoKa2': 6915.30}
# wavelength values from International Tables of Crystallography:
# Vol C, 2nd Ed. page 203
# CuKa1: 1.54059292(45) the value in bracket is the uncertainty
# CuKa2: 1.5444140(19)
# CuKa12: mixture 2:1 a1 and a2
# CuKb:  1.392246(14)
# MoKa1: 0.70931713(41)
# Xray data booklet:
# CoKa1,2 energies

def lam2en(inp):
    """
    converts the input wavelength in Angstrom to an energy in eV

    Parameter
    ---------
     inp : wavelength in Angstrom

    Returns
    -------
     float, energy in eV

    Examples
    --------
     >>> energy = lam2en(1.5406)
    """
    #  E(eV) = h*c/(e * lambda(A)) *1e10
    inp = wavelength(inp)
    c = scipy.constants
    out = c.h * c.speed_of_light / (c.e * inp) * 1e10
    return out


def en2lam(inp):
    """
    converts the input energy in eV to a wavelength in Angstrom

    Parameter
    ---------
     inp : energy in eV

    Returns
    -------
     float, wavlength in Angstrom

    Examples
    --------
     >>> wavelength = en2lam(8048)
    """
    #  lambda(A) = h*c/(e * E(eV)) *1e10
    inp = energy(inp)
    c = scipy.constants
    out = c.h * c.speed_of_light / (c.e * inp) * 1e10
    return out


def energy(en):
    """
    convert common energy names to energies in eV

    so far this works with CuKa1, CuKa2, CuKa12, CuKb, MoKa1

    Parameter
    ---------
     en: energy either as scalar or array with value in eV, which
         will be returned unchanged; or string with name of emission line

    Returns
    -------
     energy in eV as float
    """

    if isinstance(en, numbers.Number):
        return numpy.double(en)
    elif isinstance(en, (numpy.ndarray, list, tuple)):
        return numpy.array(en)
    elif isinstance(en, basestring):
        return energies[en]
    else:
        raise InputError("wrong type for argument en")


def wavelength(wl):
    """
    convert common energy names to energies in eV

    so far this works with CuKa1, CuKa2, CuKa12, CuKb, MoKa1

    Parameter
    ---------
     wl: wavelength; If scalar or array the wavelength in Angstrom will be
         returned unchanged, string with emission name is converted to
         wavelength

    Returns
    -------
     wavelength in Angstrom as float

    """

    if isinstance(wl, numbers.Number):
        return numpy.double(wl)
    elif isinstance(wl, (numpy.ndarray, list, tuple)):
        return numpy.array(wl)
    elif isinstance(wl, basestring):
        return en2lam(energies[wl])
    else:
        raise InputError("wrong type for argument wavelength")


def exchange_path(orig, new, keep=0):
    """
    function to exchange the root of a path with the option of keeping the
    inner directory structure. This for example includes such a conversion
    /dir_a/subdir/images/sample -> /home/user/data/images/sample
    where the two innermost directory names are kept (keep=2)

    Parameter
    ---------
     orig:  original path which should be replaced by the new path
     new:   new path which should be used instead
     keep:  (optional) number of inner most directory names which should be
            kept the same in the output (default = 0)

    Returns
    -------
     directory path string

    Examples
    --------
    >>> exchange_path('/dir_a/subdir/img/sam', '/home/user/data', 2)
    '/home/user/data/img/sam'
    """
    subdirs = []
    o = orig
    for i in range(keep):
        o, s = os.path.split(o)
        subdirs.append(s)
    out = new
    subdirs.reverse()
    for s in subdirs:
        out = os.path.join(out, s)
    return out


def exchange_filepath(orig, new, keep=0):
    """
    function to exchange the root of a filename with the option of keeping the
    inner directory structure. This for example includes such a conversion
    /dir_a/subdir/sample/file.txt -> /home/user/data/sample/file.txt
    where the innermost directory name is kept (keep=1)

    Parameter
    ---------
     orig:  original filename which should have its data root replaced
     new:   new path which should be used instead
     keep:  (optional) number of inner most directory names which should be
            kept the same in the output (default = 0). Note that the filename
            is always return unchanged also with keep=0.

    Returns
    -------
     filename string

    Examples
    --------
    >>> exchange_filepath('/dir_a/subdir/sam/file.txt', '/data', 1)
    '/data/sam/file.txt'
    """
    if new:
        return exchange_path(orig, new, keep+1)
    else:
        return orig
