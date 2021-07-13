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
# Copyright (C) 2010-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities utilities contains a conglomeration of useful functions
this part of utilities does not need the config class
"""

import abc
import numbers
import os.path
import re

import numpy
import scipy.constants

from .exception import InputError

try:  # works in Python >3.4
    ABC = abc.ABC
except AttributeError:  # Python 2.7
    ABC = abc.ABCMeta('ABC', (object, ), {'__slots__': ()})


__all__ = ['ABC', 'check_kwargs', 'clear_bit', 'en2lam', 'energies', 'energy',
           'exchange_filepath', 'exchange_path', 'is_valid_variable_name',
           'lam2en', 'makeNaturalName', 'set_bit', 'wavelength']

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
# CoKa1
# CoKa2


def set_bit(f, offset):
    """
    sets the bit at an offset
    """
    mask = 1 << offset
    return(f | mask)


def clear_bit(f, offset):
    """
    clears the bet at an offset
    """
    mask = ~(1 << offset)
    return(f & mask)


def lam2en(inp):
    """
    converts the input wavelength in angstrom to an energy in eV

    Parameters
    ----------
    inp :   float or str
        wavelength in angstrom

    Returns
    -------
    float
        energy in eV

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
    converts the input energy in eV to a wavelength in angstrom

    Parameters
    ----------
    inp :   float or str
        energy in eV

    Returns
    -------
    float
        wavlength in angstrom

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

    Parameters
    ----------
    en :    float, array-like or str
        energy either as scalar or array with value in eV, which will be
        returned unchanged; or string with name of emission line

    Returns
    -------
    float or array-like
        energy in eV
    """

    if isinstance(en, numbers.Number):
        return numpy.double(en)
    elif isinstance(en, (numpy.ndarray, list, tuple)):
        return numpy.asarray(en)
    elif isinstance(en, str):
        return energies[en]
    else:
        raise InputError("wrong type for argument en")


def wavelength(wl):
    """
    convert common energy names to energies in eV

    so far this works with CuKa1, CuKa2, CuKa12, CuKb, MoKa1

    Parameters
    ----------
    wl :    float, array-like or str
        wavelength; If scalar or array the wavelength in angstrom will be
        returned unchanged, string with emission name is converted to
        wavelength

    Returns
    -------
    float or array-like
        wavelength in angstrom
    """

    if isinstance(wl, numbers.Number):
        return numpy.double(wl)
    elif isinstance(wl, (numpy.ndarray, list, tuple)):
        return numpy.asarray(wl)
    elif isinstance(wl, str):
        return en2lam(energies[wl])
    else:
        raise InputError("wrong type for argument wavelength")


def exchange_path(orig, new, keep=0, replace=None):
    """
    function to exchange the root of a path with the option of keeping the
    inner directory structure. This for example includes such a conversion
    /dir_a/subdir/images/sample -> /home/user/data/images/sample
    where the two innermost directory names are kept (keep=2), or equally
    the three outer most are replaced (replace=3). One can either give keep,
    or replace, with replace taking preference if both are given. Note that
    replace=1 on Linux/Unix replaces only the root for absolute paths.

    Parameters
    ----------
    orig :      str
        original path which should be replaced by the new path
    new :       str
        new path which should be used instead
    keep :      int, optional
        number of inner most directory names which should be kept the same in
        the output (default = 0)
    replace :   int, optional
        number of outer most directory names which should be replaced in the
        output (default = None)

    Returns
    -------
    str
        directory path string

    Examples
    --------
    >>> exchange_path('/dir_a/subdir/img/sam', '/home/user/data', keep=2)
    '/home/user/data/img/sam'
    """
    subdirs = []
    o = orig
    if replace is None:
        for i in range(keep):
            o, s = os.path.split(o)
            subdirs.append(s)
        out = new
        subdirs.reverse()
        for s in subdirs:
            out = os.path.join(out, s)
    else:
        while True:
            o, s = os.path.split(o)
            if not s:
                subdirs.append(o)
                break
            elif not o:
                subdirs.append(s)
                break
            else:
                subdirs.append(s)
        subdirs.reverse()
        out = new
        for s in subdirs[replace:]:
            out = os.path.join(out, s)
    return out


def exchange_filepath(orig, new, keep=0, replace=None):
    """
    function to exchange the root of a filename with the option of keeping the
    inner directory structure. This for example includes such a conversion
    /dir_a/subdir/sample/file.txt -> /home/user/data/sample/file.txt
    where the innermost directory name is kept (keep=1), or equally
    the three outer most are replaced (replace=3). One can either give keep,
    or replace, with replace taking preference if both are given. Note that
    replace=1 on Linux/Unix replaces only the root for absolute paths.

    Parameters
    ----------
    orig :      str
        original filename which should have its data root replaced
    new :       str
        new path which should be used instead
    keep :      int, optional
        number of inner most directory names which should be kept the same in
        the output (default = 0)
    replace :   int, optional
        number of outer most directory names which should be replaced in the
        output (default = None)

    Returns
    -------
    str
        filename string

    Examples
    --------
    >>> exchange_filepath('/dir_a/subdir/sam/file.txt', '/data', 1)
    '/data/sam/file.txt'
    """
    if new:
        if replace is None:
            return exchange_path(orig, new, keep+1)
        else:
            return exchange_path(orig, new, replace=replace)
    else:
        return orig


def makeNaturalName(name, check=False):
    ret = re.sub('[^0-9a-zA-Z]', '_', name.strip())
    isvalid = is_valid_variable_name(ret)
    if not check or isvalid:
        return ret
    elif not isvalid:
        raise ValueError("'{}' is not valid variable name".format(ret))


def is_valid_variable_name(name):
    return name.isidentifier()


def check_kwargs(kwargs, valid_kwargs, identifier):
    """
    Raises an TypeError if kwargs included a key which is not in valid_kwargs.

    Parameters
    ----------
    kwargs :        dict
        keyword arguments dictionary
    valid_kwargs :  dict
        dictionary with valid keyword arguments and their description
    identifier :    str
        string to identifier the caller of this function
    """
    desc = ', '.join(["'%s': %s" % (k, d) for k, d in valid_kwargs.items()])
    for k in kwargs:
        if k not in valid_kwargs:
            raise TypeError("%s: unknown keyword argument ('%s') given; "
                            "allowed are %s" % (identifier, k, desc))
