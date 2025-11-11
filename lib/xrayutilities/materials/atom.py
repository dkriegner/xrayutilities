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
# Copyright (c) 2015-2025 Dominik Kriegner <dominik.kriegner@gmail.com>

"""module handling physical properties of elements.

The Atom class manages the database access for atomic scattering factors and
the atomic mass.
"""

import hashlib
import re
from importlib.resources import files

import numpy

from .. import config, utilities
from . import database

dbfile = files("xrayutilities.materials") / "data" / config.DBNAME
_db = database.DataBase(dbfile)
_db.Open()


def get_key(*args):
    """Generate a hash key for several possible types of arguments"""
    tup = []
    for a in args:
        if isinstance(a, numpy.ndarray):
            tup.append(hashlib.md5(a).digest())
        elif isinstance(a, list):
            tup.append(hash(tuple(a)))
        else:
            tup.append(hash(a))
    return hash(tuple(tup))


class Atom:
    max_cache_length = 1000

    def __init__(self, name, num):
        self.name = name
        self.ostate = re.sub("[A-Za-z]", "", name)
        for r, o in zip(("dot", "p", "m"), (".", "+", "-")):
            self.ostate = self.ostate.replace(o, r)

        self.basename = re.sub("[^A-Za-z]", "", name)
        self.num = num
        self.__weight = None
        self.__color = None
        self.__radius = numpy.nan
        self._dbcache = {prop: [] for prop in ("f0", "f1", "f2", "f")}

    def __key__(self):
        """Key function to return the elements number"""
        return self.num

    def __lt__(self, other_el):
        """Make elements sortable by their key"""
        return self.__key__() < other_el.__key__()

    @property
    def weight(self):
        if not self.__weight:
            _db.SetMaterial(self.basename)
            self.__weight = _db.weight
        return self.__weight

    @property
    def color(self):
        if self.__color is None:
            _db.SetMaterial(self.basename)
            self.__color = _db.color
        return self.__color

    @property
    def radius(self):
        if self.__radius is numpy.nan:
            _db.SetMaterial(self.basename)
            self.__radius = _db.radius
        return self.__radius

    def get_cache(self, prop, key):
        """Check cached value to speed up repeated database requests.

        Returns
        -------
        bool
            True then result contains the cached otherwise False and result is
            None
        result :    database value

        """
        history = self._dbcache[prop]
        for idx, (k, result) in enumerate(history):
            if k == key:
                history.insert(0, history.pop(idx))  # move to front
                return True, result
        return False, None

    def set_cache(self, prop, key, result):
        """Set result to be cached to speed up future calls"""
        history = self._dbcache[prop]
        if len(history) == self.max_cache_length:
            history.pop(-1)
        history.insert(0, (key, result))

    def f0(self, q):
        key = get_key(q)
        f, res = self.get_cache("f0", key)
        if f:
            return res
        _db.SetMaterial(self.basename)
        res = _db.GetF0(q, self.ostate)
        self.set_cache("f0", key, res)
        return res

    def f1(self, en="config"):
        key = get_key(en)
        f, res = self.get_cache("f1", key)
        if f:
            return res
        if isinstance(en, str) and en == "config":
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        res = _db.GetF1(utilities.energy(en))
        self.set_cache("f1", key, res)
        return res

    def f2(self, en="config"):
        key = get_key(en)
        f, res = self.get_cache("f2", key)
        if f:
            return res
        if isinstance(en, str) and en == "config":
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        res = _db.GetF2(utilities.energy(en))
        self.set_cache("f2", key, res)
        return res

    def f(self, q, en="config"):
        """Function to calculate the atomic structure factor F

        Parameters
        ----------
        q :     float, array-like
            momentum transfer
        en :    float or str, optional
            energy for which F should be calculated, if omitted the value from
            the xrayutilities configuration is used

        Returns
        -------
        float or array-like
            value(s) of the atomic structure factor

        """
        key = get_key(q, en)
        f, res = self.get_cache("f", key)
        if f:
            return res

        res = self.f0(q) + self.f1(en) + 1.0j * self.f2(en)
        self.set_cache("f", key, res)
        return res

    def __str__(self):
        ostr = self.name
        ostr += f" ({self.num:2d})"
        return ostr

    def __repr__(self):
        return self.__str__()
