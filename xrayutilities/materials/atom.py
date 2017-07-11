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
# Copyright (C) 2009-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module containing the Atom class which handles the database access for atomic
scattering factors and the atomic mass.
"""
import hashlib
import os.path
import re

import numpy

from . import __path__, database
from .. import config, utilities

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str


_db = database.DataBase(os.path.join(__path__[0], "data", config.DBNAME))
_db.Open()


def get_key(*args):
    """
    generate a hash key for several possible types of arguments
    """
    tup = []
    for a in args:
        if isinstance(a, numpy.ndarray):
            tup.append(hashlib.md5(a).digest())
        elif isinstance(a, list):
            tup.append(hash(tuple(a)))
        else:
            tup.append(hash(a))
    return hash(tuple(tup))


class Atom(object):
    max_cache_length = 1000

    def __init__(self, name, num):
        self.name = name
        self.ostate = re.sub('[A-Za-z]', '', name)
        for r, o in zip(('dot', 'p', 'm'), ('.', '+', '-')):
            self.ostate = self.ostate.replace(o, r)

        self.basename = re.sub('[^A-Za-z]', '', name)
        self.num = num
        self.__weight = None
        self._dbcache = dict([(prop, []) for prop in ('f0', 'f1', 'f2', 'f')])

    def swapbasename(self,name):
        self.name = name

    def __key__(self):
        """ key function to return the elements number """
        return self.num

    def __lt__(self, other_el):
        """ make elements sortable by their key """
        return self.__key__() < other_el.__key__()

    @property
    def weight(self):
        if not self.__weight:
            _db.SetMaterial(self.basename)
            self.__weight = _db.weight
        return self.__weight

    def get_cache(self, prop, key):
        """
        check if a cached value exists to speed up repeated database requests

        Returns
        -------
         flag, result: if the flag is True then result contains the cached
                       result, otherwise result is None
        """
        history = self._dbcache[prop]
        for idx, (k, result) in enumerate(history):
            if k == key:
                history.insert(0, history.pop(idx))  # move to front
                return True, result
        return False, None

    def set_cache(self, prop, key, result):
        """
        set result to be cached to speed up future calls
        """
        history = self._dbcache[prop]
        if len(history) == self.max_cache_length:
            history.pop(-1)
        history.insert(0, (key, result))

    def f0(self, q):
        key = get_key(q)
        f, res = self.get_cache('f0', key)
        if f:
            return res
        _db.SetMaterial(self.basename)
        res = _db.GetF0(q, self.ostate)
        self.set_cache('f0', key, res)
        return res

    def f1(self, en='config'):
        key = get_key(en)
        f, res = self.get_cache('f1', key)
        if f:
            return res
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        res = _db.GetF1(utilities.energy(en))
        self.set_cache('f1', key, res)
        return res

    def f2(self, en='config'):
        key = get_key(en)
        f, res = self.get_cache('f2', key)
        if f:
            return res
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        res = _db.GetF2(utilities.energy(en))
        self.set_cache('f2', key, res)
        return res

    def f(self, q, en='config'):
        """
        function to calculate the atomic structure factor F

        Parameters
        ----------
         q:     momentum transfer
         en:    energy for which F should be calculated, if omitted the value
                from the xrayutilities configuration is used

        Returns
        -------
         f (float)
        """
        key = get_key(q, en)
        f, res = self.get_cache('f', key)
        if f:
            return res

        res = self.f0(q) + self.f1(en) + 1.j * self.f2(en)
        self.set_cache('f2', key, res)
        return res

    def __str__(self):
        ostr = self.name
        ostr += " (%2d)" % self.num
        return ostr

    def __repr__(self):
        return self.__str__()
