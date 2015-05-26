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
# Copyright (C) 2009-2015 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module containing the Atom class which handles the database access for atomic
scattering factors and the atomic mass.
"""
import atexit
import os.path
import re

from . import __path__
from . import database
from .. import config
from .. import utilities

_db = database.DataBase(os.path.join(__path__[0], "data", config.DBNAME))
_db.Open()


def _db_cleanup():
    _db.Close()

atexit.register(_db_cleanup)


class Atom(object):

    def __init__(self, name, num):
        self.name = name
        self.ostate = re.sub('[A-Za-z]', '', name)
        for r, o in zip(('dot', 'p', 'm'), ('.', '+', '-')):
            self.ostate = self.ostate.replace(o, r)

        self.basename = re.sub('[^A-Za-z]', '', name)
        self.num = num

    def __key__(self):
        """ key function to return the elements number """
        return self.num

    def __lt__(self, other_el):
        """ make elements sortable by their key """
        return self.__key__() < other_el.__key__()

    @property
    def weight(self):
        _db.SetMaterial(self.basename)
        return _db.weight

    def f0(self, q):
        _db.SetMaterial(self.basename)
        return _db.GetF0(q, self.ostate)

    def f1(self, en="config"):
        if en == "config":
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        return _db.GetF1(utilities.energy(en))

    def f2(self, en="config"):
        if en == "config":
            en = utilities.energy(config.ENERGY)

        _db.SetMaterial(self.basename)
        return _db.GetF2(utilities.energy(en))

    def f(self, q, en="config"):
        """
        function to calculate the atomic structure factor F

        Parameter
        ---------
         q:     momentum transfer
         en:    energy for which F should be calculated, if omitted the value
                from the xrayutilities configuration is used

        Returns
        -------
         f (float)
        """
        f = self.f0(q) + self.f1(en) + 1.j * self.f2(en)
        return f

    def __str__(self):
        ostr = self.name
        ostr += " (%2d)" % self.num
        return ostr

    def __repr__(self):
        return self.__str__()
