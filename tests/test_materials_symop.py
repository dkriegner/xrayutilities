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
# Copyright (C) 2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import itertools
import re
import unittest

import numpy
import xrayutilities as xu
from xrayutilities.materials import SymOp


class TestMaterialsSymOp(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.x, cls.y, cls.z = numpy.random.rand(3)
        a, b, c = numpy.random.rand(3) * 2 + 4
        al, be, gam = numpy.random.rand(3) * 60 + 60
        pdict = {'a': a, 'b': b, 'c': c, 'alpha': al, 'beta': be, 'gamma': gam}

        cls.lats = []
        cls.gps = []
        for sg in xu.materials.spacegrouplattice.wp.keys():
            # determine parameters for this space group
            sgnr = int(sg.split(':')[0])
            csys, nargs = xu.materials.spacegrouplattice.sgrp_sym[sgnr]
            params = xu.materials.spacegrouplattice.sgrp_params[csys][0]
            p = [eval(par, pdict) for par in params]
            # generate test lattice
            cls.lats.append(xu.materials.SGLattice(sg, *p))
            # get general Wyckoff position of space group
            wp = xu.materials.spacegrouplattice.wp
            gplabel = sorted(wp[sg], key=lambda s: int(s[:-1]))[-1]
            cls.gps.append(wp[sg][gplabel][1])

    def test_SymOp_fromxyz(self):
        """
        tests that the creation of the symmetry operations from the xyz string
        of the Wyckoff positions is reproducible/reversable
        """
        for lat, gp in zip(self.lats, self.gps):
            symopsxyz = map(lambda s: '({})'.format(s.xyz()), lat.symops)
            self.assertCountEqual(symopsxyz, gp)

    def test_equivalent_hkl(self):
        hkl = numpy.random.randint(-11, 12, 3)
        for lat in self.lats:
            ehkl = numpy.unique(numpy.einsum('...ij,j', lat._hklsym, hkl),
                                axis=0)
            ehkl = set(tuple(e) for e in ehkl)
            self.assertEqual(lat.equivalent_hkls(hkl), ehkl)

    def test_iscentrosymmetric(self):
        centrosym = list(itertools.chain([2], range(10, 16), range(47, 75),
                                         range(83, 89), range(123, 143),
                                         [147, 148], range(162, 168),
                                         [175, 176], range(191, 195),
                                         range(200, 207), range(221, 231)))
        for lat in self.lats:
            if lat.space_group_nr in centrosym:
                self.assertTrue(lat.iscentrosymmetric)
            else:
                self.assertFalse(lat.iscentrosymmetric)

    def test_Wyckoff_consistency(self):
        """
        tests that all Wyckoff positions are consistent with the symmetry
        operations
        """
        reint = re.compile('[0-9]+')
        pardict = {'x': self.x, 'y': self.y, 'z': self.z}
        wp = xu.materials.spacegrouplattice.wp
        for lat, gp in zip(self.lats, self.gps):
            # check every Wyckoff position
            for wpkey in wp[lat.space_group]:
                uniquepos = []
                poscount = int(reint.match(wpkey).group())
                thispositions = list(
                    map(lambda p: SymOp.foldback(eval(p, pardict)),
                        wp[lat.space_group][wpkey][1]))
                pos0 = SymOp.foldback(thispositions[0])
                genpos = [s.apply(pos0) for s in lat.symops]
                uniquepos = [genpos[0], ]
                for p in genpos:
                    considered = False
                    for u in uniquepos:
                        if numpy.allclose(p, u):
                            considered = True
                    if not considered:
                        uniquepos.append(p)
                comparepos = [(t, g) for t in thispositions for g in uniquepos
                              if numpy.allclose(t, g)]
                # check that number of Wyckoff position entries are correct
                self.assertEqual(poscount, len(thispositions))
                # check that an equal number of unique positions is produced
                self.assertEqual(poscount, len(uniquepos))
                # check that unique positions are equal to Wyckoff positions
                self.assertEqual(poscount, len(comparepos))


if __name__ == '__main__':
    unittest.main()
