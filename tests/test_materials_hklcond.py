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

import math
import os
import unittest

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 0


@unittest.skipIf('CI' in os.environ, "slow test not running on CI")
class Test_Materials_reflection_condition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = xu.PowderExperiment(en=10000)
        cls.ksinmax = cls.p.k0 * math.sin(math.radians(179.9)/2)
        cls.materials = []
        for name, obj in xu.materials.predefined_materials.__dict__.items():
            if isinstance(obj, xu.materials.Crystal):
                cls.materials.append(obj)

    def _test_material(self, m, test_allowed=False):
        """
        helper function to test one specific material

        Parameters
        ----------
        m: xu.materials.Crystal
         material to be tested
        test_allowed: bool
         boolean to determine wether allowed peaks must have non-zero structure
         factor. This test is only performed if reflection conditions are
         available, and two atoms have a minimal distance of 0.01Ang! still
         there is a chance that by chance atoms occupy positions of a
         different space group which renders this test option dangerous.
         default: False
        """
        # calculate maximal Bragg indices
        hma = int(math.ceil(m.a / math.pi * self.ksinmax))
        hmi = -hma
        kma = int(math.ceil(m.b / math.pi * self.ksinmax))
        kmi = -kma
        lma = int(math.ceil(m.c / math.pi * self.ksinmax))
        lmi = -lma

        # calculate structure factors
        qmax = 2 * self.ksinmax
        hkl = numpy.mgrid[hma:hmi-1:-1,
                          kma:kmi-1:-1,
                          lma:lmi-1:-1].reshape(3, -1).T

        q = m.Q(hkl)
        qnorm = numpy.linalg.norm(q, axis=1)
        mask = numpy.logical_and(qnorm > 0, qnorm <= qmax)
        errorinfo = str(m)
        allowed = numpy.asarray([m.lattice.hkl_allowed(b) for b in hkl[mask]])
        # all forbidden peaks must have structure factor close to zero
        s = numpy.abs(
            m.StructureFactorForQ(q[mask][numpy.logical_not(allowed)],
                                  self.p.energy))
        mviolate = numpy.logical_not(numpy.isclose(s, 0))
        if numpy.any(mviolate):
            for h, sf in zip(hkl[mask][numpy.logical_not(allowed)][mviolate],
                             s[mviolate]):
                errorinfo += "%s\t%s\n" % (h, sf)
        self.assertTrue(numpy.allclose(s, 0), msg=errorinfo)
        # check if atoms are too near (reduce chance of accidental extinction)
        if test_allowed:
            for at, pos, occ, b in m.lattice.base():
                env = m.environment(*pos, maxdist=0.01)
                if len(env) > 1 or env[0][-1] != 1:
                    test_allowed = False
                    break
        if test_allowed and 'n/a' not in m.lattice.reflection_conditions():
            # all allowed peaks must have non-zero structure factor
            s = numpy.abs(
                m.StructureFactorForQ(q[mask][allowed], self.p.energy))
            mviolate = numpy.isclose(s, 0)
            if numpy.any(mviolate):
                for h, sf in zip(hkl[mask][allowed][mviolate], s[mviolate]):
                    errorinfo += "%s\t%s\n" % (h, sf)
            self.assertTrue(numpy.all(numpy.logical_not(numpy.isclose(s, 0))),
                            msg=errorinfo)

    def test_hklcond_predefined(self):
        for m in self.materials:
            self._test_material(m)

    def test_hklcond_random(self):
        # create random materials with random unit cell and test them
        x, y, z = numpy.random.rand(3)
        a, b, c = numpy.random.rand(3) * 2 + 4
        al, be, gam = numpy.random.rand(3) * 60 + 60
        pdict = {'a': a, 'b': b, 'c': c, 'alpha': al, 'beta': be, 'gamma': gam}
        wp = xu.materials.spacegrouplattice.wp

        for sg in wp.keys():
            # determine parameters for this space group
            sgnr = int(sg.split(':')[0])
            csys, nargs = xu.materials.spacegrouplattice.sgrp_sym[sgnr]
            params = xu.materials.spacegrouplattice.sgrp_params[csys][0]
            p = [eval(par, pdict) for par in params]
            # test all Wyckoff positions
            for wplabel in wp[sg].keys():
                wpentry = wp[sg][wplabel]
                wppar = []
                if wpentry[0] & 1:
                    wppar.append(x)
                if wpentry[0] & 2:
                    wppar.append(y)
                if wpentry[0] & 4:
                    wppar.append(z)
                kwdict = {'atoms': [xu.materials.elements.Dummy, ], 'pos':
                          [(wplabel, wppar), ]}
                # generate test lattice
                lat = xu.materials.SGLattice(sg, *p, **kwdict)
                self._test_material(xu.materials.Crystal("SG%s" % sg, lat))

    def test_get_allowed_hkl(self):
        """
        use some random hkls to test the get_allowed_hkl function. The internal
        methods used by this function is also tested more thoroughly by other
        unit tests.
        """
        qmax = 2 * self.ksinmax
        N = 5
        for m in self.materials:
            if 'n/a' in m.lattice.reflection_conditions():
                continue
            hkls = m.lattice.get_allowed_hkl(qmax)
            hma = int(math.ceil(m.a / math.pi * self.ksinmax))
            hmi = -hma
            kma = int(math.ceil(m.b / math.pi * self.ksinmax))
            kmi = -kma
            lma = int(math.ceil(m.c / math.pi * self.ksinmax))
            lmi = -lma
            errorinfo = str(m)
            errorinfo += "HKL min/max: %d %d %d / %d %d %d" % (hmi, kmi, lmi,
                                                               hma, kma, lma)
            for h, k, l in zip(numpy.random.randint(hmi, hma, N),
                               numpy.random.randint(kmi, kma, N),
                               numpy.random.randint(lmi, lma, N)):
                qnorm = numpy.linalg.norm(m.Q(h, k, l))
                if qnorm > qmax or qnorm == 0:
                    continue
                r = abs(m.StructureFactor(m.Q(h, k, l)))**2
                # test that a peak with non-zero structure factor is indeed
                # allowed
                # Note: the opposite test is not possible because of accidental
                # extinctions
                if not numpy.isclose(r, 0):
                    self.assertIn((h, k, l), hkls, msg=errorinfo)


if __name__ == '__main__':
    unittest.main()
