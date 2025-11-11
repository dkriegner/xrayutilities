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
# Copyright (C) 2014-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = "alignment.log.gz"
datadir = os.path.join(os.path.dirname(__file__), "data")
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(
    not os.path.isfile(fullfilename),
    "additional test data needed (http://xrayutilities.sf.io)",
)
class TestIO_SPEC_RA_Log(unittest.TestCase):
    peaks = ["asymaz1", "symaz1"]
    niterations = 639

    @classmethod
    def setUpClass(cls):
        cls.logfile = xu.io.RA_Alignment(fullfilename)

    def test_peaknames(self):
        self.assertEqual(self.peaks, self.logfile.peaks)

    def test_iterations(self):
        self.assertEqual(
            self.niterations, sum([sum(i) for i in self.logfile.iterations])
        )


if __name__ == "__main__":
    unittest.main()
