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
# Copyright (C) 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'speclog.log.gz'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_SPECLog(unittest.TestCase):
    prompt = 'PSIC'
    line_cnt = 2046542
    testcmd = '84.PSIC>  mvr mu -.075; ct'
    testcmdline = 25

    def setUp(self):
        self.logfile = xu.io.SPECLog(testfile, self.prompt, path=datadir)

    def test_linenumber(self):
        self.assertEqual(self.line_cnt, self.logfile.line_counter)
        self.assertEqual(self.testcmd, str(self.logfile[self.testcmdline]))


if __name__ == '__main__':
    unittest.main()
