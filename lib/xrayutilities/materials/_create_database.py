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
# Copyright (C) 2009-2012 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2012-2020 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
script to create the HDF5 database from the raw data of XOP.
this file is only needed for administration and also used during the build

It accepts two command line arguments. The first one is the filename
of the database (required).

The second optional argument is the directory of the source data file used to
generate the database. If no path is given the 'data' sub-folder of the scripts
directory is used.
"""

import sys

# local import
from database import createAndFillDatabase

if __name__ == "__main__":
    createAndFillDatabase(*sys.argv[1:], verbose=False)
