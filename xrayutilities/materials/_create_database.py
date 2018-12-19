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
# Copyright (C) 2012-2018 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
script to create the HDF5 database from the raw data of XOP
his file is only needed for administration and also used during the unit tests

Optionally it accepts two command line arguments. The first one is the filename
of the database. If the filename has no path or a relative path the data-folder
given as second argument will be used as root of the output file.

The second optional argument is the directory of the source data file used to
generate the database. If no path is given the 'data' sub-folder of the scripts
directory is used. The script therefore works if called from the extracted
tarball directory but not after installation. After installation the respective
data path must be specified. The script can be run with 0, one or two command
line arguments as described above.

See tests/test_materials_database for an example.
"""

import lzma
import os.path
import sys

# local import
from database import (DataBase, add_color_from_JMOL, add_f0_from_intertab,
                      add_f1f2_from_kissel, add_mass_from_NIST,
                      add_radius_from_WIKI, init_material_db)

verbose = False

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'elements.db'
if len(sys.argv) > 2:
    dataroot = sys.argv[2]
else:
    dataroot = os.path.join(os.path.dirname(__file__), 'data')

fullfilename = os.path.join(dataroot, fname)

dbf = DataBase(fullfilename)
dbf.Create('elementdata',
           'Database with elemental data from XOP and Kissel databases')

init_material_db(dbf)

# add a dummy element, this is useful not only for testing and should be
# kept in future! It can be used for structure factor calculation tests, and
# shows how the a database entry can be generated manually
dbf.SetMaterial('dummy')
dbf.SetF0([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # atomic structure factors
dbf.SetF1F2((0, 1e5), (0, 0), (0, 0))  # zero dispersion correction

add_mass_from_NIST(dbf, os.path.join(dataroot, 'nist_atom.dat'), verbose)
add_color_from_JMOL(dbf, os.path.join(dataroot, 'colors.dat'), verbose)
add_radius_from_WIKI(dbf, os.path.join(dataroot, 'atomic_radius.dat'), verbose)

# add F0(Q) for every element
# with lzma.open(os.path.join('data', 'f0_xop.dat.xz'), 'r') as xop:
#    add_f0_from_xop(dbf, xop, verbose)
with lzma.open(os.path.join(dataroot, 'f0_InterTables.dat.xz'), 'r') as itf:
    add_f0_from_intertab(dbf, itf, verbose)

# add F1 and F2 from database
with lzma.open(os.path.join(dataroot, 'f1f2_asf_Kissel.dat.xz'), 'r') as kf:
    add_f1f2_from_kissel(dbf, kf, verbose)
# with lzma.open(os.path.join(dataroot, 'f1f2_Henke.dat'), 'r') as hf:
#    add_f1f2_from_henkedb(dbf, hf, verbose)

# Also its possible to add custom data from different databases; e.g.
# created by Hepaestus (http://bruceravel.github.io/demeter/). This is also
# possible for specific elements only, therefore extract the data from
# Hephaestus or any other source producing ASCII files with three columns
# (energy (eV), f1, f2). To import such data use:
# add_f1f2_from_ascii_file(dbf, os.path.join(dataroot, 'Ga.f1f2'), 'Ga',
#                          verbose)

dbf.Close()
