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
# Copyright (C) 2012-2015 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
script to create the HDF5 database from the raw data of XOP
this file is only needed for administration
"""

import lzma
import os.path

# local import
from database import (DataBase, add_f0_from_intertab, add_f1f2_from_kissel,
                      add_mass_from_NIST, init_material_db)

filename = os.path.join('data', 'elements.db')
dbf = DataBase(filename)
dbf.Create('elementdata',
           'Database with elemental data from XOP and Kissel databases')

init_material_db(dbf)

# add a dummy element, this is useful not only for testing and should be
# kept in future! It can be used for structure factor calculation tests, and
# shows how the a database entry can be generated manually
dbf.SetMaterial('dummy')
dbf.SetF0([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # atomic structure factors
dbf.SetF1F2((0, 1e5), (0, 0), (0, 0))  # zero dispersion correction

add_mass_from_NIST(dbf, os.path.join('data', 'nist_atom.dat'))

# add F0(Q) for every element
# with lzma.open(os.path.join('data', 'f0_xop.dat.xz'), 'r') as xop:
#    add_f0_from_xop(dbf, xop)
with lzma.open(os.path.join('data', 'f0_InterTables.dat.xz'), 'r') as itf:
    add_f0_from_intertab(dbf, itf)

# add F1 and F2 from database
with lzma.open(os.path.join('data', 'f1f2_asf_Kissel.dat.xz'), 'r') as kf:
    add_f1f2_from_kissel(dbf, kf)
# with lzma.open(os.path.join('data', 'f1f2_Henke.dat'), 'r') as hf:
#    add_f1f2_from_henkedb(dbf, hf)

# Also its possible to add custom data from different databases; e.g.
# created by Hepaestus (http://bruceravel.github.io/demeter/). This is also
# possible for specific elements only, therefore extract the data from
# Hephaestus or any other source producing ASCII files with three columns
# (energy (eV), f1, f2). To import such data use:
# add_f1f2_from_ascii_file(dbf, os.path.join('data', 'Ga.f1f2'), 'Ga')

dbf.Close()
