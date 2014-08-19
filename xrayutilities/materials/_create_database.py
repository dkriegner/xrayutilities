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
# Copyright (C) 2012-2014 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
script to create the HDF5 database from the raw data of XOP
this file is only needed for administration
"""

import os

#if __name__ == "__main__" and __package__ is None:
#        __package__ = "xrayutilities"
#from . import database as db
execfile('database.py')

filename = os.path.join("data","elements.db")

dbf = DataBase(filename)
dbf.Create(filename,"Database with elemental data from XOP and Kissel databases")

init_material_db(dbf)

add_mass_from_NIST(dbf,os.path.join("data","nist_atom.dat"))
add_f0_from_xop(dbf,os.path.join("data","f0_xop.dat"))
add_f1f2_from_kissel(dbf,os.path.join("data","f1f2_asf_Kissel.dat"))
# alternative use the Henke database
# add_f1f2_from_henkedb(dbf,os.path.join("data","f1f2_Henke.dat"))

# Also its possible to add costum data from different databases; e.g. created by Hepaestus (http://bruceravel.github.io/demeter/)
# this is also possible for specific elements only, therefore extract the data from Hephaestus or any other source producing
# ASCII files with three columns (energy (eV), f1, f2)
# add_f1f2_from_ascii_file(dbf,os.path.join("data","Ga.f1f2"),'Ga')

dbf.Close()
