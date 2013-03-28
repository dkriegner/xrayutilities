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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy

Si = xu.materials.Si  # load material from materials submodule

# initialize experimental class with directions from experiment
exp = xu.HXRD(Si.Q(1,1,-2),Si.Q(1,1,1))

# calculate angles and print them to the screen
angs = exp.Q2Ang(Si.Q(1,1,1))
print(("|F000|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(0,0,0),exp.energy)))))

print("Si (111)")
print(("phi:%8.4f" %angs[2]))
print(("om: %8.4f" %angs[0]))
print(("tt: %8.4f" %angs[3]))
print(("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(1,1,1),exp.energy)))))


angs = exp.Q2Ang(Si.Q(2,2,4))
print("Si (224)")
print("phi:%8.4f" %angs[2])
print("om: %8.4f" %angs[0])
print("tt: %8.4f" %angs[3])
print("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(2,2,4),exp.energy))))


angs = exp.Q2Ang(Si.Q(3,3,1))
print("Si (331)")
print("phi:%8.4f" %angs[2])
print("om: %8.4f" %angs[0])
print("tt: %8.4f" %angs[3])
print("|F|: %8.3f" %(numpy.abs(Si.StructureFactor(Si.Q(3,3,1),exp.energy))))
