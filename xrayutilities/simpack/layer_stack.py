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
# Copyright (C) 2015-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import collections


class Layer(object):
    """
    Object describing part of a thin film sample. The properties of a layer
    are:

    material:   an xrayutilties material describing optical and crystal
                properties of the thin film
    thickness:  film thickness in Angstrom
    roughness:  root mean square roughness of the top interface in Angstrom
    """
    def __init__(self, material, thickness, roughness=None):
        """
        constructor for the material saving its properties

        Parameters
        ----------
         material:  an xrayutilities material to describe the properties of the
                    layer
         thickness: thickness of the layer in Angstrom
         roughness: roughness of the top interface in Angstrom
        """
        self.material = material
        self.thickness = thickness
        self.roughness = roughness


class LayerStack(collections.MutableSequence):
    """
    extends the built in list type to enable building a stack of Layer by
    various methods.
    """
    def __init__(self, name, *args):
        self.list = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, Layer):
            raise TypeError('LayerStack can only contain Layer as entries!')

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __str__(self):
        return str(self.list)

