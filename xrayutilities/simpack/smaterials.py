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

from ..exception import InputError
from ..materials import Material


class SMaterial(object):
    """
    Simulation Material. Extends the xrayutilities Materials by properties
    needed for simulations
    """
    def __init__(self, material, **kwargs):
        """
        initialize a simulation material by specifiying its Material and
        optional other properties

        Parameters
        ----------
         material:  Material object containing optical/crystal properties of
                    for the simulation
         kwargs:    optional properties of the material needed for the
                    simulation
        """
        self.material = material
        for kw in kwargs:
            self.__setattr__(kw, kwargs[kw])

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __repr__(self):
        s = '{cls}-{name} ('.format(name=self.material.name,
                                    cls=self.__class__.__name__)
        for k in self.__dict__:
            if k != 'material':
                s += '{key}: {value}, '.format(key=k, value=getattr(self, k))
        return s + ')'


class MaterialList(collections.MutableSequence):
    """
    class representing the basics of a list of materials for simulations within
    xrayutilities. It extends the built in list type.
    """
    def __init__(self, name, *args):
        self.name = name
        self.list = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, SMaterial):
            raise TypeError('%s can only contain SMaterial as entries!'
                            % self.__class__.__name__)

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
        s = '{name}\n{l}'.format(name=self.name, l=str(self.list))
        return s


class Layer(SMaterial):
    """
    Object describing part of a thin film sample. The properties of a layer
    are:

    material:   an xrayutilties material describing optical and crystal
                properties of the thin film
    thickness:  film thickness in Angstrom
    roughness:  root mean square roughness of the top interface in Angstrom
    """
    def __init__(self, material, thickness, **kwargs):
        """
        constructor for the material saving its properties

        Parameters
        ----------
         material:  an xrayutilities material to describe the properties of the
                    layer
         thickness: thickness of the layer in Angstrom
         kwargs:    optional keyword arguments with further layer properties.
                    currently only 'roughness' is supported
        """
        for kw in kwargs:
            if kw not in ('roughness',):
                raise TypeError('%s is an invalid keyword argument' % kw)
        kwargs['thickness'] = thickness
        super(self.__class__, self).__init__(material, **kwargs)


class LayerStack(MaterialList):
    """
    extends the built in list type to enable building a stack of Layer by
    various methods.
    """
    def check(self, v):
        if not isinstance(v, Layer):
            raise TypeError('LayerStack can only contain Layer as entries!')
