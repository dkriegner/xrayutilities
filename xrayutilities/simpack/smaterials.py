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
import copy
import numpy

from ..exception import InputError
from ..materials import Material, Crystal, PseudomorphicMaterial, Alloy


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
        self.name = material.name
        self.material = material
        for kw in kwargs:
            self.__setattr__(kw, kwargs[kw])

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __radd__(self, other):
        return MaterialList('%s + %s' % (other.name, self.name), other, self)

    def __add__(self, other):
        return MaterialList('%s + %s' % (self.name, other.name), self, other)

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("unsupported operand type(s) for *: "
                            "'SMaterial' and '%s'" % type(other))
        if other < 1:
            raise ValueError("multiplication factor needs to be positive!")
        m = MaterialList('%d * (%s)' % (other, self.name), self)
        for i in range(other-1):
            m.append(self)
        return m

    __rmul__ = __mul__

    def __repr__(self):
        s = '\n{cls}-{name} ('.format(name=self.material.name,
                                      cls=self.__class__.__name__)
        for k in self.__dict__:
            if k not in ('material', 'name'):
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
        self.namelist = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, SMaterial):
            raise TypeError('%s can only contain SMaterial as entries!'
                            % self.__class__.__name__)

    def _get_unique_name(self, v):
        if v.name not in self.namelist:
            return v.name
        else:
            num = 1
            name = '{name}_{num:d}'.format(name=v.name, num=num)
            while name in self.namelist:
                num += 1
                name = '{name}_{num:d}'.format(name=v.name, num=num)
            return name

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.namelist[i] = self._get_unique_name(v)
        self.list[i] = v

    def insert(self, i, v):
        if isinstance(v, MaterialList):
            vs = v
        else:
            vs = [v, ]
        for j, val in enumerate(vs):
            self.check(val)
            self.namelist.insert(i+j, self._get_unique_name(val))
            self.list.insert(i+j, val)

    def __radd__(self, other):
        ml = MaterialList('%s + %s' % (other.name, self.name))
        ml.append(other)
        ml.append(self)
        return ml

    def __add__(self, other):
        ml = MaterialList('%s + %s' % (self.name, other.name))
        ml.append(self)
        ml.append(other)
        return ml

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("unsupported operand type(s) for *: "
                            "'MaterialList' and '%s'" % type(other))
        if other < 1:
            raise ValueError("multiplication factor needs to be positive!")
        m = MaterialList('%d * (%s)' % (other, self.name), self)
        for i in range(other-1):
            m.append(self)
        return m

    __rmul__ = __mul__

    def __str__(self):
        s = '{name}\n{l}'.format(name=self.name, l=str(self.list))
        return s

    def __repr__(self):
        return self.name


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
                    'roughness' is the root mean square roughness (\AA)
                    'density' relativ density of the material; 1 for nominal
                    density
                    'relaxation' is the degree of relaxation in case of
                    crystalline thin films
        """
        for kw in kwargs:
            if kw not in ('roughness', 'density', 'relaxation'):
                raise TypeError('%s is an invalid keyword argument' % kw)
        kwargs['thickness'] = thickness
        super(Layer, self).__init__(material, **kwargs)


class LayerStack(MaterialList):
    """
    extends the built in list type to enable building a stack of Layer by
    various methods.
    """
    def check(self, v):
        if not isinstance(v, Layer):
            raise TypeError('LayerStack can only contain Layer as entries!')


class CrystalStack(LayerStack):
    """
    extends the built in list type to enable building a stack of crystalline
    Layers by various methods.
    """
    def check(self, v):
        super(CrystalStack, self).check(v)
        if not isinstance(v.material, Crystal):
            raise TypeError('CrystalStack can only contain crystalline Layers'
                            ' as entries!')


class GradedLayerStack(CrystalStack):
    """
    generates a sequence of layers with a gradient in chemical composition
    """
    def __init__(self, alloy, xfrom, xto, nsteps, thickness, **kwargs):
        """
        constructor for a graded buffer of the material 'alloy' with chemical
        composition from 'xfrom' to 'xto' with 'nsteps' number of sublayers.
        The total thickness of the graded buffer is 'thickness'

        Parameters
        ----------
         alloy:     Alloy function which allows to create a material with
                    chemical composition 'x' by alloy(x)
         xfrom/xto: chemical composition from the bottom to top
         nsteps:    number of sublayers in the graded buffer
         thickness: total thickness of the graded stack
        """
        nfrom = alloy(xfrom).name
        nto = alloy(xto).name
        super(GradedLayerStack, self).__init__('(' + nfrom + '-' + nto + ')')
        for x in numpy.linspace(xfrom, xto, nsteps):
            l = Layer(alloy(x), thickness/nsteps, **kwargs)
            self.append(l)


class PseudomorphicStack001(CrystalStack):
    """
    generate a sequence of pseudomorphic crystalline Layers. Surface
    orientation is assumed to be 001 and materials should be cubic/tetragonal.
    """

    def make_epitaxial(self, i):
        l = self.list[i]
        if i == 0:
            return l
        psub = self.list[i-1].material
        mpseudo = PseudomorphicMaterial(psub, l.material, l.relaxation)
        self.list[i].material = mpseudo

    def __delitem__(self, i):
        del self.list[i]
        for j in range(i, len(self)):
            self.make_epitaxial(j)

    def __setitem__(self, i, v):
        self.check(v)
        self.namelist[i] = self._get_unique_name(v)
        self.list[i] = v
        for j in range(i, len(self)):
            self.make_epitaxial(j)

    def insert(self, i, v):
        if isinstance(v, MaterialList):
            vs = v
        else:
            vs = [v, ]
        for j, val in enumerate(vs):
            self.check(val)
            self.namelist.insert(i+j, self._get_unique_name(val))
            self.list.insert(i+j, copy.copy(val))
            for k in range(i+j, len(self)):
                self.make_epitaxial(k)
