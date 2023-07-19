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
# Copyright (c) 2016-2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import collections.abc
import copy
import numbers

import numpy

from .. import utilities
from ..materials import Crystal, PseudomorphicMaterial
from ..math import CoordinateTransform, Transform


def _multiply(a, b):
    """
    implement multiplication of SMaterial and MaterialList with integer
    """
    if not isinstance(b, int):
        raise TypeError("unsupported operand type(s) for *: "
                        "'%s' and '%s'" % (type(a), type(b)))
    if b < 1:
        raise ValueError("multiplication factor needs to be positive!")
    m = MaterialList('%d * (%s)' % (b, a.name), a)
    for _ in range(b-1):
        m.append(copy.deepcopy(a))
    return m


class SMaterial:
    """
    Simulation Material. Extends the xrayutilities Materials by properties
    needed for simulations
    """

    def __init__(self, material, name=None, **kwargs):
        """
        initialize a simulation material by specifiying its Material and
        optional other properties

        Parameters
        ----------
        material :  Material (Crystal, or Amorphous)
            Material object containing optical/crystal properties of for the
            simulation; a deepcopy is used internally.
        name : str, optional
            name of the material used in the simulations
        kwargs :    dict
            optional properties of the material needed for the simulation
        """
        if name is not None:
            self.name = utilities.makeNaturalName(name, check=True)
        else:
            self.name = utilities.makeNaturalName(material.name, check=True)
        self.material = copy.deepcopy(material)
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self._material = material
        if isinstance(material, Crystal):
            self._structural_params = []
            # make lattice parameters attributes
            for param, value in material.lattice.free_parameters.items():
                self._structural_params.append(param)
                setattr(self, param, value)
            # make attributes from atom positions
            for i, wp in enumerate(material.lattice._wbase):
                if wp[1][1] is not None:
                    for j, p in enumerate(wp[1][1]):
                        name = '_'.join(('at%d' % i, wp[0].name,
                                         wp[1][0], str(j), 'pos'))
                        self._structural_params.append(name)
                        setattr(self, name, p)
            # make attributes from atom occupations
            for i, wp in enumerate(material.lattice._wbase):
                name = '_'.join(('at%d' % i, wp[0].name,
                                 wp[1][0], 'occupation'))
                self._structural_params.append(name)
                setattr(self, name, wp[2])
            # make attributes from Debye waller exponents
            for i, wp in enumerate(material.lattice._wbase):
                name = '_'.join(('at%d' % i, wp[0].name, wp[1][0], 'biso'))
                self._structural_params.append(name)
                setattr(self, name, wp[3])

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if hasattr(self, 'material'):
            if isinstance(self.material, Crystal):
                if name in self.material.lattice.free_parameters:
                    setattr(self.material.lattice, name, value)
                if name.startswith('at'):
                    nsplit = name.split('_')
                    idx = int(nsplit[0][2:])
                    wp = self.material.lattice._wbase[idx]
                    # wyckoff position parameter
                    if nsplit[-1] == 'pos':
                        pidx = int(nsplit[-2])
                        wyckpos = (wp[1][0], list(wp[1][1]))
                        wyckpos[1][pidx] = value
                        self.material.lattice._wbase[idx] = (wp[0], wyckpos,
                                                             wp[2], wp[3])
                    # site occupation
                    if nsplit[-1] == 'occupation':
                        self.material.lattice._wbase[idx] = (wp[0], wp[1],
                                                             value, wp[3])
                    # site DW exponent
                    if nsplit[-1] == 'biso':
                        self.material.lattice._wbase[idx] = (wp[0], wp[1],
                                                             wp[2], value)

    def __radd__(self, other):
        return MaterialList(f'{other.name} + {self.name}', other, self)

    def __add__(self, other):
        return MaterialList(f'{self.name} + {other.name}', self, other)

    def __mul__(self, other):
        return _multiply(self, other)

    __rmul__ = __mul__

    def __repr__(self):
        s = f'{self.__class__.__name__}-{self.name} ('
        for k in self.__dict__:
            if k not in ('name', '_material', '_structural_params'):
                v = getattr(self, k)
                if isinstance(v, numbers.Number):
                    s += f'{k}: {v:.5g}, '
                else:
                    s += f'{k}: {v}, '
        return s + ')'


class MaterialList(collections.abc.MutableSequence):
    """
    class representing the basics of a list of materials for simulations within
    xrayutilities. It extends the built in list type.
    """

    def __init__(self, name, *args):
        if not isinstance(name, str):
            raise TypeError("'name' argument must be a string")
        self.name = name
        self.list = list()
        self.namelist = list()
        self.extend(list(args))

    def check(self, v):
        if not isinstance(v, SMaterial):
            raise TypeError('%s can only contain SMaterial as entries!'
                            % self.__class__.__name__)

    def _set_unique_name(self, v):
        if v.name in self.namelist:
            splitname = v.name.split('_')
            if len(splitname) > 1:
                try:
                    num = int(splitname[-1])
                    basename = '_'.join(splitname[:-1])
                except ValueError:
                    num = 1
                    basename = v.name
            else:
                num = 1
                basename = v.name
            name = f'{basename}_{num:d}'
            while name in self.namelist:
                num += 1
                name = f'{basename}_{num:d}'
            v.name = name
        return v.name

    def __len__(self): return len(self.list)

    def __getitem__(self, i): return self.list[i]

    def __delitem__(self, i): del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.namelist[i] = self._set_unique_name(v)
        self.list[i] = v

    def insert(self, i, v):
        if isinstance(v, MaterialList):
            vs = v
        else:
            vs = [v, ]
        for j, val in enumerate(vs):
            self.check(val)
            self.namelist.insert(i+j, self._set_unique_name(val))
            self.list.insert(i+j, val)

    def __radd__(self, other):
        ml = MaterialList(f'{other.name} + {self.name}')
        ml.append(other)
        ml.append(self)
        return ml

    def __add__(self, other):
        ml = MaterialList(f'{self.name} + {other.name}')
        ml.append(self)
        ml.append(other)
        return ml

    def __mul__(self, other):
        return _multiply(self, other)

    __rmul__ = __mul__

    def __str__(self):
        layer = ',\n  '.join([str(entry) for entry in self.list])
        s = f'{self.name} [\n  {layer}\n]'
        return s

    def __repr__(self):
        return self.name


class Layer(SMaterial):
    """
    Object describing part of a thin film sample. The properties of a layer
    are :

    Attributes
    ----------
    material :  Material (Crystal or Amorhous)
        an xrayutilties material describing optical and crystal properties of
        the thin film
    thickness : float
        film thickness in angstrom
    """

    _valid_init_kwargs = {
        'name': 'Custom name of the Layer',
        'roughness': 'root mean square roughness',
        'density': 'density in kg/m^3',
        'relaxation': 'degree of relaxation',
        'lat_correl': 'lateral correlation length'
    }

    def __init__(self, material, thickness, **kwargs):
        """
        constructor for the material saving its properties

        Parameters
        ----------
        material :  Material (Crystal or Amorhous)
            an xrayutilties material describing optical and crystal properties
            of the thin film
        thickness : float
            film thickness in angstrom
        kwargs :    dict
            optional keyword arguments with further layer properties.
        roughness : float, optional
            root mean square roughness of the top interface in angstrom
        density :    float, optional
            density of the material in kg/m^3; If not specified the density of
            the material will be used.
        relaxation : float, optional
            the degree of relaxation in case of crystalline thin films
        lat_correl : float, optional
            the lateral correlation length for diffuse reflectivity
            calculations
        """
        utilities.check_kwargs(kwargs, self._valid_init_kwargs,
                               self.__class__.__name__)
        kwargs['thickness'] = thickness
        super().__init__(material, **kwargs)

    def __getattr__(self, name):
        """
        return default values for properties if they were not set
        """
        if name == "density":
            return self.material.density
        if name == "roughness":
            return 0
        if name == "lat_correl":
            return numpy.inf
        if name == "relaxation":
            return 1
        return super().__getattribute__(name)


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
        super().check(v)
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
        alloy :         function
            Alloy function which allows to create a material with chemical
            composition 'x' by alloy(x)
        xfrom, xto :    float
            chemical composition from the bottom to top
        nsteps :        int
            number of sublayers in the graded buffer
        thickness :     float
            total thickness of the graded stack
        """
        nfrom = alloy(xfrom).name
        nto = alloy(xto).name
        super().__init__('(' + nfrom + '-' + nto + ')')
        for x in numpy.linspace(xfrom, xto, nsteps):
            layer = Layer(alloy(x), thickness/nsteps, **kwargs)
            self.append(layer)


class PseudomorphicStack001(CrystalStack):
    """
    generate a sequence of pseudomorphic crystalline Layers. Surface
    orientation is assumed to be 001 and materials must be cubic/tetragonal.
    """
    trans = Transform(numpy.identity(3))

    def make_epitaxial(self, i):
        """Make the i-th sublayer pseudomorphic to the layer below."""
        layer = self.list[i]
        if i == 0:
            return
        psub = self.list[i-1].material
        mpseudo = PseudomorphicMaterial(psub, layer.material, layer.relaxation,
                                        trans=self.trans)
        self.list[i].material = mpseudo

    def __delitem__(self, i):
        del self.list[i]
        for j in range(i, len(self)):
            self.make_epitaxial(j)

    def __setitem__(self, i, v):
        self.check(v)
        self.namelist[i] = self._set_unique_name(v)
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
            self.namelist.insert(i+j, self._set_unique_name(val))
            self.list.insert(i+j, copy.copy(val))
            for k in range(i+j, len(self)):
                self.make_epitaxial(k)


class PseudomorphicStack111(PseudomorphicStack001):
    """
    generate a sequence of pseudomorphic crystalline Layers. Surface
    orientation is assumed to be 111 and materials must be cubic.
    """
    trans = CoordinateTransform((1, -1, 0), (1, 1, -2), (1, 1, 1))


class Powder(SMaterial):
    """
    Object describing part of a powder sample. The properties of a powder
    are:

    Attributes
    ----------
    material :   Crystal
        an xrayutilties material (Crystal) describing optical and crystal
        properties of the powder
    volume :     float
        powder's volume (in pseudo units, since only the relative volume enters
        the calculation)

    crystallite_size_lor :      float, optional
        Lorentzian crystallite size fwhm (m)
    crystallite_size_gauss :    float, optional
        Gaussian crystallite size fwhm (m)
    strain_lor :                float, optional
        extra peak width proportional to tan(theta)
    strain_gauss :              float, optional
        extra peak width proportional to tan(theta)
    preferred_orientation :     tuple, optional
        HKL of the preferred orientation
    preferred_orientation_factor : float, optional
        March-Dollase preferred orientation factor: < 1 for platy crystallits ,
        > 1 for rod-like crystallites, and = 1 for random orientation of
        crystallites.
    """

    _valid_init_kwargs = {
        'name': 'Custom name of the Powder',
        'crystallite_size_lor': 'Lorentzian crystallite size',
        'crystallite_size_gauss': 'Gaussian crystallite size',
        'strain_lor': 'microstrain broadening',
        'strain_gauss': 'microstrain broadening',
        'preferred_orientation': 'HKL of the preferred orientation',
        'preferred_orientation_factor':
        'March-Dollase preferred orientation factor'
    }

    def __init__(self, material, volume, **kwargs):
        """
        constructor for the material saving its properties

        Parameters
        ----------
        material :      Crystal
            an xrayutilties material (Crystal) describing optical and crystal
            properties of the powder
        volume :        float
            powder's volume (in pseudo units, since only the relative volume
            enters the calculation)
        kwargs :        dict
            optional keyword arguments with further powder properties.
        crystallite_size_lor :      float, optional
            Lorentzian crystallite size fwhm (m)
        crystallite_size_gauss :    float, optional
            Gaussian crystallite size fwhm (m)
        strain_lor, strain_gauss :  float, optional
            extra peak width proportional to tan(theta);
            typically interpreted as microstrain broadening
        """
        utilities.check_kwargs(kwargs, self._valid_init_kwargs,
                               self.__class__.__name__)
        kwargs['volume'] = volume
        super().__init__(material, **kwargs)


class PowderList(MaterialList):
    """
    extends the built in list type to enable building a list of Powder
    by various methods.
    """

    def check(self, v):
        if not isinstance(v, Powder):
            raise TypeError('PowderList can only contain Powder as entries!')
