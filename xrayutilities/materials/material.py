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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2016 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2012 Tanja Etzelstorfer <tanja.etzelstorfer@jku.at>

"""
Classes decribing materials. Materials are devided with respect to their
crystalline state in either Amorphous or Crystal types.  While for most
materials their crystalline state is defined few materials are also included as
amorphous which can be useful for calculation of their optical properties.
"""

import numpy
import scipy.optimize
import warnings
import operator
import numbers
import re

from . import lattice
from . import elements
from . import cif
from .atom import Atom
from .. import math
from .. import utilities
from .. import config
from ..exception import InputError

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str

numpy.seterr(divide='ignore', invalid='ignore')

map_ijkl2ij = {"00": 0, "11": 1, "22": 2,
               "12": 3, "20": 4, "01": 5,
               "21": 6, "02": 7, "10": 8}
map_ij2ijkl = {"0": [0, 0], "1": [1, 1], "2": [2, 2],
               "3": [1, 2], "4": [2, 0], "5": [0, 1],
               "6": [2, 1], "7": [0, 2], "8": [1, 0]}


def index_map_ijkl2ij(i, j):
    return map_ijkl2ij["%i%i" % (i, j)]


def index_map_ij2ijkl(ij):
    return map_ij2ijkl["%i" % ij]


def Cij2Cijkl(cij):
    """
    Converts the elastic constants matrix (tensor of rank 2) to
    the full rank 4 cijkl tensor.

    required input arguments:
     cij ................ (6,6) cij matrix as a numpy array

    return value:
     cijkl .............. (3,3,3,3) cijkl tensor as numpy array
    """

    # first have to build a 9x9 matrix from the 6x6 one
    m = numpy.zeros((9, 9), dtype=numpy.double)
    m[0:6, 0:6] = cij[:, :]
    m[6:9, 0:6] = cij[3:6, :]
    m[0:6, 6:9] = cij[:, 3:6]
    m[6:9, 6:9] = cij[3:6, 3:6]

    # now create the full tensor
    cijkl = numpy.zeros((3, 3, 3, 3), dtype=numpy.double)

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    mi = index_map_ijkl2ij(i, j)
                    mj = index_map_ijkl2ij(k, l)
                    cijkl[i, j, k, l] = m[mi, mj]
    return cijkl


def Cijkl2Cij(cijkl):
    """
    Converts the full rank 4 tensor of the elastic constants to
    the (6,6) matrix of elastic constants.

    required input arguments:
     cijkl .............. (3,3,3,3) cijkl tensor as numpy array

    return value:
     cij ................ (6,6) cij matrix as a numpy array
    """

    # build the temporary 9x9 matrix
    m = numpy.zeros((9, 9), dtype=numpy.double)

    for i in range(0, 9):
        for j in range(0, 9):
            ij = index_map_ij2ijkl(i)
            kl = index_map_ij2ijkl(j)
            m[i, j] = cijkl[ij[0], ij[1], kl[0], kl[1]]

    cij = m[0:6, 0:6]

    return cij


class Material(object):
    """
    base class for all Materials. common properties of amorphous and
    crystalline materials are described by this class from which Amorphous and
    Crystal are derived from.
    """
    def __init__(self, name, cij=None):
        if cij is None:
            self.cij = numpy.zeros((6, 6), dtype=numpy.double)
        elif isinstance(cij, list):
            self.cij = numpy.array(cij, dtype=numpy.double)
        elif isinstance(cij, numpy.ndarray):
            self.cij = cij
        else:
            raise TypeError("Elastic constants must be a list or numpy array!")

        self.name = name
        self.cijkl = Cij2Cijkl(self.cij)
        self.transform = None
        self._density = None

    def __getattr__(self, name):
        if name.startswith("c"):
            index = name[1:]
            if len(index) > 2:
                raise AttributeError("Cij indices must be between 1 and 6")

            i = int(index[0])
            j = int(index[1])

            if i > 6 or i < 1 or j > 6 or j < 1:
                raise AttributeError("Cij indices must be between 1 and 6")

            if self.transform:
                cij = Cijkl2Cij(self.transform(Cij2Cijkl(self.cij)))
            else:
                cij = self.cij

            return cij[i - 1, j - 1]

    def _getmu(self):
        return self.cij[3, 3]

    def _getlam(self):
        return self.cij[0, 1]

    def _getnu(self):
        return self.lam / 2. / (self.mu + self.lam)

    def _getdensity(self):
        return self._density

    density = property(_getdensity)
    mu = property(_getmu)
    lam = property(_getlam)
    nu = property(_getnu)

    def delta(self, en='config'):
        pass

    def beta(self, en='config'):
        pass

    def chi0(self, en='config'):
        """
        calculates the complex chi_0 values often needed in simulations.
        They are closely related to delta and beta
        (n = 1 + chi_r0/2 + i*chi_i0/2   vs.  n = 1 - delta + i*beta)
        """
        return (-2 * self.delta(en) + 2j * self.beta(en))

    def idx_refraction(self, en="config"):
        """
        function to calculate the complex index of refraction of a material
        in the x-ray range

        Parameters
        ----------
         en:    energy of the x-rays, if omitted the value from the
                xrayutilities configuration is used

        Returns
        -------
         n (complex)
        """
        n = 1. - self.delta(en) + 1.j * self.beta(en)
        return n

    def critical_angle(self, en='config', deg=True):
        """
        calculate critical angle for total external reflection

        Parameters
        ----------
         en:    energy of the x-rays, if omitted the value from the
                xrayutilities configuration is used
         deg:   return angle in degree if True otherwise radians (default:True)

        Returns
        -------
         Angle of total external reflection

        """
        rn = 1. - self.delta(en)

        alphac = numpy.arccos(rn)
        if deg:
            alphac = numpy.degrees(alphac)

        return alphac

    def absorption_length(self, en='config'):
        """
        wavelength dependent x-ray absorption length defined as
        mu = lambda/(2*pi*2*beta) with lambda and beta as the x-ray
        wavelength and complex part of the refractive index respectively.

        Parameters
        ----------
         en:    energy of the x-rays (in eV, optional)

        Returns
        -------
         the absorption length in um
        """
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)
        return utilities.en2lam(en) / (2 * numpy.pi * self.beta(en) * 2) / 1e4

    def __str__(self):
        ostr = "%s: %s\n" % (self.__class__.__name__, self.name)
        if numpy.any(self.cij):
            ostr += "Elastic tensor (6x6):\n"
            d = numpy.get_printoptions()
            numpy.set_printoptions(precision=2, linewidth=78, suppress=False)
            ostr += str(self.cij) + '\n'
            numpy.set_printoptions(d)

        return ostr


class Amorphous(Material):
    """
    amorphous materials are described by this class
    """
    def __init__(self, name, density, atoms=None, cij=None):
        """
        constructor of an amorphous material. The amorphous material is
        described by its density and atom composition.

        Parameters
        ----------
         name:      name of the material. To allow automatic parsing of the
                    chemical elements use the abbreviation of the chemical
                    element from the periodic table. To specify alloys, use
                    e.g. 'Ir0.2Mn0.8' or 'H2O'.
         density:   mass density in kg/m^3
         atoms:     list of atoms together with their fractional content.
                    When the name is a simply chemical formula then this can be
                    None.  To specify more complicated materials [('Ir', 0.2),
                    ('Mn', 0.8), ...]. Instead of the elements as string you
                    can also use an Atom object. If the contents to not add up
                    to 1 they will be corrected without notice.
        """
        super(Amorphous, self).__init__(name, cij)
        self._density = density
        self.base = list()
        if atoms is None:
            comp = Amorphous.parseChemForm(name)
            if config.VERBOSITY >= config.DEBUG:
                print("XU.materials.Amorphous: using '%s' as chemical formula"
                      % ''.join(['%s%.2f ' % (e.name, c) for e, c in comp]))
            for (e, c) in comp:
                self.base.append((e, c))
        else:
            frsum = numpy.sum([at[1] for at in atoms])
            for at, fr in atoms:
                if not isinstance(at, Atom):
                    a = getattr(elements, at)
                else:
                    a = at
                self.base.append((a, fr/frsum))

    @staticmethod
    def parseChemForm(cstring):
        """
        Parse a string containing a simple chemical formula and transform it to
        a list of elements together with their relative atomic fraction. e.g.
        'H2O' -> [(H, 2/3), (O, 1/3)], where H and O are the Element objects of
        Hydrogen and Oxygen. Note that every chemical element needs to start
        with a capital letter! Complicated formulas containing bracket are not
        supported!

        Parameters
        ----------
         cstring:   string containing the chemical fomula

        Returns
        -------
         list of tuples with chemical element and atomic fraction
        """
        if re.findall('[\(\)]', cstring):
            raise ValueError('unsupported chemical formula (%s) given.'
                             % cstring)
        elems = re.findall('[A-Z][^A-Z]*', cstring)
        r = re.compile("([a-zA-Z]+)([0-9\.]+)")
        ret = []
        csum = 0
        for e in elems:
            if r.match(e):
                elstr, cont = r.match(e).groups()
                cont = float(cont)
            else:
                elstr, cont = (e, 1.0)
            ret.append((elstr, cont))
            csum += cont
        for i, r in enumerate(ret):
            ret[i] = (getattr(elements, r[0]), r[1]/csum)
        return ret

    def _get_f(self, q, en):
        """
        optimized method to calculate the atomic scattering factor for all
        atoms in the unit cell by calling the database only as much as needed.

        Parameters
        ----------
         q:     momentum transfer for which the atomic scattering factor
                should be calculated
         en:    x-ray energy (eV)

        Returns
        -------
         list of atomic scattering factors for every atom in the unit cell
        """
        f = {}
        for at, occ in self.base:
            if at.num not in f:
                f[at.num] = at.f(q, en)
        return [f[a.num] for a, o in self.base]

    def delta(self, en='config'):
        """
        function to calculate the real part of the deviation of the
        refractive index from 1 (n=1-delta+i*beta)

        Parameters
        ----------
         en:    x-ray energy eV, if omitted the value from the xrayutilities
                configuration is used

        Returns
        -------
         delta (float)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        lam = utilities.en2lam(en)
        delta = 0.
        m = 0.
        f = self._get_f(0., en)
        for (at, occ), fa in zip(self.base, f):
            delta += numpy.real(fa) * occ
            m += at.weight * occ

        delta *= re / (2 * numpy.pi) * lam ** 2 / (m / self.density) * 1e-30
        return delta

    def beta(self, en='config'):
        """
        function to calculate the imaginary part of the deviation
        of the refractive index from 1 (n=1-delta+i*beta)

        Parameters
        ----------
         en:    x-ray energy eV, if omitted the value from the xrayutilities
                configuration is used

        Returns
        -------
         beta (float)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        lam = utilities.en2lam(en)
        beta = 0.
        m = 0.
        f = self._get_f(0., en)
        for (at, occ), fa in zip(self.base, f):
            beta += numpy.imag(fa) * occ
            m += at.weight * occ

        beta *= re / (2 * numpy.pi) * lam ** 2 / (m / self.density) * 1e-30
        return beta

    def chi0(self, en='config'):
        """
        calculates the complex chi_0 values often needed in simulations.
        They are closely related to delta and beta
        (n = 1 + chi_r0/2 + i*chi_i0/2   vs.  n = 1 - delta + i*beta)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        lam = utilities.en2lam(en)
        beta = 0.
        delta = 0.
        m = 0.
        f = self._get_f(0., en)
        for (at, occ), f0 in zip(self.base, f):
            beta += numpy.imag(f0) * occ
            delta += numpy.real(f0) * occ
            m += at.weight * occ

        beta *= re / (2 * numpy.pi) * lam ** 2 / (m / self.density) * 1e-30
        delta *= re / (2 * numpy.pi) * lam ** 2 / (m / self.density) * 1e-30
        return (-2 * delta + 2j * beta)

    def __str__(self):
        ostr = super(Amorphous, self).__str__()
        ostr += "density: %.2f\n" % self.density
        if len(self.base) > 0:
            ostr += "atoms: "
            for at, o in self.base:
                ostr += "(%s, %.3f) " % (at.name, o)
            ostr += "\n"

        return ostr


class Crystal(Material):
    """
    Crystalline materials are described by this class
    """
    def __init__(self, name, lat, cij=None, thetaDebye=None):
        super(Crystal, self).__init__(name, cij)

        self.lattice = lat
        self.rlattice = lat.ReciprocalLattice()
        if isinstance(thetaDebye, numbers.Number):
            self.thetaDebye = float(thetaDebye)
        else:
            self.thetaDebye = thetaDebye

    @classmethod
    def fromCIF(cls, ciffilename):
        """
        Create a Crystal from a CIF file. Name and

        Parameters
        ----------
         ciffilename:  filename of the CIF file

        Returns
        -------
         Crystal instance
        """
        cf = cif.CIFFile(ciffilename)
        lat = cf.Lattice()
        return cls(cf.name, lat)

    def _geta1(self):
        return self.lattice.a1

    def _geta2(self):
        return self.lattice.a2

    def _geta3(self):
        return self.lattice.a3

    def _getb1(self):
        return self.rlattice.a1

    def _getb2(self):
        return self.rlattice.a2

    def _getb3(self):
        return self.rlattice.a3

    def _get_matrixB(self):
        return self.rlattice.transform.matrix

    a1 = property(_geta1)
    a2 = property(_geta2)
    a3 = property(_geta3)
    b1 = property(_getb1)
    b2 = property(_getb2)
    b3 = property(_getb3)
    B = property(_get_matrixB)

    def Q(self, *hkl):
        """
        Return the Q-space position for a certain material.

        Parameters
        ----------
         hkl:   list or numpy array with the Miller indices
                (or Q(h,k,l) is also possible)

        """
        return self.rlattice.GetPoint(*hkl)

    def environment(self, *pos, **kwargs):
        """
        Returns a list of neighboring atoms for a given position within the the
        unit cell.

        Parameters
        ----------
         pos: list or numpy array with the fractional coordinated in the
              unit cell

        keyword arguments:
         maxdist:  maximum distance wanted in the list of neighbors
                   (default: 7)

        Returns
        -------
         list of tuples with (distance,atomType,multiplicity) giving distance
         (sorted) and type of neighboring atoms together with the amount of
         atoms at the given distance

        """

        for k in kwargs.keys():
            if k not in ['maxdist']:
                raise Exception("unknown keyword argument given: allowed is"
                                "only 'maxdist': maximum distance needed in "
                                "the output")

        maxdist = kwargs.get('maxdist', 7)

        if len(pos) < 3:
            pos = pos[0]
            if len(pos) < 3:
                raise InputError("need 3 coordinates of the "
                                 "reference position")

        refpos = self.a1 * \
            pos[0] + self.a2 * pos[1] + self.a3 * pos[2]

        l = []
        Na = 2 * int(numpy.ceil(maxdist / numpy.linalg.norm(self.a1)))
        Nb = 2 * int(numpy.ceil(maxdist / numpy.linalg.norm(self.a2)))
        Nc = 2 * int(numpy.ceil(maxdist / numpy.linalg.norm(self.a3)))
        if numpy.iterable(self.lattice.base):
            for a, p, o, b in self.lattice.base:
                ucpos = (self.a1 * p[0] + self.a2 * p[1] + self.a3 * p[2])
                for i in range(-Na, Na + 1):
                    for j in range(-Nb, Nb + 1):
                        for k in range(-Nc, Nc + 1):
                            atpos = ucpos + (self.a1 * i + self.a2 * j +
                                             self.a3 * k)
                            distance = numpy.linalg.norm(atpos - refpos)
                            if distance <= maxdist:
                                l.append((distance, a, o))
        else:
            for i in range(-Na, Na + 1):
                for j in range(-Nb, Nb + 1):
                    for k in range(-Nc, Nc + 1):
                        atpos = (self.a1 * i + self.a2 * j + self.a3 * k)
                        distance = numpy.linalg.norm(atpos - refpos)
                        if distance <= maxdist:
                            l.append((distance, '__dummy__', 1.))

        # sort
        l.sort(key=operator.itemgetter(1))
        l.sort(key=operator.itemgetter(0))
        rl = []
        if len(l) >= 1:
            mult = l[0][2]
        else:
            return rl
        for i in range(1, len(l)):
            if (numpy.abs(l[i - 1][0] - l[i][0]) < config.EPSILON and
                    l[i - 1][1] == l[i][1]):
                mult += l[i - 1][2]  # add occupancy
            else:
                rl.append((l[i - 1][0], l[i - 1][1], mult))
                mult = l[i][2]
        rl.append((l[-1][0], l[-1][1], mult))

        return rl

    def planeDistance(self, *hkl):
        """
        determines the lattice plane spacing for the planes specified by (hkl)

        Parameters
        ----------
         h,k,l:  Miller indices of the lattice planes given either as
                 list,tuple or seperate arguments

        Returns
        -------
         d:   the lattice plane spacing as float

        Examples
        --------
        >>> xu.materials.Si.planeDistance(0,0,4)
        1.3577600000000001

        or

        >>> xu.materials.Si.planeDistance((1,1,1))
        3.1356124059796255

        """
        if len(hkl) < 3:
            hkl = hkl[0]
            if len(hkl) < 3:
                raise InputError("need 3 indices for the lattice point")

        d = 2 * numpy.pi / numpy.linalg.norm(self.rlattice.GetPoint(hkl))

        return d

    def _getdensity(self):
        """
        calculates the mass density of an material from the mass of the atoms
        in the unit cell.

        Returns
        -------
        mass density in kg/m^3
        """
        m = 0.
        for at, pos, occ, b in self.lattice.base:
            m += at.weight * occ

        return m / self.lattice.UnitCellVolume() * 1e30

    density = property(_getdensity)

    def _get_f(self, q, en):
        """
        optimized method to calculate the atomic scattering factor for all
        atoms in the unit cell by calling the database only as much as needed.

        Parameters
        ----------
         q:     momentum transfer for which the atomic scattering factor
                should be calculated
         en:    x-ray energy (eV)

        Returns
        -------
         list of atomic scattering factors for every atom in the unit cell
        """
        f = {}
        if self.lattice.base:
            for at, pos, occ, b in self.lattice.base:
                if at.num not in f:
                    f[at.num] = at.f(q, en)
            return [f[a.num] for a, p, o, b in self.lattice.base]
        else:
            return None

    def _get_lamen(self, en):
        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)
        lam = utilities.en2lam(en)
        return lam, en

    def delta(self, en='config'):
        """
        function to calculate the real part of the deviation of the
        refractive index from 1 (n=1-delta+i*beta)

        Parameters
        ----------
         en:    x-ray energy eV, if omitted the value from the xrayutilities
                configuration is used

        Returns
        -------
         delta (float)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10

        lam, en = self._get_lamen(en)
        delta = 0.
        f = self._get_f(0, en)
        for (at, pos, occ, b), fa in zip(self.lattice.base, f):
            delta += numpy.real(fa) * occ

        delta *= re / (2 * numpy.pi) * lam ** 2 / \
            self.lattice.UnitCellVolume()
        return delta

    def beta(self, en='config'):
        """
        function to calculate the imaginary part of the deviation
        of the refractive index from 1 (n=1-delta+i*beta)

        Parameters
        ----------
         en:    x-ray energy eV, if omitted the value from the xrayutilities
                configuration is used

        Returns
        -------
         beta (float)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10

        lam, en = self._get_lamen(en)
        beta = 0.
        f = self._get_f(0, en)
        for (at, pos, occ, b), fa in zip(self.lattice.base, f):
            beta += numpy.imag(fa) * occ

        beta *= re / (2 * numpy.pi) * lam ** 2 / self.lattice.UnitCellVolume()
        return beta

    def chi0(self, en='config'):
        """
        calculates the complex chi_0 values often needed in simulations.
        They are closely related to delta and beta
        (n = 1 + chi_r0/2 + i*chi_i0/2   vs.  n = 1 - delta + i*beta)
        """
        re = scipy.constants.physical_constants['classical electron radius'][0]
        re *= 1e10

        lam, en = self._get_lamen(en)
        beta = 0.
        delta = 0.
        if self.lattice.base:
            f = self._get_f(0, en)
            for (at, pos, occ, b), f0 in zip(self.lattice.base, f):
                beta += numpy.imag(f0) * occ
                delta += numpy.real(f0) * occ

        beta *= re / (2 * numpy.pi) * lam ** 2 / self.lattice.UnitCellVolume()
        delta *= re / (2 * numpy.pi) * lam ** 2 / \
            self.lattice.UnitCellVolume()
        return (-2 * delta + 2j * beta)

    def _debyewallerfactor(self, temp, qnorm):
        """
        Calculate the Debye Waller temperature factor according to the Debye
        temperature

        Parameters
        ----------
         temp:      actual temperature (K)
         qnorm:     norm of the q-vector(s) for which the factor should be
                    calculated

        Returns
        -------
         the Debye Waller factor(s) with the same shape as qnorm
        """
        if temp != 0 and self.thetaDebye:
            # W(q) = 3/2* hbar^2*q^2/(m*kB*tD) * (D1(tD/T)/(tD/T) + 1/4)
            # DWF = exp(-W(q)) consistent with Vaclav H. and several books
            hbar = scipy.constants.hbar
            kb = scipy.constants.Boltzmann
            x = self.thetaDebye / float(temp)
            m = 0.
            for a, p, o, b in self.lattice.base:
                m += a.weight
            m = m / len(self.lattice.base)
            exponentf = 3 / 2. * hbar ** 2 * 1.0e20 / \
                (m * kb * self.thetaDebye) * (math.Debye1(x) / x + 0.25)
            if config.VERBOSITY >= config.DEBUG:
                print("XU.materials.Crystal: DWF = exp(-W*q**2) W= %g"
                      % exponentf)
            dwf = numpy.exp(-exponentf * qnorm ** 2)
        else:
            dwf = 1.0
        return dwf

    def chih(self, q, en='config', temp=0, polarization='S'):
        """
        calculates the complex polarizability of a material for a certain
        momentum transfer and energy

        Parameters
        ----------
         q:     momentum transfer in (1/A)
         en:    xray energy in eV, if omitted the value from the xrayutilities
                configuration is used
         temp:  temperature used for Debye-Waller-factor calculation
         polarization:  either 'S' (default) sigma or 'P' pi polarization

        Returns
        -------
         (abs(chih_real),abs(chih_imag)) complex polarizability
        """

        if isinstance(q, (list, tuple)):
            q = numpy.array(q, dtype=numpy.double)
        elif isinstance(q, numpy.ndarray):
            pass
        else:
            raise TypeError("q must be a list or numpy array!")
        qnorm = math.VecNorm(q)

        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        if polarization not in ('S', 'P'):
            raise ValueError("polarization must be 'S':sigma or 'P': pi!")

        if self.lattice.base is None:
            return (0, 0)

        dwf = self._debyewallerfactor(temp, qnorm)

        sr = 0. + 0.j
        si = 0. + 0.j
        # a: atom, p: position, o: occupancy, b: temperature-factor
        if self.lattice.base:
            f = self._get_f(qnorm, en)
            for (a, p, o, b), F in zip(self.lattice.base, f):
                r = self.lattice.GetPoint(p)
                if temp == 0:
                    dwf = numpy.exp(-b * qnorm ** 2 / (4 * numpy.pi) ** 2)
                fr = numpy.real(F) * o
                fi = numpy.imag(F) * o
                sr += fr * numpy.exp(-1.j * math.VecDot(q, r)) * dwf
                si += fi * numpy.exp(-1.j * math.VecDot(q, r)) * dwf

        # classical electron radius
        c = scipy.constants
        r_e = 1 / (4 * numpy.pi * c.epsilon_0) * c.e ** 2 / \
            (c.electron_mass * c.speed_of_light ** 2) * 1e10
        lam = utilities.en2lam(en)

        fact = -lam ** 2 * r_e / (numpy.pi * self.lattice.UnitCellVolume())
        rchi = numpy.abs(fact * sr)
        ichi = numpy.abs(fact * si)
        if polarization == 'P':
            theta = numpy.arcsin(qnorm * utilities.en2lam(en) / (4*numpy.pi))
            rchi *= numpy.cos(2 * theta)
            ichi *= numpy.cos(2 * theta)

        return rchi, ichi

    def dTheta(self, Q, en='config'):
        """
        function to calculate the refractive peak shift

        Parameters
        ----------
         Q:     momentum transfer (1/A)
         en:    x-ray energy (eV), if omitted the value from the xrayutilities
                configuration is used

        Returns
        -------
         deltaTheta: peak shift in degree
        """

        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)
        lam = utilities.en2lam(en)
        dth = numpy.degrees(
            2 * self.delta(en) / numpy.sin(2 * numpy.arcsin(
                lam * numpy.linalg.norm(Q) / (4 * numpy.pi))))
        return dth

    def __str__(self):
        ostr = super(Crystal, self).__str__()
        ostr += "Lattice:\n"
        ostr += self.lattice.__str__()
        ostr += "Reciprocal lattice:\n"
        ostr += self.rlattice.__str__()

        return ostr

    def StructureFactor(self, q, en='config', temp=0):
        """
        calculates the structure factor of a material
        for a certain momentum transfer and energy
        at a certain temperature of the material

        Parameters
        ----------
         q:     vectorial momentum transfer (vectors as list,tuple
                or numpy array are valid)
         en:    energy in eV, if omitted the value from the xrayutilities
                configuration is used
         temp:  temperature used for Debye-Waller-factor calculation

        Returns
        -------
         the complex structure factor
        """

        if isinstance(q, (list, tuple)):
            q = numpy.array(q, dtype=numpy.double)
        elif isinstance(q, numpy.ndarray):
            pass
        else:
            raise TypeError("q must be a list or numpy array!")

        if isinstance(en, basestring) and en == 'config':
            en = utilities.energy(config.ENERGY)

        if self.lattice.base is None:
            return 1.

        qnorm = math.VecNorm(q)
        dwf = self._debyewallerfactor(temp, qnorm)

        s = 0. + 0.j
        f = self._get_f(qnorm, en)
        # a: atom, p: position, o: occupancy, b: temperature-factor
        for (a, p, o, b), fq in zip(self.lattice.base, f):
            r = self.lattice.GetPoint(p)
            if temp == 0:
                dwf = numpy.exp(-b * qnorm ** 2 /
                                (4 * numpy.pi) ** 2)
            s += fq * o * numpy.exp(-1.j * math.VecDot(q, r)) * dwf

        return s

    def StructureFactorForEnergy(self, q0, en, temp=0):
        """
        calculates the structure factor of a material
        for a certain momentum transfer and a bunch of energies

        Parameters
        ----------
         q0:    vectorial momentum transfer (vectors as list,tuple
                or numpy array are valid)
         en:    list, tuple or array of energy values in eV
         temp:  temperature used for Debye-Waller-factor calculation

        Returns
        -------
         complex valued structure factor array
        """
        if isinstance(q0, (list, tuple)):
            q = numpy.array(q0, dtype=numpy.double)
        elif isinstance(q0, numpy.ndarray):
            q = q0
        else:
            raise TypeError("q must be a list or numpy array!")
        qnorm = math.VecNorm(q)

        if isinstance(en, (list, tuple)):
            en = numpy.array(en, dtype=numpy.double)
        elif isinstance(en, numpy.ndarray):
            pass
        else:
            raise TypeError("Energy data must be provided as a list "
                            "or numpy array!")

        if self.lattice.base is None:
            return numpy.ones(len(en))

        dwf = self._debyewallerfactor(temp, qnorm)

        s = 0. + 0.j
        f = self._get_f(qnorm, en)
        # a: atom, p: position, o: occupancy, b: temperature-factor
        for (a, p, o, b), fq in zip(self.lattice.base, f):
            if temp == 0:
                dwf = numpy.exp(-b * qnorm ** 2 / (4 * numpy.pi) ** 2)
            r = self.lattice.GetPoint(p)
            s += fq * o * dwf * numpy.exp(-1.j * math.VecDot(q, r))

        return s

    def StructureFactorForQ(self, q, en0='config', temp=0):
        """
        calculates the structure factor of a material
        for a bunch of momentum transfers and a certain energy

        Parameters
        ----------
         q:     vectorial momentum transfers;
                list of vectores (list, tuple or array) of length 3
                e.g.: (Si.Q(0,0,4),Si.Q(0,0,4.1),...) or
                numpy.array([Si.Q(0,0,4),Si.Q(0,0,4.1)])
         en0:   energy value in eV, if omitted the value from the xrayutilities
                configuration is used
         temp:  temperature used for Debye-Waller-factor calculation

        Returns
        -------
         complex valued structure factor array
        """
        if isinstance(q, (list, tuple, numpy.ndarray)):
            q = numpy.asarray(q, dtype=numpy.double)
        else:
            raise TypeError("q must be a list or numpy array!")
        if len(q.shape) != 2:
            raise ValueError("q does not have the correct shape (shape = %s)"
                             % str(q.shape))
        qnorm = numpy.linalg.norm(q, axis=1)

        if isinstance(en0, basestring) and en0 == 'config':
            en0 = utilities.energy(config.ENERGY)

        if self.lattice.base is None:
            return numpy.ones(len(q))

        dwf = self._debyewallerfactor(temp, qnorm)

        s = 0. + 0.j
        f = self._get_f(qnorm, en0)
        # a: atom, p: position, o: occupancy, b: temperature-factor
        for (a, p, o, b), fq in zip(self.lattice.base, f):
            if temp == 0:
                dwf = numpy.exp(-b * qnorm ** 2 / (4 * numpy.pi) ** 2)

            r = self.lattice.GetPoint(p)
            s += fq * o * numpy.exp(-1.j * numpy.dot(q, r)) * dwf

        return s

    def ApplyStrain(self, strain):
        """
        Applies a certain strain on the lattice of the material. The result is
        a change in the base vectors of the real space as well as reciprocal
        space lattice.  The full strain matrix (3x3) needs to be given.  Note:
        NO elastic response of the material will be considered!
        """
        # let strain act on the base vectors
        self.lattice.ApplyStrain(strain)
        # recalculate the reciprocal lattice
        self.rlattice = self.lattice.ReciprocalLattice()

    def GetMismatch(self, mat):
        """
        Calculate the mismatch strain between the material and a second
        material
        """
        raise NotImplementedError("XU.material.GetMismatch: "
                                  "not implemented yet")

    def distances(self):
        """
        function to obtain distances of atoms in the crystal up to the unit
        cell size (largest value of a,b,c is the cut-off)

        returns a list of tuples with distance d and number of occurence n
        [(d1,n1),(d2,n2),...]

        Note: if the base of the material is empty the list will be empty
        """

        if not self.lattice.base:
            return []

        cutoff = numpy.max((self.lattice.a, self.lattice.b, self.lattice.c))

        tmp_data = []

        for i in range(len(self.lattice.base)):
            at1 = self.lattice.base[i]
            for j in range(len(self.lattice.base)):
                at2 = self.lattice.base[j]
                dis = numpy.linalg.norm(self.lattice.GetPoint(at1[1] - at2[1]))
                dis2 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((1, 0, 0))))
                dis3 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((0, 1, 0))))
                dis4 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((0, 0, 1))))
                dis5 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((-1, 0, 0))))
                dis6 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((0, -1, 0))))
                dis7 = numpy.linalg.norm(self.lattice.GetPoint(
                    at1[1] - at2[1] + numpy.array((0, 0, -1))))
                distances = sorted([dis, dis2, dis3, dis4, dis5, dis6, dis7])

                for dis in distances:
                    if dis < cutoff:
                        tmp_data.append(dis)

        # sort the list and compress equal entries
        tmp_data.sort()

        self._distances = [0]
        self._dis_hist = [0]
        for dis in tmp_data:
            if numpy.round(dis - self._distances[-1],
                           int(numpy.abs(numpy.log10(config.EPSILON)))) == 0:
                self._dis_hist[-1] += 1
            else:
                self._distances.append(dis)
                self._dis_hist.append(1)

        # create return value
        ret = []
        for i in range(len(self._distances)):
            ret.append((self._distances[i], self._dis_hist[i]))

        return ret


def CubicElasticTensor(c11, c12, c44):
    """
    Assemble the 6x6 matrix of elastic constants for a cubic material from the
    three independent components of a cubic crystal

    Parameters
    ----------
     c11,c12,c44:   independent components of the elastic tensor of cubic
                    materials

    Returns
    -------
     6x6 matrix with elastic constants
    """
    m = numpy.zeros((6, 6), dtype=numpy.double)
    m[0, 0] = c11
    m[1, 1] = c11
    m[2, 2] = c11
    m[3, 3] = c44
    m[4, 4] = c44
    m[5, 5] = c44
    m[0, 1] = m[0, 2] = c12
    m[1, 0] = m[1, 2] = c12
    m[2, 0] = m[2, 1] = c12

    return m


def HexagonalElasticTensor(c11, c12, c13, c33, c44):
    """
    Assemble the 6x6 matrix of elastic constants for a hexagonal material from
    the five independent components of a hexagonal crystal

    Parameters
    ----------
     c11,c12,c13,c33,c44:   independent components of the elastic tensor of
                            a hexagonal material

    Returns
    -------
     6x6 matrix with elastic constants

    """
    m = numpy.zeros((6, 6), dtype=numpy.double)
    m[0, 0] = m[1, 1] = c11
    m[2, 2] = c33
    m[3, 3] = m[4, 4] = c44
    m[5, 5] = 0.5 * (c11 - c12)
    m[0, 1] = m[1, 0] = c12
    m[0, 2] = m[1, 2] = m[2, 0] = m[2, 1] = c13

    return m


def WZTensorFromCub(c11ZB, c12ZB, c44ZB):
    """
    Determines the hexagonal elastic tensor from the values of the cubic
    elastic tensor under the assumptions presented in Phys. Rev. B 6, 4546
    (1972), which are valid for the WZ <-> ZB polymorphs.

    Parameters
    ----------
     c11,c12,c44:   independent components of the elastic tensor of cubic
                    materials

    Returns
    -------
     6x6 matrix with elastic constants

    Implementation according to a patch submitted by Julian Stangl
    """
    # matrix conversions: cubic (111) to hexagonal (001) direction
    P = (1 / 6.) * numpy.array([[3, 3, 6],
                                [2, 4, 8],
                                [1, 5, -2],
                                [2, 4, -4],
                                [2, -2, 2],
                                [1, -1, 4]])
    Q = (1 / (3 * numpy.sqrt(2))) * numpy.array([1, -1, -2])

    cZBvec = numpy.array([c11ZB, c12ZB, c44ZB])
    cWZvec_BAR = numpy.dot(P, cZBvec)
    delta = numpy.dot(Q, cZBvec)
    D = numpy.array([delta**2 / cWZvec_BAR[2], 0, -delta**2 / cWZvec_BAR[2],
                     0, delta**2 / cWZvec_BAR[0], delta**2 / cWZvec_BAR[2]])
    cWZvec = cWZvec_BAR - D.T

    return HexagonalElasticTensor(cWZvec[0], cWZvec[2], cWZvec[3],
                                  cWZvec[1], cWZvec[4])


# define general material useful for peak position calculations
def GeneralUC(a=4, b=4, c=4, alpha=90, beta=90, gamma=90,
              name="General", base=None):
    """
    general material with primitive unit cell but possibility for different
    a,b,c and alpha,beta,gamma

    Parameters
    ----------
     a,b,c      unit cell extenstions (Angstrom)
     alpha      angle between unit cell vectors b,c
     beta       angle between unit cell vectors a,c
     gamma      angle between unit cell vectors a,b
     base       instance of LatticeBase

    returns a Crystal object with the specified properties
    """
    return Crystal(name, lattice.TriclinicLattice(
                   a, b, c, alpha, beta, gamma, base=base))


class Alloy(Crystal):

    def __init__(self, matA, matB, x):
        lat = lattice.Lattice(matA.a1, matA.a2, matA.a3,
                              base=matA.lattice.base)
        super(Alloy, self).__init__("None", lat, matA.cij)
        self.matA = matA
        self.matB = matB
        self._setxb(x)

    @staticmethod
    def lattice_const_AB(latA, latB, x):
        """
        method to set the composition of the Alloy.
        x is the atomic fraction of the component B
        """
        return (latB - latA) * x + latA

    def _getxb(self):
        return self._xb

    def _setxb(self, x):
        self._xb = x
        self.name = ("%s(%2.2f)%s(%2.2f)"
                     % (self.matA.name, 1. - x, self.matB.name, x))
        # modify the lattice
        self.lattice.a1 = self.lattice_const_AB(self.matA.a1, self.matB.a1, x)
        self.lattice.a2 = self.lattice_const_AB(self.matA.a2, self.matB.a2, x)
        self.lattice.a3 = self.lattice_const_AB(self.matA.a3, self.matB.a3, x)
        self.rlattice = self.lattice.ReciprocalLattice()

        # set elastic constants
        self.cij = (self.matB.cij - self.matA.cij) * x + self.matA.cij
        self.cijkl = (self.matB.cijkl - self.matA.cijkl) * x + self.matA.cijkl

    x = property(_getxb, _setxb)

    def _checkfinitenumber(self, arg, name=""):
        if isinstance(arg, numbers.Number) and numpy.isfinite(arg):
            return float(arg)
        else:
            raise TypeError("argument (%s) must be a scalar!" % name)

    def _checkarray(self, arg, name=""):
        if isinstance(arg, (list, tuple, numpy.ndarray)):
            return numpy.array(arg, dtype=numpy.double)
        else:
            raise TypeError("argument (%s) must be of type "
                            "list, tuple or numpy.ndarray" % name)

    def _definehelpers(self, hkl, cijA, cijB):
        """
        define helper functions for solving the content from reciprocal space
        positions
        """
        def a1(x):
            return self.lattice_const_AB(self.matA.a1, self.matB.a1, x)

        def a2(x):
            return self.lattice_const_AB(self.matA.a2, self.matB.a2, x)

        def a3(x):
            return self.lattice_const_AB(self.matA.a3, self.matB.a3, x)

        def V(x):
            return numpy.dot(a3(x), numpy.cross(a1(x), a2(x)))

        def b1(x):
            return 2 * numpy.pi / V(x) * numpy.cross(a2(x), a3(x))

        def b2(x):
            return 2 * numpy.pi / V(x) * numpy.cross(a3(x), a1(x))

        def b3(x):
            return 2 * numpy.pi / V(x) * numpy.cross(a1(x), a2(x))

        def qhklx(x):
            return hkl[0] * b1(x) + hkl[1] * b2(x) + hkl[2] * b3(x)

        def frac(x):
            return ((cijB[0, 2] + cijB[1, 2] - (cijA[0, 2] + cijA[1, 2])) * x +
                    (cijA[0, 2] + cijA[1, 2])) / \
                   ((cijB[2, 2] - cijA[2, 2]) * x + cijA[2, 2])

        return a1, a2, a3, V, b1, b2, b3, qhklx, frac

    def RelaxationTriangle(self, hkl, sub, exp):
        """
        function which returns the relaxation trianlge for a
        Alloy of given composition. Reciprocal space coordinates are
        calculated using the user-supplied experimental class

        Parameters
        ----------
         hkl : Miller Indices
         sub : substrate material or lattice constant (Instance of Crystal
               class or float)
         exp : Experiment class from which the Transformation object and ndir
               are needed

        Returns
        -------
         qy,qz : reciprocal space coordinates of the corners of the relaxation
                 triangle

        """
        hkl = self._checkarray(hkl, "hkl")
        trans = exp._transform
        ndir = exp.ndir / numpy.linalg.norm(exp.ndir)

        if isinstance(sub, Crystal):
            asub = sub.lattice.a
        elif isinstance(sub, float):
            asub = sub
        else:
            raise TypeError("Second argument (sub) must be of type float or "
                            "an instance of xrayutilities.materials.Crystal")

        # test if inplane direction of hkl is the same as the one for the
        # experiment otherwise warn the user
        hklinplane = numpy.cross(numpy.cross(exp.ndir, hkl), exp.ndir)
        if (numpy.linalg.norm(numpy.cross(hklinplane, exp.idir)) >
                config.EPSILON):
            warnings.warn("Alloy: given hkl differs from the geometry of the "
                          "Experiment instance in the azimuthal direction")

        # transform elastic constants to correct coordinate frame
        cijA = Cijkl2Cij(trans(self.matA.cijkl, rank=4))
        cijB = Cijkl2Cij(trans(self.matB.cijkl, rank=4))

        a1, a2, a3, V, b1, b2, b3, qhklx, frac = self._definehelpers(hkl,
                                                                     cijA,
                                                                     cijB)

        qr_i = numpy.abs(trans(qhklx(self.x))[1])
        qr_p = numpy.abs(trans(qhklx(self.x))[2])
        qs_i = 2 * numpy.pi / asub * numpy.linalg.norm(numpy.cross(ndir, hkl))
        qs_p = 2 * numpy.pi / asub * numpy.abs(numpy.dot(ndir, hkl))

        # calculate pseudomorphic points for A and B
        def abulk(x):
            return numpy.linalg.norm(a1(x))

        def aperp(x):
            return abulk(self.x) * (1 + frac(x) * (1 - asub / abulk(self.x)))

        qp_i = 2 * numpy.pi / asub * numpy.linalg.norm(numpy.cross(ndir, hkl))
        qp_p = 2 * numpy.pi / aperp(self.x) * numpy.abs(numpy.dot(ndir, hkl))

        # assembly return values
        qy = numpy.array([qr_i, qp_i, qs_i, qr_i], dtype=numpy.double)
        qz = numpy.array([qr_p, qp_p, qs_p, qr_p], dtype=numpy.double)

        return qy, qz


class CubicAlloy(Alloy):

    def __init__(self, matA, matB, x):
        # here one could check if material is really cubic!!
        Alloy.__init__(self, matA, matB, x)

    def ContentBsym(self, q_perp, hkl, inpr, asub, relax):
        """
        function that determines the content of B
        in the alloy from the reciprocal space position
        of a symetric peak. As an additional input the substrates
        lattice parameter and the degree of relaxation must be given

        Parameters
        ----------
         q_perp : perpendicular peak position of the reflection
                  hkl of the alloy in reciprocal space
         hkl : Miller indices of the measured symmetric reflection (also
               defines the surface normal
         inpr : Miller indices of a Bragg peak defining the inplane reference
                direction
         asub:   substrate lattice constant
         relax:  degree of relaxation (needed to obtain the content from
                 symmetric reciprocal space position)

        Returns
        -------
         content : the content of B in the alloy determined from the input
                   variables

        """

        # check input parameters
        q_perp = self._checkfinitenumber(q_perp, "q_perp")
        hkl = self._checkarray(hkl, "hkl")
        inpr = self._checkarray(inpr, "inpr")
        asub = self._checkfinitenumber(asub, "asub")
        relax = self._checkfinitenumber(relax, "relax")

        # calculate lattice constants from reciprocal space positions
        n = self.rlattice.GetPoint(hkl) / \
            numpy.linalg.norm(self.rlattice.GetPoint(hkl))
        q_hkl = self.rlattice.GetPoint(hkl)
        # the following line is not generally true! only cubic materials
        aperp = 2 * numpy.pi / q_perp * numpy.abs(numpy.dot(n, hkl))

        # transform the elastic tensors to a coordinate frame attached to the
        # surface normal
        inp1 = numpy.cross(n, inpr) / numpy.linalg.norm(numpy.cross(n, inpr))
        inp2 = numpy.cross(n, inp1)
        trans = math.CoordinateTransform(inp1, inp2, n)

        if config.VERBOSITY >= config.DEBUG:
            print("XU.materials.Alloy.ContentB: inp1/inp2: ", inp1, inp2)
        cijA = Cijkl2Cij(trans(self.matA.cijkl, rank=4))
        cijB = Cijkl2Cij(trans(self.matB.cijkl, rank=4))

        a1, a2, a3, V, b1, b2, b3, qhklx, frac = self._definehelpers(hkl,
                                                                     cijA,
                                                                     cijB)

        # the following line is not generally true! only cubic materials
        def abulk_perp(x):
            return numpy.abs(2 * numpy.pi / numpy.inner(qhklx(x), n) *
                             numpy.inner(n, hkl))

        # can we use abulk_perp here? for cubic materials this should work?!
        def ainp(x):
            return asub + relax * (abulk_perp(x) - asub)

        if config.VERBOSITY >= config.DEBUG:
            print("XU.materials.Alloy.ContentB: abulk_perp: %8.5g"
                  % (abulk_perp(0.)))

        def equation(x):
            return ((aperp - abulk_perp(x)) +
                    (ainp(x) - abulk_perp(x)) * frac(x))

        x = scipy.optimize.brentq(equation, -0.1, 1.1)

        return x

    def ContentBasym(self, q_inp, q_perp, hkl, sur):
        """
        function that determines the content of B
        in the alloy from the reciprocal space position
        of an asymmetric peak.

        Parameters
        ----------
         q_inp : inplane peak position of reflection hkl of
                 the alloy in reciprocal space
         q_perp : perpendicular peak position of the reflection
                  hkl of the alloy in reciprocal space
         hkl : Miller indices of the measured asymmetric reflection
         sur : Miller indices of the surface (determines the perpendicular
               direction)

        Returns
        -------
         content, [a_inplane, a_perp, a_bulk_perp(x), eps_inplane, eps_perp]:
                the content of B in the alloy determined from the input
                variables and the lattice constants calculated from the
                reciprocal space positions as well as the strain (eps) of the
                layer
        """

        # check input parameters
        q_inp = self._checkfinitenumber(q_inp, "q_inp")
        q_perp = self._checkfinitenumber(q_perp, "q_perp")
        hkl = self._checkarray(hkl, "hkl")
        sur = self._checkarray(sur, "sur")

        # check if reflection is asymmetric
        if numpy.linalg.norm(
                numpy.cross(self.rlattice.GetPoint(hkl),
                            self.rlattice.GetPoint(sur))) < 1.e-8:
            raise InputError("Miller indices of a symmetric reflection were"
                             "given where an asymmetric reflection is needed")

        # calculate lattice constants from reciprocal space positions
        n = self.rlattice.GetPoint(sur) / \
            numpy.linalg.norm(self.rlattice.GetPoint(sur))
        q_hkl = self.rlattice.GetPoint(hkl)
        # the following two lines are not generally true! only cubic materials
        ainp = 2 * numpy.pi / q_inp * numpy.linalg.norm(numpy.cross(n, hkl))
        aperp = 2 * numpy.pi / q_perp * numpy.abs(numpy.dot(n, hkl))

        # transform the elastic tensors to a coordinate frame attached to the
        # surface normal
        inp1 = numpy.cross(n, q_hkl) / numpy.linalg.norm(numpy.cross(n, q_hkl))
        inp2 = numpy.cross(n, inp1)
        trans = math.CoordinateTransform(inp1, inp2, n)

        cijA = Cijkl2Cij(trans(self.matA.cijkl, rank=4))
        cijB = Cijkl2Cij(trans(self.matB.cijkl, rank=4))

        a1, a2, a3, V, b1, b2, b3, qhklx, frac = self._definehelpers(hkl,
                                                                     cijA,
                                                                     cijB)

        # the following two lines are not generally true! only cubic materials
        def abulk_inp(x):
            return numpy.abs(2 * numpy.pi / numpy.inner(qhklx(x), inp2) *
                             numpy.linalg.norm(numpy.cross(n, hkl)))

        def abulk_perp(x):
            return numpy.abs(2 * numpy.pi / numpy.inner(qhklx(x), n) *
                             numpy.inner(n, hkl))

        if config.VERBOSITY >= config.DEBUG:
            print("XU.materials.Alloy.ContentB: abulk_inp/perp: %8.5g %8.5g"
                  % (abulk_inp(0.), abulk_perp(0.)))

        def equation(x):
            return ((aperp - abulk_perp(x)) +
                    (ainp - abulk_inp(x)) * frac(x))

        x = scipy.optimize.brentq(equation, -0.1, 1.1)

        eps_inplane = (ainp - abulk_perp(x)) / abulk_perp(x)
        eps_perp = (aperp - abulk_perp(x)) / abulk_perp(x)

        return x, [ainp, aperp, abulk_perp(x), eps_inplane, eps_perp]


def PseudomorphicMaterial(sub, layer, relaxation=0, trans=None):
    """
    This function returns a material whos lattice is pseudomorphic on a
    particular substrate material. The two materials must have similar unit
    cell definitions for the algorithm to work correctly, i.e. it does not work
    for combiniations of materials with different lattice symmetry.

    Parameters
    ----------
     sub:         substrate material
     layer:       bulk material of the layer
     relaxation:  degree of relaxation 0: pseudomorphic, 1: relaxed
                  (default: 0)
     trans:       Transformation which transforms lattice directions into a
                  surface orientated coordinate frame (x,y inplane, z out of
                  plane). If None a (001) surface geometry of a cubic material
                  is assumed.

    Returns
    -------
    An instance of Crystal holding the new pseudomorphically
    strained material.
    """
    def get_inplane(lat):
        """determine inplane lattice parameter"""
        return (math.VecNorm(lat.GetPoint(trans.inverse((1, 0, 0)))) +
                math.VecNorm(lat.GetPoint(trans.inverse((0, 1, 0))))) / 2.

    if not trans:
        trans = math.Transform(numpy.identity(3))

    slat = sub.lattice
    llat = layer.lattice
    # calculate the strain
    asub = get_inplane(sub.lattice)
    abulk = get_inplane(layer.lattice)
    apar = asub + (abulk - asub) * relaxation
    epar = (apar - abulk) / abulk
    cT = trans(layer.cijkl, rank=4)

    eperp = -epar * (cT[1, 1, 2, 2] + cT[2, 2, 0, 0]) / (cT[2, 2, 2, 2])
    eps = trans.inverse(numpy.diag((epar, epar, eperp)), rank=2)

    # create the pseudomorphic material
    pmlatt = lattice.Lattice(layer.a1, layer.a2, layer.a3,
                             base=layer.lattice.base)
    pmat = Crystal(layer.name, pmlatt, layer.cij)
    pmat.ApplyStrain(eps)

    return pmat
