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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>
import abc
import collections.abc
import copy
import warnings

import numpy
from scipy.constants import physical_constants
from scipy.misc import derivative

from .. import materials, utilities
from ..math import heaviside
from .models import LayerModel


def getit(it, key):
    """
    generator to obtain items from nested iterable
    """
    for elem in it:
        if isinstance(elem, collections.abc.Iterable):
            if key in elem:
                yield elem[key]
            else:
                for subelem in getit(elem, key):
                    yield subelem


def getfirst(iterable, key):
    """
    helper function to obtain the first item in a nested iterable
    """
    return next(getit(iterable, key))


def GradedBuffer(xfrom, xto, nsteps, thickness, relaxation=1):
    """
    create a multistep graded composition buffer.

    Parameters
    ----------
    xfrom :     float
        begin of the composition gradient
    xto :       float
        end of the composition gradient
    nsteps :    int
        number of steps of the gradient
    thickness : float
        total thickness of the Buffer in A
    relaxation : float
        relaxation of the buffer

    Returns
    -------
    list
        layer list needed for the Darwin model simulation
    """
    subthickness = thickness/nsteps
    gradedx = numpy.linspace(xfrom, xto, nsteps)
    layerlist = []
    for x in gradedx:
        layerlist.append({'t': subthickness, 'x': x, 'r': relaxation})
    return layerlist


class DarwinModel(LayerModel):
    """
    model class inmplementing the basics of the Darwin theory for layers
    materials.  This class is not fully functional and should be used to derive
    working models for particular material systems.

    To make the class functional the user needs to implement the
    init_structurefactors() and _calc_mono() methods
    """
    ncalls = 0

    def __init__(self, qz, qx=0, qy=0, **kwargs):
        """
        constructor of the model class. The arguments consist of basic
        parameters which are needed to prepare the calculation of the model.

        Parameters
        ----------
        qz :        array-like
            momentum transfer values for the calculation
        qx, qy :    float, optional
            inplane momentum transfer (not implemented!)
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        resolution_width :  float, optional
            width of the resolution function (deg)
        polarization : {'S', 'P', 'both'}
            polarization of the x-ray beam. If set to 'both' also Cmono, the
            polarization factor of the monochromator should be set
        experiment : Experiment, optional
            experiment class containing geometry and energy of the experiment.
        Cmono :     float, optional
            polarization factor of the monochromator
        energy :    float or str, optional
            x-ray energy in eV
        """
        self.polarization = kwargs.pop('polarization', 'S')
        exp = kwargs.pop('experiment', None)
        self.Cmono = kwargs.pop('Cmono', 1)
        super().__init__(exp, **kwargs)

        self.npoints = len(qz)
        self.qz = numpy.asarray(qz)
        self.qinp = (qx, qy)
        if self.qinp != (0, 0):
            raise NotImplementedError('asymmetric CTR simulation is not yet '
                                      'supported -> approach the authors')
        self.init_structurefactors()
        # initialize coplanar geometry
        k = self.exp.k0
        qv = numpy.asarray([(qx, qy, q) for q in self.qz])
        Q = numpy.linalg.norm(qv, axis=1)
        theta = numpy.arcsin(Q / (2 * k))
        domega = numpy.arctan2(numpy.sqrt(qx**2 + qy**2), self.qz)
        self.alphai, self.alphaf = (theta + domega, theta - domega)
        # polarization factor
        self.C = {'S': numpy.ones(len(self.qz)),
                  'P': numpy.abs(numpy.cos(self.alphai + self.alphaf))}

    def init_structurefactors(self):
        """
        calculates the needed atomic structure factors
        """
        pass

    def _calc_mono(self, pdict, pol):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
        pdict :     dict
            property dictionary, contains all the properties for the structure
            factor calculation
        pol :       {'S', 'P'}
            polarization of the x-rays; sigma or pi

        Returns
        -------
        r, rbar, t :    float or array-like
            reflection, backside reflection, and tranmission coefficients
        """
        pass

    def _calc_double(self, ra, rabar, ta, rb, rbbar, tb, d):
        """
        calculate reflection coefficients for the double layer from the
        reflection values of the single layers

        Parameters
        ----------
        ra, rabar, ta : float or array-like
            reflection, backside reflection, and transmission coefficients of
            layer A
        rb, rbbar, tb : float or array-like
            same for layer B
        d :             float
            distance between the layers

        Returns
        -------
        r, rbar, t :    float or array-like
            reflection, backside reflection, and tranmission coefficients
        """
        self.ncalls += 1
        e = numpy.exp(-1j*self.qz*d)
        eh = numpy.exp(-1j*self.qz*d/2)
        denom = (1-rabar*rb*e)
        rab = ra + rb*(ta*ta*e)/denom
        rabbar = rbbar + rabar*(tb*tb*e)/(1-rbbar*ra*e)
        tab = ta*tb*eh/denom
        return rab, rabbar, tab

    def simulate(self, ml):
        """
        main simulation function for the Darwin model. will calculate the
        reflected intensity

        Parameters
        ----------
        ml :        iterable
            monolayer sequence of the sample. This should be created with the
            function make_monolayer(). see its documentation for details
        """
        self.ncalls = 0
        Ih = {'S': numpy.zeros(len(self.qz)), 'P': numpy.zeros(len(self.qz))}
        geomfact = heaviside(self.alphai) * heaviside(self.alphaf)
        for pol in self.get_polarizations():
            r, rbar, t = (numpy.zeros(self.npoints, dtype=numpy.complex),
                          numpy.zeros(self.npoints, dtype=numpy.complex),
                          numpy.ones(self.npoints, dtype=numpy.complex))
            for nrep, subml in ml:
                r, rbar, t = self._recur_sim(nrep, subml, r, rbar, t, pol)
            self.r, self.rbar, self.t = r, rbar, t
            Ih[pol] = numpy.abs(self.r)**2 * geomfact

        ret = self.join_polarizations(Ih['S'], Ih['P'])
        return self._create_return(self.qz, numpy.sqrt(ret))

    def _recur_sim(self, nrep, ml, r, rbar, t, pol):
        """
        recursive simulation function for the calculation of the reflected,
        backside reflected and transmitted wave factors (internal)

        Parameters
        ----------
        ml :        iterable
            monolayer sequence of the calculation block. This should be created
            with the function make_monolayer(). see its documentation for
            details
        r :         float or array-like
            reflection factor of the upper layers (needed for the recursion)
        rbar :      float or array-like
            back-side reflection factor of the upper layers (needed for the
            recursion)
        t :         float or array-like
            transmission factor of the upper layers (needed for the recursion)
        pol :       {'S', 'P'}
            polarization of the x-rays

        Returns
        -------
        r, rbar, t : float or array-like
            reflection and transmission of the full stack
        """
        if isinstance(ml, list):
            rm, rmbar, tm = (None, None, None)
            for nsub, subml in ml:
                rm, rmbar, tm = self._recur_sim(nsub, subml, rm,
                                                rmbar, tm, pol)
            d = getfirst(ml, 'd')
        else:
            rm, rmbar, tm = self._calc_mono(ml, pol)
            d = ml['d']

        Nmax = int(numpy.log(nrep) / numpy.log(2)) + 1
        for i in range(Nmax):
            if r is None:
                r, rbar, t = rm, rmbar, tm
            elif (nrep // (2**i)) % 2 == 1:
                r, rbar, t = self._calc_double(r, rbar, t, rm, rmbar, tm, d)
            rm, rmbar, tm = self._calc_double(rm, rmbar, tm, rm, rmbar, tm, d)

        return r, rbar, t


class DarwinModelAlloy(DarwinModel, utilities.ABC):
    """
    extension of the DarwinModel for an binary alloy system were one parameter
    is used to determine the chemical composition

    To make the class functional the user needs to implement the
    get_dperp_apar() method and define the substrate lattice parameter (asub).
    See the DarwinModelSiGe001 class for an implementation example.
    """
    @abc.abstractmethod
    def get_dperp_apar(self, x, apar, r=1):
        """
        calculate inplane lattice parameter and the out of plane lattice plane
        spacing (of the atomic planes!) from composition and relaxation.

        Parameters
        ----------
        x :         float
            chemical composition parameter
        apar :      float
            inplane lattice parameter of the material below the current layer
            (onto which the present layer is strained to). This value also
            served as a reference for the relaxation parameter.
        r :         float
            relaxation parameter. 1=relaxed, 0=pseudomorphic

        Returns
        -------
        dperp :     float
        apar :      float
        """
        raise NotImplementedError("abstract method needs to be overwritten")

    def make_monolayers(self, s):
        """
        create monolayer sequence from layer list

        Parameters
        ----------
        s :     list
            layer model. list of layer dictionaries including possibility to
            form superlattices. As an example 5 repetitions of a
            Si(10nm)/Ge(15nm) superlattice on Si would like like:

            >>> s = [(5, [{'t': 100, 'x': 0, 'r': 0},
            >>>           {'t': 150, 'x': 1, 'r': 0}]),
            >>>      {'t': 3500000, 'x': 0, 'r': 0}]

            the dictionaries must contain 't': thickness in A, 'x': chemical
            composition, and either 'r': relaxation or 'ai': inplane lattice
            parameter.
            Future implementations for asymmetric peaks might include layer
            type 'l' (not yet implemented). Already now any additional property
            in the dictionary will be handed on to the returned monolayer list.
        asub :  float
            inplane lattice parameter of the substrate

        Returns
        -------
        list
            monolayer list in a format understood by the simulate and
            xGe_profile methods
        """
        ml = []
        ai = self.asub
        for subl in copy.copy(s):
            ml, ai = self._recur_makeml(subl, ml, ai)
        return ml

    def _recur_makeml(self, s, ml, apar):
        """
        recursive creation of a monolayer structure (internal)

        Parameters
        ----------
        s :     list
            layer model. list of layer dictionaries
        ml :    list
            list of layers below what should be added (needed for recursion)
        apar :  float
            inplane lattice parameter of the current surface

        Returns
        -------
        list
            monolayer list in a format understood by the simulate and
            prop_profile methods
        """

        if isinstance(s, tuple):
            nrep, sd = s
            if isinstance(sd, dict):
                sd = [sd, ]
            if any([r > 0 for r in getit(sd, 'r')]):  # if relaxation
                for i in range(nrep):
                    for subsd in sd:
                        ml, apar = self._recur_makeml(subsd, ml, apar=apar)
            else:  # no relaxation in substructure
                subl = []
                for subsd in sd:
                    subl, apar = self._recur_makeml(subsd, subl, apar=apar)
                ml.insert(0, (nrep, subl))
        elif isinstance(s, dict):
            x = s.pop('x')
            if callable(x):  # composition profile in layer
                t = 0
                T = s.pop('t')
                if 'r' in s:
                    if s['r'] > 0:
                        warnings.warn(
                            """relaxation for composition gradient may yield
                            weird lattice parameter variation! Consider
                            supplying the inplane lattice parameter 'ai'
                            directly!""")
                while t < T:
                    if 'r' in s:
                        r = abs(derivative(x, t, dx=1.4, n=1))*s['r']
                        dperp, apar = self.get_dperp_apar(x(t), apar, r)
                    else:
                        dperp, apar = self.get_dperp_apar(x(t), s['ai'])
                    t += dperp
                    d = copy.copy(s)
                    d.pop('r')
                    d.update({'d': dperp, 'x': x(t), 'ai': apar})
                    ml.insert(0, (1, d))
            else:  # constant composition layer
                if 'r' in s:
                    dperp, apar = self.get_dperp_apar(x, apar, s.pop('r'))
                else:
                    dperp, apar = self.get_dperp_apar(x, s.pop('ai'))
                nmono = int(numpy.ceil(s['t']/dperp))
                s.update({'d': dperp, 'x': x, 'ai': apar})
                ml.insert(0, (nmono, s))
        else:
            raise Exception('wrong type (%s) of sublayer, must be tuple or'
                            ' dict' % (type(s)))
        return ml, apar

    def prop_profile(self, ml, prop):
        """
        calculate the profile of chemical composition or inplane lattice
        spacing from a monolayer list. One value for each monolayer in the
        sample is returned.

        Parameters
        ----------
        ml :    list
            monolayer list created by make_monolayer()
        prop :  str
            name of the property which should be evaluated. Use 'x' for the
            chemical composition and 'ai' for the inplane lattice parameter.

        Returns
        -------
        zm :    ndarray
            z-position, z-0 is the surface
        propx : ndarray
            value of the property prop for every monolayer
        """

        def startinterval(start, inter, N):
            return numpy.arange(start, start+inter*(N+0.5), inter)

        def _recur_prop(nrep, ml, zp, propx, propn):
            if isinstance(ml, list):
                lzp, lprop = ([], [])
                for nreps, subml in ml:
                    lzp, lprop = _recur_prop(nreps, subml, lzp, lprop, propn)
                d = getfirst(ml, 'd')
            else:
                lzp = -ml['d']
                lprop = ml[propn]
                d = ml['d']

            Nmax = int(numpy.log(nrep) / numpy.log(2)) + 1
            for i in range(Nmax):
                if (nrep // (2**i)) % 2 == 1:
                    try:
                        curzp = zp[-1]
                    except IndexError:
                        curzp = 0.0
                    zp = numpy.append(zp, lzp+curzp)
                    propx = numpy.append(propx, lprop)
                try:
                    curlzp = lzp[-1]
                except IndexError:
                    curlzp = lzp
                lzp = numpy.append(lzp, lzp+curlzp)
                lprop = numpy.append(lprop, lprop)
            return zp, propx

        zm = []
        propx = []
        for nrep, subml in ml:
            zm, propx = _recur_prop(nrep, subml, zm, propx, prop)
        return zm, propx


class DarwinModelSiGe001(DarwinModelAlloy):
    """
    model class implementing the Darwin theory of diffraction for SiGe layers.
    The model is based on separation of the sample structure into building
    blocks of atomic planes from which a multibeam dynamical model is
    calculated.
    """
    Si = materials.Si
    Ge = materials.Ge
    eSi = materials.elements.Si
    eGe = materials.elements.Ge
    aSi = materials.Si.a1[0]
    asub = aSi  # needed for the make_monolayer function
    re = physical_constants['classical electron radius'][0] * 1e10

    @classmethod
    def abulk(cls, x):
        """
        calculate the bulk (relaxed) lattice parameter of the alloy
        """
        return cls.aSi + (0.2 * x + 0.027 * x ** 2)

    @staticmethod
    def poisson_ratio(x):
        """
        calculate the Poisson ratio of the alloy
        """
        return 2 * (63.9-15.6*x) / (165.8-37.3*x)  # according to IOFFE

    @classmethod
    def get_dperp_apar(cls, x, apar, r=1):
        """
        calculate inplane lattice parameter and the out of plane lattice plane
        spacing (of the atomic planes!) from composition and relaxation

        Parameters
        ----------
        x :     float
            chemical composition parameter
        apar :  float
            inplane lattice parameter of the material below the current layer
            (onto which the present layer is strained to). This value also
            served as a reference for the relaxation parameter.
        r :     float, optional
            relaxation parameter. 1=relaxed, 0=pseudomorphic

        Returns
        -------
        dperp : float
            perpendicular d-spacing
        apar :  float
            inplane lattice parameter
        """
        abulk = cls.abulk(x)
        aparl = apar + (abulk - apar) * r
        dperp = abulk*(1+cls.poisson_ratio(x)*(1-aparl/abulk))/4.
        return dperp, aparl

    def init_structurefactors(self, temp=300):
        """
        calculates the needed atomic structure factors

        Parameters
        ----------
        temp :  float, optional
            temperature used for the Debye model
        """
        en = self.exp.energy
        q = numpy.sqrt(self.qinp[0]**2 + self.qinp[1]**2 + self.qz**2)
        self.fSi = self.eSi.f(q, en) * self.Si._debyewallerfactor(temp, q)
        self.fGe = self.eGe.f(q, en) * self.Ge._debyewallerfactor(temp, q)
        self.fSi0 = self.eSi.f(0, en)
        self.fGe0 = self.eGe.f(0, en)

    def _calc_mono(self, pdict, pol):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
        pdict :     dict
            property dictionary, contains the layer properties:
            'x':   Ge-content of the layer (0: Si, 1: Ge);
            'l':   index of the layer in the unit cell (0, 1, 2, 3). important
                   for asymmetric peaks only!
        pol :       {'S', 'P'}
            polarization of the x-rays

        Returns
        -------
        r, rbar, t : float or array-like
            reflection, backside reflection, and tranmission coefficients
        """
        ainp = pdict.get('ai')
        xGe = pdict.get('x')
        # pre-factor for reflection: contains footprint correction
        gamma = 4*numpy.pi*self.re/(self.qz*ainp**2)
#        ltype = pdict.get('l', 0)
#        if ltype == 0: # for asymmetric peaks (not yet implemented)
#            p1, p2 = (0, 0), (0.5, 0.5)
#        elif ltype == 1:
#            p1, p2 = (0.25, 0.25), (0.75, 0.75)
#        elif ltype == 2:
#            p1, p2 = (0.5, 0.), (0., 0.5)
#        elif ltype == 3:
#            p1, p2 = (0.75, 0.25), (0.25, 0.75)

        r = -1j*gamma * self.C[pol] * (self.fSi+(self.fGe-self.fSi)*xGe) * 2
        # * (exp(1j*(h*p1[0]+k*p1[1])) + exp(1j*(h*p1[0]+k*p1[1])))
        t = 1 + 1j*gamma * (self.fSi0+(self.fGe0-self.fSi0)*xGe) * 2
        return r, numpy.copy(r), t


class DarwinModelGaInAs001(DarwinModelAlloy):
    """
    Darwin theory of diffraction for Ga_{1-x} In_x As layers.
    The model is based on separation of the sample structure into building
    blocks of atomic planes from which a multibeam dynamical model is
    calculated.
    """
    GaAs = materials.GaAs
    InAs = materials.InAs
    eGa = materials.elements.Ga
    eIn = materials.elements.In
    eAs = materials.elements.As
    aGaAs = materials.GaAs.a1[0]
    asub = aGaAs  # needed for the make_monolayer function
    re = physical_constants['classical electron radius'][0] * 1e10

    @classmethod
    def abulk(cls, x):
        """
        calculate the bulk (relaxed) lattice parameter of the Ga_{1-x}In_{x}As
        alloy
        """
        return cls.aGaAs + 0.40505*x

    @staticmethod
    def poisson_ratio(x):
        """
        calculate the Poisson ratio of the alloy
        """
        return 2 * (4.54 + 0.8*x) / (8.34 + 3.56*x)  # according to IOFFE

    @classmethod
    def get_dperp_apar(cls, x, apar, r=1):
        """
        calculate inplane lattice parameter and the out of plane lattice plane
        spacing (of the atomic planes!) from composition and relaxation

        Parameters
        ----------
        x :     float
            chemical composition parameter
        apar :  float
            inplane lattice parameter of the material below the current layer
            (onto which the present layer is strained to). This value also
            served as a reference for the relaxation parameter.
        r :     float
            relaxation parameter. 1=relaxed, 0=pseudomorphic

        Returns
        -------
        dperp : float
            perpendicular d-spacing
        apar :  float
            inplane lattice parameter
        """
        abulk = cls.abulk(x)
        aparl = apar + (abulk - apar) * r
        dperp = abulk*(1+cls.poisson_ratio(x)*(1-aparl/abulk))/4.
        return dperp, aparl

    def init_structurefactors(self, temp=300):
        """
        calculates the needed atomic structure factors

        Parameters
        ----------
        temp :      float, optional
            temperature used for the Debye model
        """
        en = self.exp.energy
        q = numpy.sqrt(self.qinp[0]**2 + self.qinp[1]**2 + self.qz**2)
        fAs = self.eAs.f(q, en)
        self.fGaAs = (self.eGa.f(q, en) + fAs) \
            * self.GaAs._debyewallerfactor(temp, q)
        self.fInAs = (self.eIn.f(q, en) + fAs) \
            * self.InAs._debyewallerfactor(temp, q)
        self.fGaAs0 = self.eGa.f(0, en) + self.eAs.f(0, en)
        self.fInAs0 = self.eIn.f(0, en) + self.eAs.f(0, en)

    def _calc_mono(self, pdict, pol):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
        pdict :     dict
            property dictionary, contains the layer properties:
            'x':   In-content of the layer (0: GaAs, 1: InAs)
        pol :       {'S', 'P'}
            polarization of the x-rays

        Returns
        -------
        r, rbar, t : float or array-like
            reflection, backside reflection, and tranmission coefficients
        """
        ainp = pdict.get('ai')
        xInAs = pdict.get('x')
        # pre-factor for reflection: contains footprint correction
        gamma = 4*numpy.pi * self.re/(self.qz*ainp**2)
        r = -1j*gamma*self.C[pol]*(self.fGaAs+(self.fInAs-self.fGaAs)*xInAs)
        t = 1 + 1j*gamma * (self.fGaAs0+(self.fInAs0-self.fGaAs0)*xInAs)
        return r, numpy.copy(r), t


class DarwinModelAlGaAs001(DarwinModelAlloy):
    """
    Darwin theory of diffraction for Al_x Ga_{1-x} As layers.
    The model is based on separation of the sample structure into building
    blocks of atomic planes from which a multibeam dynamical model is
    calculated.
    """
    GaAs = materials.GaAs
    AlAs = materials.AlAs
    eGa = materials.elements.Ga
    eAl = materials.elements.Al
    eAs = materials.elements.As
    aGaAs = materials.GaAs.a1[0]
    asub = aGaAs  # needed for the make_monolayer function
    re = physical_constants['classical electron radius'][0] * 1e10

    @classmethod
    def abulk(cls, x):
        """
        calculate the bulk (relaxed) lattice parameter of the Al_{x}Ga_{1-x}As
        alloy
        """
        return cls.aGaAs + 0.0078*x

    @staticmethod
    def poisson_ratio(x):
        """
        calculate the Poisson ratio of the alloy
        """
        return 2 * (5.38+0.32*x) / (11.88+0.14*x)  # according to IOFFE

    @classmethod
    def get_dperp_apar(cls, x, apar, r=1):
        """
        calculate inplane lattice parameter and the out of plane lattice plane
        spacing (of the atomic planes!) from composition and relaxation

        Parameters
        ----------
        x :     float
            chemical composition parameter
        apar :  float
            inplane lattice parameter of the material below the current layer
            (onto which the present layer is strained to). This value also
            served as a reference for the relaxation parameter.
        r :     float
            relaxation parameter. 1=relaxed, 0=pseudomorphic

        Returns
        -------
        dperp : float
            perpendicular d-spacing
        apar :  float
            inplane lattice parameter
        """
        abulk = cls.abulk(x)
        aparl = apar + (abulk - apar) * r
        dperp = abulk*(1+cls.poisson_ratio(x)*(1-aparl/abulk))/4.
        return dperp, aparl

    def init_structurefactors(self, temp=300):
        """
        calculates the needed atomic structure factors

        Parameters
        ----------
        temp :  float, optional
            temperature used for the Debye model
        """
        en = self.exp.energy
        q = numpy.sqrt(self.qinp[0]**2 + self.qinp[1]**2 + self.qz**2)
        fAs = self.eAs.f(q, en)
        self.fGaAs = (self.eGa.f(q, en) + fAs) \
            * self.GaAs._debyewallerfactor(temp, q)
        self.fAlAs = (self.eAl.f(q, en) + fAs) \
            * self.AlAs._debyewallerfactor(temp, q)
        self.fGaAs0 = self.eGa.f(0, en) + self.eAs.f(0, en)
        self.fAlAs0 = self.eAl.f(0, en) + self.eAs.f(0, en)

    def _calc_mono(self, pdict, pol):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
        pdict :     dict
            property dictionary, contains the layer properties:
            'x':   Al-content of the layer (0: GaAs, 1: AlAs)
        pol :       {'S', 'P'}
            polarization of the x-rays

        Returns
        -------
        r, rbar, t : float or array-like
            reflection, backside reflection, and tranmission coefficients
        """
        ainp = pdict.get('ai')
        xAlAs = pdict.get('x')
        # pre-factor for reflection: contains footprint correction
        gamma = 4*numpy.pi * self.re/(self.qz*ainp**2)
        r = -1j*gamma*self.C[pol]*(self.fGaAs+(self.fAlAs-self.fGaAs)*xAlAs)
        t = 1 + 1j*gamma * (self.fGaAs0+(self.fAlAs0-self.fGaAs0)*xAlAs)
        return r, numpy.copy(r), t
