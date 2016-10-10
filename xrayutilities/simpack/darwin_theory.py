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

import collections
import numpy
from scipy.constants import physical_constants
from scipy.misc import derivative

from . import LayerModel
from .. import materials


def getit(it, key):
    """
    generator to obtain items from nested iterable
    """
    for elem in it:
        if isinstance(elem, collections.Iterable):
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
     xfrom:     begin of the composition gradient
     xto:       end of the composition gradient
     nsteps:    number of steps of the gradient
     thickness: total thickness of the Buffer in A
     relaxation:    relaxation of the buffer

    Returns
    -------
     layer object needed for the Darwin model simulation
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
    def __init__(self, qz, qx=0, qy=0, **kwargs):
        """
        constructor of the model class. The arguments consist of basic
        parameters which are needed to prepare the calculation of the model.

        Parameters
        ----------
         qz:        momentum transfer values for the calculation
         qx,qy:     inplane momentum transfer (not implemented!)
         *kwargs:   optional parameters for the simulation. supported are:
            'experiment': Experiment class containing geometry and energy of
                          the experiment.
        """
        exp = kwargs.pop('experiment', None)
        super(LayerModel, self).__init__(exp, **kwargs)
        self.npoints = len(qz)
        self.qz = qz
        self.qinp = (qx, qy)
        if self.qinp != (0, 0):
            raise NotImplementedError('asymmetric CTR simulation is not yet '
                                      'supported -> approach the authors')
        self.init_structurefactors()
        self.ncalls = 0

    def init_structurefactors(self):
        """
        calculates the needed atomic structure factors
        """
        pass

    def _calc_mono(self, pdict):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
         pdict: property dictionary, contains all the properties for the
                structure factor calculation

        Returns
        -------
         r, rbar, t: reflection, backside reflection, and tranmission
                     coefficients
        """
        pass

    def _calc_double(self, ra, rabar, ta, rb, rbbar, tb, d):
        """
        calculate reflection coefficients for the double layer from the
        reflection values of the single layers

        Parameters
        ----------
         ra, rabar, ta: relfection, backside reflection, and transmission
                        coefficients of layer A
         rb, rbbar, tb: same for layer B
         d:             distance between the layers
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
         ml:        monolayer sequence of the sample. This should be created
                    with the function make_monolayer(). see its documentation
                    for details
        """
        self.ncalls = 0
        r, rbar, t = (numpy.zeros(self.npoints, dtype=numpy.complex),
                      numpy.zeros(self.npoints, dtype=numpy.complex),
                      numpy.ones(self.npoints, dtype=numpy.complex))
        for nrep, subml in ml:
            r, rbar, t = self._recur_sim(nrep, subml, r, rbar, t)
        self.r, self.rbar, self.t = r, rbar, t
        return self._create_return(self.qz, self.r)

    def _recur_sim(self, nrep, ml, r, rbar, t):
        """
        recursive simulation function for the calculation of the reflected,
        backside reflected and transmitted wave factors (internal)

        Parameters
        ----------
         ml:        monolayer sequence of the calculation block. This should be
                    created with the function make_monolayer(). see its
                    documentation for details
         r:         reflection factor of the upper layers (needed for the
                    recursion)
         rbar:      back-side reflection factor of the upper layers (needed for
                    the recursion)
         t:         transmission factor of the upper layers (needed for the
                    recursion)

        Returns
        -------
         r, rbar, t: reflection and transmission of the full stack
        """
        if isinstance(ml, list):
            rm, rmbar, tm = (numpy.zeros(self.npoints, dtype=numpy.complex),
                             numpy.zeros(self.npoints, dtype=numpy.complex),
                             numpy.ones(self.npoints, dtype=numpy.complex))
            for nsub, subml in ml:
                rm, rmbar, tm = self._recur_sim(nsub, subml, rm, rmbar, tm)
            d = getfirst(ml, 'd')
        else:
            rm, rmbar, tm = self._calc_mono(ml)
            d = ml['d']

        Nmax = int(numpy.log(nrep) / numpy.log(2)) + 1
        for i in range(Nmax):
            if (nrep // (2**i)) % 2 == 1:
                r, rbar, t = self._calc_double(r, rbar, t, rm, rmbar, tm, d)
            rm, rmbar, tm = self._calc_double(rm, rmbar, tm, rm, rmbar, tm, d)

        return r, rbar, t


class DarwinModelSiGe001(DarwinModel):
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
    C = 1  # polarization factor (needs to be implemented)
    re = physical_constants['classical electron radius'][0] * 1e10

    def init_structurefactors(self, temp=300):
        """
        calculates the needed atomic structure factors

        Parameters (optional)
        ---------------------
         temp:      temperature used for the Debye model
        """
        en = self.exp.energy
        q = numpy.sqrt(self.qinp[0]**2 + self.qinp[1]**2 + self.qz**2)
        self.fSi = self.eSi.f(q, en) * self.Si._debyewallerfactor(temp, q)
        self.fGe = self.eGe.f(q, en) * self.Ge._debyewallerfactor(temp, q)
        self.fSi0 = self.eSi.f(0, en)
        self.fGe0 = self.eGe.f(0, en)

    def _calc_mono(self, pdict):
        """
        calculate the reflection and transmission coefficients of monolayer

        Parameters
        ----------
         pdict: property dictionary, contains the layer properties:
           x:   Ge-content of the layer (0: Si, 1: Ge)
           l:   index of the layer in the unit cell (0, 1, 2, 3). important for
                    asymmetric peaks only!

        Returns
        -------
         r, rbar, t: reflection, backside reflection, and tranmission
                     coefficients
        """
        ainp = pdict.get('ai')
        xGe = pdict.get('x')
        # pre-factor for reflection: contains footprint correction
        gamma = 4*numpy.pi*self.re*self.C/(self.qz*ainp**2)
        ltype = pdict.get('l', 0)
#        if ltype == 0: # for asymmetric peaks (not yet implemented)
#            p1, p2 = (0, 0), (0.5, 0.5)
#        elif ltype == 1:
#            p1, p2 = (0.25, 0.25), (0.75, 0.75)
#        elif ltype == 2:
#            p1, p2 = (0.5, 0.), (0., 0.5)
#        elif ltype == 3:
#            p1, p2 = (0.75, 0.25), (0.25, 0.75)

        r = -1j*gamma * (self.fSi+(self.fGe-self.fSi)*xGe) * 2
        # * (exp(1j*(h*p1[0]+k*p1[1])) + exp(1j*(h*p1[0]+k*p1[1])))
        t = 1 + 1j*gamma * (self.fSi0+(self.fGe0-self.fSi0)*xGe) * 2
        return r, numpy.copy(r), t

    def make_monolayers(self, s, asub=aSi):
        """
        create monolayer sequence from layer list

        Parameters
        ----------
         s:     layer model. list of layer dictionaries including possibility
                to form superlattices. As an example 5 repetitions of a
                Si(10nm)/Ge(15nm) superlattice on Si would like like:
                s = [(5, [{'t': 100, 'x': 0, 'r': 0},
                          {'t': 150, 'x': 1, 'r': 0}]),
                     {'t': 3500000, 'x': 0, 'r': 0}]
                the dictionaries can contain 't': thickness in A, 'x':
                Ge-content, 'r': relaxation or 'ai': inplane lattice parameter,
                'l': layer type (not yet implemented)
         asub:  inplane lattice parameter of the substrate

        Returns
        -------
         monolayer list in a format understood by the simulate and xGe_profile
         methods
        """
        ml = []
        ai = asub
        for subl in s:
            ml, ai = self._recur_makeml(subl, ml, ai)
        return ml

    def _recur_makeml(self, s, ml, apar):
        """
        recursive creation of a monolayer structure (internal)

        Parameters
        ----------
         s:     layer model. list of layer dictionaries
         ml:    list of layers below what should be added (needed for
                recursion)
         apar:  inplane lattice parameter of the current surface

        Returns
        -------
         monolayer list in a format understood by the simulate and xGe_profile
         methods
        """
        def rc12c11(xGe):
            """ ratio of elastic parameters of SiGe """
            return (63.9-15.6*xGe) / (165.8-37.3*xGe)  # IOFFE

        def get_aperp(x, r, apar):
            """
            determine out of plane mono-layer spacing from relaxation and
            Ge-content
            """
            abulk = self.aSi + (0.2 * x + 0.027 * x ** 2)
            aparl = apar + (abulk - apar) * r
            dperp = abulk*(1+2*rc12c11(x)*(1-aparl/abulk))/4.
            return dperp, aparl

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
            xGe = s['x']
            if callable(xGe):  # composition profile in layer
                t = 0
                while t < s['t']:
                    r = abs(derivative(xGe, t, dx=1.4, n=1))*s['r']
                    dperp, apar = get_aperp(xGe(t), r, apar)
                    t += dperp
                    ml.insert(0, (1, {'d': dperp, 'x': xGe(t), 'ai': apar}))
            else:  # constant composition layer
                dperp, apar = get_aperp(xGe, s['r'], apar)
                nmono = int(numpy.ceil(s['t']/dperp))
                ml.insert(0, (nmono, {'d': dperp, 'x': xGe, 'ai': apar}))
        else:
            raise Exception('wrong type (%s) of sublayer, must be tuple or'
                            ' dict' % (type(s)))
        return ml, apar

    def prop_profile(self, ml):
        """
        calculate the Ge profile and inplane lattice spacing from a monolayer
        list. One value for each monolayer in the sample is returned.

        Parameters
        ----------
         ml:    monolayer list created by make_monolayer()

        Returns
        -------
         zm, xGe, ainp:   z-position, Ge content, and inplane lattice spacing
                          for every monolayer. z=0 is the surface
        """

        def startinterval(start, inter, N):
            return numpy.arange(start, start+inter*(N+0.5), inter)

        def _recur_prop(nrep, ml, zp, xg, ai):
            if isinstance(ml, list):
                lzp, lxg, lai = ([], [], [])
                for nreps, subml in ml:
                    lzp, lxg, lai = _recur_prop(nreps, subml, lzp, lxg, lai)
                d = getfirst(ml, 'd')
            else:
                lzp = -ml['d']
                lxg = ml['x']
                lai = ml['ai']
                d = ml['d']

            Nmax = int(numpy.log(nrep) / numpy.log(2)) + 1
            for i in range(Nmax):
                if (nrep // (2**i)) % 2 == 1:
                    if len(zp) > 0:
                        curzp = zp[-1]
                    else:
                        curzp = 0.0
                    zp = numpy.append(zp, lzp+curzp)
                    xg = numpy.append(xg, lxg)
                    ai = numpy.append(ai, lai)
                try:
                    curlzp = lzp[-1]
                except:
                    curlzp = lzp
                lzp = numpy.append(lzp, lzp+curlzp)
                lxg = numpy.append(lxg, lxg)
                lai = numpy.append(lai, lai)
            return zp, xg, ai

        zm = []
        xGe = []
        ainp = []
        for nrep, subml in ml:
            zm, xGe, ainp = _recur_prop(nrep, subml, zm, xGe, ainp)
        return zm, xGe, ainp
