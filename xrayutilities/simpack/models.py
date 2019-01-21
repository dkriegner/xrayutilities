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
# Copyright (C) 2016-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

from __future__ import division

import abc
import math as pymath

import numpy
import scipy.constants as constants
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.special import erf, j0

from . import Layer, LayerStack
from .. import config, utilities
from ..exception import InputError
from ..experiment import Experiment
from ..math import NormGauss1d, NormLorentz1d, heaviside, solve_quartic


def startdelta(start, delta, num):
    end = start + delta * (num - 1)
    return numpy.linspace(start, end, int(num))


class Model(object):
    """
    generic model class from which further models can be derived from
    """

    def __init__(self, experiment, **kwargs):
        """
        constructor for a generical simulation model.
        currently only the experiment class describing the diffraction geometry
        is stored in the base class

        Parameters
        ----------
        experiment : Experiment
            class describing the diffraction geometry, energy and wavelength of
            the model
        resolution_width :  float, optional
            defines the width of the resolution
        I0 :                float, optional
            the primary beam flux/intensity
        background :        float, optional
            the background added to the simulation
        energy :            float or str
            the experimental energy in eV
        resolution_type :   {'Gauss', 'Lorentz'}, optional
            type of resolution function, default: Gauss
        """
        local_fit_params = ['resolution_width', 'I0', 'background', 'energy', ]
        if not hasattr(self, 'fit_paramnames'):
            self.fit_paramnames = []
        self.fit_paramnames += local_fit_params
        for kw in kwargs:
            if kw not in local_fit_params + ['resolution_type', ]:
                raise TypeError('%s is an invalid keyword argument' % kw)

        if experiment:
            self.exp = experiment
        else:
            self.exp = Experiment([1, 0, 0], [0, 0, 1])
        self.resolution_width = kwargs.get('resolution_width', 0)
        self.resolution_type = kwargs.get('resolution_type', 'Gauss')
        self.I0 = kwargs.get('I0', 1)
        self.background = kwargs.get('background', 0)
        if 'energy' in kwargs:
            self.energy = kwargs['energy']

    @property
    def energy(self):
        return self.exp.energy

    @energy.setter
    def energy(self, en):
        self.exp.energy = en

    def convolute_resolution(self, x, y):
        """
        convolve simulation result with a resolution function

        Parameters
        ----------
        x :     array-like
            x-values of the simulation, units of x also decide about the unit
            of the resolution_width parameter
        y :     array-like
            y-values of the simulation

        Returns
        -------
        array-like
            convoluted y-data with same shape as y
        """
        if self.resolution_width == 0:
            return y
        else:
            dx = numpy.mean(numpy.gradient(x))
            nres = int(20 * numpy.abs(self.resolution_width / dx))
            xres = startdelta(-10*self.resolution_width, dx, nres + 1)
            # the following works only exactly for equally spaced data points
            if self.resolution_type == 'Gauss':
                fres = NormGauss1d
            else:
                fres = NormLorentz1d
            resf = fres(xres, numpy.mean(xres), self.resolution_width)
            resf /= numpy.sum(resf)  # proper normalization for discrete conv.
            # pad y to avoid edge effects
            interp = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
            nextmin = numpy.ceil(nres/2.)
            nextpos = numpy.floor(nres/2.)
            xext = numpy.concatenate(
                (startdelta(x[0]-dx, -dx, nextmin)[-1::-1],
                 x,
                 startdelta(x[-1]+dx, dx, nextpos)))
            ypad = numpy.asarray(interp(xext))
            return numpy.convolve(ypad, resf, mode='valid')

    def scale_simulation(self, y):
        """
        scale simulation result with primary beam flux/intensity and add a
        background.

        Parameters
        ----------
        y :     array-like
            y-values of the simulation

        Returns
        -------
        array-like
            scaled y-values
        """
        return y * self.I0 + self.background


class LayerModel(Model, utilities.ABC):
    """
    generic model class from which further thin film models can be derived from
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a thin film model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        *args :         LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        **kwargs :      dict
            optional parameters for the simulation. ones not listed below are
            forwarded to the superclass.
        experiment :    Experiment, optional
            class containing geometry and energy of the experiment.
        surface_hkl :   list or tuple, optional
            Miller indices of the surface (default: (0, 0, 1))
        """
        exp = kwargs.pop('experiment', None)
        super(LayerModel, self).__init__(exp, **kwargs)
        self.lstack_params = []
        self.lstack_structural_params = False
        self.xlabelstr = 'x (1)'
        if len(args) == 1:
            if isinstance(args[0], Layer):
                self.lstack = LayerStack('Stack for %s'
                                         % self.__class__.__name__, *args)
            else:
                self.lstack = args[0]
        else:
            self.lstack = LayerStack('Stack for %s' % self.__class__.__name__,
                                     *args)

    @abc.abstractmethod
    def simulate(self):
        """
        abstract method that every implementation of a LayerModel has to
        override.
        """
        pass

    def _create_return(self, x, E, ai=None, af=None, Ir=None,
                       rettype='intensity'):
        """
        function to create the return value of a simulation. by default only
        the diffracted intensity is returned. However, optionally also the
        incidence and exit angle as well as the reflected intensity can be
        returned.

        Parameters
        ----------
        x :         array-like
            independent coordinate value for the convolution with the
            resolution function
        E :         array-like
            electric field amplitude (complex)
        ai, af :    array-like, optional
            incidence and exit angle of the XRD beam (in radians)
        Ir :        array-like, optional
            reflected intensity
        rettype :   {'intensity', 'field', 'all'}, optional
            type of the return value. 'intensity' (default): returns the
            diffracted beam flux convoluted with the resolution function;
            'field': returns the electric field (complex) without convolution
            with the resolution function, 'all': returns the electric field,
            ai, af (both in degree), and the reflected intensity.

        Returns
        -------
        return value depends on value of rettype.
        """
        if rettype == 'intensity':
            ret = self.scale_simulation(
                self.convolute_resolution(x, numpy.abs(E)**2))
        elif rettype == 'field':
            ret = E
        elif rettype == 'all':
            ret = (E, numpy.degrees(ai), numpy.degrees(af), Ir)
        return ret

    def get_polarizations(self):
        """
        return list of polarizations which should be calculated
        """
        if self.polarization == 'both':
            return ('S', 'P')
        else:
            return (self.polarization,)

    def join_polarizations(self, Is, Ip):
        """
        method to calculate the total diffracted intensity from the intensities
        of S and P-polarization.
        """
        if self.polarization == 'both':
            ret = (Is + self.Cmono * Ip) / (1 + self.Cmono)
        else:
            if self.polarization == 'S':
                ret = Is
            else:
                ret = Ip
        return ret


class KinematicalModel(LayerModel):
    """
    Kinematical diffraction model for specular and off-specular qz-scans. The
    model calculates the kinematical contribution of one (hkl) Bragg peak,
    however considers the variation of the structure factor for different 'q'.
    The surface geometry is specified using the Experiment-object given to the
    constructor.
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a kinematic thin film model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified in
        the keyword arguments.

        Parameters
        ----------
        *args :     LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        **kwargs :  dict
            optional parameters; also see LayerModel/Model.
        experiment : Experiment
            Experiment class containing geometry and energy of the experiment.
        """
        super(KinematicalModel, self).__init__(*args, **kwargs)
        self.lstack_params += ['thickness', ]
        self.lstack_structural_params = True
        self.xlabelstr = r'momentum transfer $Q_z$ ($\AA^{-1}$)'
        # precalc optical properties
        self._init_en = 0
        self.init_chi0()

    def init_chi0(self):
        """
        calculates the needed optical parameters for the simulation. If any of
        the materials/layers is changing its properties this function needs to
        be called again before another correct simulation is made. (Changes of
        thickness does NOT require this!)
        """
        if self._init_en != self.energy:  # recalc properties if energy changed
            self.chi0 = numpy.asarray([l.material.chi0(en=self.energy)
                                       for l in self.lstack])
            self._init_en = self.energy

    def _prepare_kincalculation(self, qz, hkl):
        """
        prepare kinematic calculation by calculating some helper values
        """
        rel = constants.physical_constants['classical electron radius'][0]
        rel *= 1e10
        k = self.exp.k0

        # determine q-inplane
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*hkl))
        qinp = numpy.sqrt(ql0[0]**2 + ql0[1]**2)

        # calculate needed angles
        qv = numpy.asarray([t.inverse((ql0[0], ql0[1], q)) for q in qz])
        Q = numpy.linalg.norm(qv, axis=1)
        theta = numpy.arcsin(Q / (2 * k))
        domega = numpy.arctan2(qinp, qz)
        alphai, alphaf = (theta + domega, theta - domega)
        # calculate structure factors
        f = numpy.empty((len(self.lstack), len(qz)), dtype=numpy.complex)
        fhkl = numpy.empty(len(self.lstack), dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            m = l.material
            fhkl[i] = m.StructureFactor(m.Q(*hkl), en=self.energy) /\
                m.lattice.UnitCellVolume()
            f[i, :] = m.StructureFactorForQ(qv, en0=self.energy) /\
                m.lattice.UnitCellVolume()

        E = numpy.zeros(len(qz), dtype=numpy.complex)
        return rel, alphai, alphaf, f, fhkl, E, t

    def _get_qz(self, qz, alphai, alphaf, chi0, absorption, refraction):
        k = self.exp.k0
        q = qz.astype(numpy.complex)
        if absorption and not refraction:
            q += 1j * k * numpy.imag(chi0) / \
                numpy.sin((alphai + alphaf) / 2)
        if refraction:
            q = k * (numpy.sqrt(numpy.sin(alphai)**2 + chi0) +
                     numpy.sqrt(numpy.sin(alphaf)**2 + chi0))
        return q

    def simulate(self, qz, hkl, absorption=False, refraction=False,
                 rettype='intensity'):
        """
        performs the actual kinematical diffraction calculation on the Qz
        positions specified considering the contribution from a single Bragg
        peak.

        Parameters
        ----------
        qz :        array-like
            simulation positions along qz
        hkl :       list or tuple
            Miller indices of the Bragg peak whos truncation rod should be
            calculated
        absorption : bool, optional
            flag to tell if absorption correction should be used
        refraction : bool, optional
            flag to tell if basic refraction correction should be performed. If
            refraction is True absorption correction is also included
            independent of the absorption flag.
        rettype :   {'intensity', 'field', 'all'}
            type of the return value. 'intensity' (default): returns the
            diffracted beam flux convoluted with the resolution function;
            'field': returns the electric field (complex) without convolution
            with the resolution function, 'all': returns the electric field,
            ai, af (both in degree), and the reflected intensity.

        Returns
        -------
        array-like
            return value depends on the setting of `rettype`, by default only
            the calculate intensity is returned
        """
        self.init_chi0()
        rel, ai, af, f, fhkl, E, t = self._prepare_kincalculation(qz, hkl)
        # calculate interface positions
        z = numpy.zeros(len(self.lstack))
        for i, l in enumerate(self.lstack[-1:0:-1]):
            z[-i-2] = z[-i-1] - l.thickness

        # perform kinematical calculation
        for i, l in enumerate(self.lstack):
            q = self._get_qz(qz, ai, af, self.chi0[i], absorption, refraction)
            q -= t(l.material.Q(*hkl))[-1]

            if l.thickness == numpy.inf:
                E += fhkl[i] * numpy.exp(-1j * z[i] * q) / (1j * q)
            else:
                E += - fhkl[i] * numpy.exp(-1j * q * z[i]) * \
                    (1 - numpy.exp(1j * q * l.thickness)) / (1j * q)

        wf = numpy.sqrt(heaviside(ai) * heaviside(af) * rel**2 /
                        (numpy.sin(ai) * numpy.sin(af))) * E
        return self._create_return(qz, wf, ai, af, rettype=rettype)


class KinematicalMultiBeamModel(KinematicalModel):
    """
    Kinematical diffraction model for specular and off-specular qz-scans. The
    model calculates the kinematical contribution of several Bragg peaks on
    the truncation rod and considers the variation of the structure factor.
    In order to use a analytical description for the kinematic diffraction
    signal all layer thicknesses are changed to a multiple of the respective
    lattice parameter along qz. Therefore this description only works for (001)
    surfaces.
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a kinematic thin film model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified in
        the keyword arguments.

        Parameters
        ----------
        *args :     LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        **kwargs :  dict
            optional parameters. see also LayerModel/Model.
        experiment : Experiment
            Experiment class containing geometry and energy of the experiment.
        surface_hkl : list or tuple
            Miller indices of the surface (default: (0, 0, 1))
        """
        self.surface_hkl = kwargs.pop('surface_hkl', (0, 0, 1))
        super(KinematicalMultiBeamModel, self).__init__(*args, **kwargs)

    def simulate(self, qz, hkl, absorption=False, refraction=True,
                 rettype='intensity'):
        """
        performs the actual kinematical diffraction calculation on the Qz
        positions specified considering the contribution from a full
        truncation rod

        Parameters
        ----------
        qz :            array-like
            simulation positions along qz
        hkl :           list or tuple
            Miller indices of the Bragg peak whos truncation rod should be
            calculated
        absorption :    bool, optional
            flag to tell if absorption correction should be used
        refraction :    bool, optional,
            flag to tell if basic refraction correction should be performed. If
            refraction is True absorption correction is also included
            independent of the absorption flag.
        rettype :       {'intensity', 'field', 'all'}
            type of the return value. 'intensity' (default): returns the
            diffracted beam flux convoluted with the resolution function;
            'field': returns the electric field (complex) without convolution
            with the resolution function, 'all': returns the electric field,
            ai, af (both in degree), and the reflected intensity.

        Returns
        -------
        array-like
            return value depends on the setting of `rettype`, by default only
            the calculate intensity is returned
        """
        self.init_chi0()
        rel, ai, af, f, fhkl, E, t = self._prepare_kincalculation(qz, hkl)

        # calculate interface positions for integer unit-cell thickness
        z = numpy.zeros(len(self.lstack))
        for i, l in enumerate(self.lstack[-1:0:-1]):
            lat = l.material.lattice
            a3 = t(lat.GetPoint(*self.surface_hkl))[-1]
            n3 = l.thickness // a3
            z[-i-2] = z[-i-1] - a3 * n3
            if config.VERBOSITY >= config.INFO_LOW and \
                    numpy.abs(l.thickness/a3 - n3) > 0.01:
                print('XU.KinematicMultiBeamModel: %s thickness changed from'
                      ' %.2fA to %.2fA (%d UCs)' % (l.name, l.thickness,
                                                    a3 * n3, n3))

        # perform kinematical calculation
        for i, l in enumerate(self.lstack):
            q = self._get_qz(qz, ai, af, self.chi0[i], absorption, refraction)
            lat = l.material.lattice
            a3 = t(lat.GetPoint(*self.surface_hkl))[-1]

            if l.thickness == numpy.inf:
                E += f[i, :] * a3 * numpy.exp(-1j * z[i] * q) /\
                    (1 - numpy.exp(1j * q * a3))
            else:
                n3 = l.thickness // a3
                E += f[i, :] * a3 * numpy.exp(-1j * z[i] * q) * \
                    (1 - numpy.exp(1j * q * a3 * n3)) /\
                    (1 - numpy.exp(1j * q * a3))

        wf = numpy.sqrt(heaviside(ai) * heaviside(af) * rel**2 /
                        (numpy.sin(ai) * numpy.sin(af))) * E
        return self._create_return(qz, wf, ai, af, rettype=rettype)


class SimpleDynamicalCoplanarModel(KinematicalModel):
    """
    Dynamical diffraction model for specular and off-specular qz-scans.
    Calculation of the flux of reflected and diffracted waves for general
    asymmetric coplanar diffraction from an arbitrary pseudomorphic multilayer
    is performed by a simplified 2-beam theory (2 tiepoints, S and P
    polarizations)

    No restrictions are made for the surface orientation.

    The first layer in the model is always assumed to be the semiinfinite
    substrate indepentent of its given thickness

    Note:
        This model should not be used in real life scenarios since the made
        approximations severely fail for distances far from the reference
        position.
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a diffraction model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        *args :     LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        **kwargs:   dict
            optional parameters for the simulation
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        resolution_width : float, optional
            the width of the resolution (deg)
        polarization: {'S', 'P', 'both'}
            polarization of the x-ray beam. If set to 'both' also Cmono, the
            polarization factor of the monochromator should be set
        Cmono :     float, optional
            polarization factor of the monochromator
        energy :    float or str
            the experimental energy in eV
        experiment : Experiment
            Experiment class containing geometry of the sample; surface
            orientation!
        """
        if not hasattr(self, 'fit_paramnames'):
            self.fit_paramnames = []
        self.fit_paramnames += ['Cmono', ]
        self.polarization = kwargs.pop('polarization', 'S')
        self.Cmono = kwargs.pop('Cmono', 1)
        super(SimpleDynamicalCoplanarModel, self).__init__(*args, **kwargs)
        self.xlabelstr = 'incidence angle (deg)'
        self.hkl = None
        self.chih = None
        self.chimh = None

    def set_hkl(self, *hkl):
        """
        To speed up future calculations of the same Bragg peak optical
        parameters can be pre-calculated using this function.

        Parameters
        ----------
        hkl :       list or tuple
            Miller indices of the Bragg peak for the calculation
        """
        if hkl is not None:
            if len(hkl) < 3:
                hkl = hkl[0]
                if len(hkl) < 3:
                    raise InputError("need 3 Miller indices")
            newhkl = numpy.asarray(hkl)
        else:
            newhkl = self.hkl

        if self.energy != self._init_en or numpy.any(newhkl != self.hkl):
            self.hkl = newhkl
            self._init_en = self.energy

            # calculate chih
            self.chih = {'S': [], 'P': []}
            self.chimh = {'S': [], 'P': []}
            for l in self.lstack:
                q = l.material.Q(self.hkl)
                thetaB = numpy.arcsin(numpy.linalg.norm(q) / 2 / self.exp.k0)
                ch = l.material.chih(q, en=self.energy, polarization='S')
                self.chih['S'].append(-ch[0] + 1j*ch[1])
                self.chih['P'].append((-ch[0] + 1j*ch[1]) *
                                      numpy.abs(numpy.cos(2*thetaB)))
                if not getattr(l, 'inversion_sym', False):
                    ch = l.material.chih(-q, en=self.energy, polarization='S')
                self.chimh['S'].append(-ch[0] + 1j*ch[1])
                self.chimh['P'].append((-ch[0] + 1j*ch[1]) *
                                       numpy.abs(numpy.cos(2*thetaB)))

            for pol in ('S', 'P'):
                self.chih[pol] = numpy.asarray(self.chih[pol])
                self.chimh[pol] = numpy.asarray(self.chimh[pol])

    def _prepare_dyncalculation(self, geometry):
        """
        prepare dynamical calculation by calculating some helper values
        """
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*self.hkl))
        hx = numpy.sqrt(ql0[0]**2 + ql0[1]**2)
        if geometry == 'lo_hi':
            hx = -hx

        # calculate vertical diffraction vector components and strain
        hz = numpy.zeros(len(self.lstack))
        for i, l in enumerate(self.lstack):
            hz[i] = t(l.material.Q(*self.hkl))[2]
        return t, hx, hz

    def simulate(self, alphai, hkl=None, geometry='hi_lo', idxref=1):
        """
        performs the actual diffraction calculation for the specified
        incidence angles.

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles (deg)
        hkl :       list or tuple, optional
            Miller indices of the diffraction vector (preferable use set_hkl
            method to speed up repeated calculations of the same peak!)
        geometry :  {'hi_lo', 'lo_hi'}, optional
            'hi_lo' for grazing exit (default) and 'lo_hi' for grazing
            incidence
        idxref :    int, optional
            index of the reference layer. In order to get accurate peak
            position of the film peak you want this to be the index of the film
            peak (default: 1). For the substrate use 0.

        Returns
        -------
        array-like
            vector of intensities of the diffracted signal
        """
        self.set_hkl(hkl)

        # return values
        Ih = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}

        t, hx, hz = self._prepare_dyncalculation(geometry)
        epsilon = (hz[idxref] - hz) / hz

        k = self.exp.k0
        thetaB = numpy.arcsin(numpy.sqrt(hx**2 + hz[idxref]**2) / 2 / k)
        # asymmetry angle
        asym = numpy.arctan2(hx, hz[idxref])
        gamma0 = numpy.sin(asym + thetaB)
        gammah = numpy.sin(asym - thetaB)

        # deviation of the incident beam from the kinematical maximum
        eta = numpy.radians(alphai) - thetaB - asym

        for pol in self.get_polarizations():
            x = numpy.zeros(len(alphai), dtype=numpy.complex)
            for i, l in enumerate(self.lstack):
                beta = (2 * eta * numpy.sin(2 * thetaB) +
                        self.chi0[i] * (1 - gammah / gamma0) -
                        2 * gammah * (gamma0 - gammah) * epsilon[i])
                y = beta / 2 / numpy.sqrt(self.chih[pol][i] *
                                          self.chimh[pol][i]) /\
                    numpy.sqrt(numpy.abs(gammah) / gamma0)
                c1 = -numpy.sqrt(self.chih[pol][i] / self.chih[pol][i] *
                                 gamma0 / numpy.abs(gammah)) *\
                    (y + numpy.sqrt(y**2 - 1))
                c2 = -numpy.sqrt(self.chih[pol][i] / self.chimh[pol][i] *
                                 gamma0 / numpy.abs(gammah)) *\
                    (y - numpy.sqrt(y**2 - 1))
                kz2mkz1 = k * numpy.sqrt(self.chih[pol][i] *
                                         self.chimh[pol][i] / gamma0 /
                                         numpy.abs(gammah)) *\
                    numpy.sqrt(y**2 - 1)
                if i == 0:  # substrate
                    pp = numpy.abs(gammah) / gamma0 * numpy.abs(c1)**2
                    m = pp < 1
                    x[m] = c1[m]
                    m = pp >= 1
                    x[m] = c2[m]
                else:  # layers
                    cphi = numpy.exp(1j * kz2mkz1 * l.thickness)
                    x = (c1 * c2 * (cphi - 1) + xs * (c1 - cphi * c2)) /\
                        (cphi * c1 - c2 + xs * (1 - cphi))
                xs = x
            Ih[pol] = numpy.abs(x)**2 * numpy.abs(gammah) / gamma0

        ret = self.join_polarizations(Ih['S'], Ih['P'])
        return self.scale_simulation(self.convolute_resolution(alphai, ret))


class DynamicalModel(SimpleDynamicalCoplanarModel):
    """
    Dynamical diffraction model for specular and off-specular qz-scans.
    Calculation of the flux of reflected and diffracted waves for general
    asymmetric coplanar diffraction from an arbitrary pseudomorphic multilayer
    is performed by a generalized 2-beam theory (4 tiepoints, S and P
    polarizations)

    The first layer in the model is always assumed to be the semiinfinite
    substrate indepentent of its given thickness
    """

    def simulate(self, alphai, hkl=None, geometry='hi_lo',
                 rettype='intensity'):
        """
        performs the actual diffraction calculation for the specified
        incidence angles and uses an analytic solution for the quartic
        dispersion equation

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles (deg)
        hkl :       list or tuple, optional
            Miller indices of the diffraction vector (preferable use set_hkl
            method to speed up repeated calculations of the same peak!)
        geometry :  {'hi_lo', 'lo_hi'}, optional
            'hi_lo' for grazing exit (default) and 'lo_hi' for grazing
            incidence
        rettype :   {'intensity', 'field', 'all'}, optional
            type of the return value. 'intensity' (default): returns the
            diffracted beam flux convoluted with the resolution function;
            'field': returns the electric field (complex) without convolution
            with the resolution function, 'all': returns the electric field,
            ai, af (both in degree), and the reflected intensity.

        Returns
        -------
        array-like
            vector of intensities of the diffracted signal, possibly changed
            return value due the rettype setting!
        """
        if len(self.get_polarizations()) > 1 and rettype != "intensity":
            raise ValueError('XU:DynamicalModel: return type (%s) not '
                             'supported with multiple polarizations!')
            rettype = 'intensity'
        self.set_hkl(hkl)

        # return values
        Ih = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}
        Ir = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}

        t, hx, hz = self._prepare_dyncalculation(geometry)

        k = self.exp.k0
        kc = k * numpy.sqrt(1 + self.chi0)
        ai = numpy.radians(alphai)
        Kix = k * numpy.cos(ai)
        Kiz = -k * numpy.sin(ai)
        Khz = numpy.sqrt(k**2 - (Kix + hx)**2)
        pp = Khz / k
        mask = numpy.logical_and(pp > 0, pp < 1)
        ah = numpy.zeros(len(ai))  # exit angles
        ah[mask] = numpy.arcsin(pp[mask])

        nal = len(ai)
        for pol in self.get_polarizations():
            if pol == 'S':
                CC = numpy.ones(nal)
            else:
                CC = abs(numpy.cos(ai+ah))
            pom = k**4 * self.chih['S'] * self.chimh['S']
            if config.VERBOSITY >= config.INFO_ALL:
                print('XU.DynamicalModel: calc. %s-polarization...' % (pol))

            M = numpy.zeros((nal, 4, 4), dtype=numpy.complex)
            for j in range(4):
                M[:, j, j] = numpy.ones(nal)

            for i, l in enumerate(self.lstack[-1::-1]):
                jL = len(self.lstack) - 1 - i
                A4 = numpy.ones(nal)
                A3 = 2 * hz[jL] * numpy.ones(nal)
                A2 = (Kix + hx)**2 + hz[jL]**2 + Kix**2 - 2 * kc[jL]**2
                A1 = 2 * hz[jL] * (Kix**2 - kc[jL]**2)
                A0 = (Kix**2 - kc[jL]**2) *\
                     ((Kix + hx)**2 + hz[jL]**2 - kc[jL]**2) - pom[jL] * CC**2
                X = solve_quartic(A4, A3, A2, A1, A0)
                X = numpy.asarray(X).T

                kz = numpy.zeros((nal, 4), dtype=numpy.complex)
                kz[:, :2] = X[numpy.imag(X) <= 0].reshape(nal, 2)
                kz[:, 2:] = X[numpy.imag(X) > 0].reshape(nal, 2)

                P = numpy.zeros((nal, 4, 4), dtype=numpy.complex)
                phi = numpy.zeros((nal, 4, 4), dtype=numpy.complex)
                c = ((Kix**2)[:, numpy.newaxis] + kz**2 - kc[jL]**2) / k**2 /\
                    self.chimh['S'][jL] / CC[:, numpy.newaxis]
                if jL > 0:
                    for j in range(4):
                        phi[:, j, j] = numpy.exp(1j * kz[:, j] * l.thickness)
                else:
                    phi = numpy.tile(numpy.identity(4), (nal, 1, 1))
                P[:, 0, :] = numpy.ones((nal, 4))
                P[:, 1, :] = c
                P[:, 2, :] = kz
                P[:, 3, :] = c * (kz + hz[jL])

                if i == 0:
                    R = numpy.copy(P)
                else:
                    temp = numpy.linalg.inv(Ps)
                    try:
                        R = numpy.matmul(temp, P)
                    except AttributeError:
                        R = numpy.einsum('...ij,...jk', temp, P)
                try:
                    M = numpy.matmul(numpy.matmul(M, R), phi)
                except AttributeError:
                    M = numpy.einsum('...ij,...jk',
                                     numpy.einsum('...ij,...jk', M, R), phi)
                Ps = numpy.copy(P)

            B = numpy.zeros((nal, 4, 4), dtype=numpy.complex)
            B[..., :2] = M[..., :2]
            B[:, 0, 2] = -numpy.ones(nal)
            B[:, 1, 3] = -numpy.ones(nal)
            B[:, 2, 2] = Kiz
            B[:, 3, 3] = -Khz
            C = numpy.zeros((nal, 4))
            C[:, 0] = numpy.ones(nal)
            C[:, 2] = Kiz

            E = numpy.einsum('...ij,...j', numpy.linalg.inv(B), C)
            Ir[pol] = numpy.abs(E[:, 2])**2  # reflected intensity
            Ih[pol] = numpy.abs(E[:, 3])**2 * numpy.abs(Khz / Kiz) * mask

        if len(self.get_polarizations()) > 1 and rettype == "intensity":
            ret = numpy.sqrt(self.join_polarizations(Ih['S'], Ih['P']))
        else:
            ret = E[:, 3] * numpy.sqrt(numpy.abs(Khz / Kiz) * mask)

        return self._create_return(alphai, ret, ai, ah, Ir, rettype=rettype)


class SpecularReflectivityModel(LayerModel):
    """
    model for specular reflectivity calculations
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a reflectivity model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        args :      LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        kwargs:     dict
            optional parameters for the simulation; supported are:
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        sample_width : float, optional
            width of the sample along the beam
        beam_width : float, optional
            beam width in the same units as the sample width
        offset :    float, optional
            angular offset of the incidence angle (deg)
        resolution_width : float, optional
            width of the resolution (deg)
        energy :    float or str
            x-ray energy  in eV
        """
        if not hasattr(self, 'fit_paramnames'):
            self.fit_paramnames = []
        self.fit_paramnames += ['sample_width', 'beam_width', 'offset']
        self.sample_width = kwargs.pop('sample_width', numpy.inf)
        self.beam_width = kwargs.pop('beam_width', 0)
        self.offset = kwargs.pop('offset', 0)
        super(SpecularReflectivityModel, self).__init__(*args, **kwargs)
        self.lstack_params += ['thickness', 'roughness', 'density']
        self.xlabelstr = 'incidence angle (deg)'
        # precalc optical properties
        self._init_en = 0
        self.init_cd()

    def init_cd(self):
        """
        calculates the needed optical parameters for the simulation. If any of
        the materials/layers is changing its properties this function needs to
        be called again before another correct simulation is made. (Changes of
        thickness and roughness do NOT require this!)
        """
        if self._init_en != self.energy:
            self.cd = numpy.asarray([-l.material.chi0(en=self.energy)/2
                                     for l in self.lstack])
            self._init_en = self.energy

    def simulate(self, alphai):
        """
        performs the actual reflectivity calculation for the specified
        incidence angles

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles

        Returns
        -------
        array-like
            vector of intensities of the reflectivity signal
        """
        self.init_cd()
        ns, np = (len(self.lstack), len(alphai))
        lai = alphai - self.offset
        # get layer properties
        t = numpy.asarray([l.thickness for l in self.lstack])
        sig = numpy.asarray([getattr(l, 'roughness', 0) for l in self.lstack])
        rho = numpy.asarray([getattr(l, 'density', 1) for l in self.lstack])
        cd = self.cd

        sai = numpy.sin(numpy.radians(lai))

        if self.beam_width > 0:
            shape = self.sample_width * sai / self.beam_width
            shape[shape > 1] = 1
        else:
            shape = numpy.ones(np)

        ETs = numpy.ones(np, dtype=numpy.complex)
        ERs = numpy.zeros(np, dtype=numpy.complex)
        ks = -self.exp.k0 * numpy.sqrt(sai**2 - 2 * cd[0] * rho[0])

        for i in range(ns):
            if i < ns-1:
                k = -self.exp.k0 * numpy.sqrt(sai**2 - 2 * cd[i+1] * rho[i+1])
                phi = numpy.exp(1j * k * t[i+1])
            else:
                k = -self.exp.k0 * sai
                phi = numpy.ones(np)
            r = (k - ks) / (k + ks) * numpy.exp(-2 * sig[i]**2 * k * ks)
            ET = phi * (ETs + r * ERs)
            ER = (r * ETs + ERs) / phi
            ETs = ET
            ERs = ER
            ks = k

        R = shape * abs(ER / ET)**2
        return self.scale_simulation(self.convolute_resolution(lai, R))

    def densityprofile(self, nz, plot=False):
        """
        calculates the electron density of the layerstack from the thickness
        and roughness of the individual layers

        Parameters
        ----------
        nz :    int
            number of values on which the profile should be calculated
        plot :  bool, optional
            flag to tell if a plot of the profile should be created

        Returns
        -------
        z :     array-like
            z-coordinates, z = 0 corresponds to the surface
        eprof : array-like
            electron profile
        """
        if plot:
            try:
                from matplotlib import pyplot as plt
            except ImportError:
                plot = False
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.simpack: Warning: plot "
                          "functionality not available")

        rel = constants.physical_constants['classical electron radius'][0]
        rel *= 1e10
        nl = len(self.lstack)

        # get layer properties
        t = numpy.asarray([l.thickness for l in self.lstack])
        sig = numpy.asarray([getattr(l, 'roughness', 0) for l in self.lstack])
        rho = numpy.asarray([getattr(l, 'density', 1) for l in self.lstack])
        delta = numpy.real(self.cd)

        totT = numpy.sum(t[1:])
        zmin = -totT - 10 * sig[0]
        zmax = 5 * sig[-1]

        z = numpy.linspace(zmin, zmax, nz)
        pre_factor = 2 * numpy.pi / self.exp.wavelength**2 / rel * 1e24

        # generate delta-rho values and interface positions
        zz = numpy.zeros(nl)
        dr = numpy.zeros(nl)
        dr[-1] = delta[-1] * rho[-1] * pre_factor
        for i in range(nl-1, 0, -1):
            zz[i-1] = zz[i] - t[i]
            dr[i-1] = delta[i-1] * rho[i-1] * pre_factor

        # calculate profile from contribution of all interfaces
        prof = numpy.zeros(nz)
        w = numpy.zeros((nl, nz))
        for i in range(nl):
            s = (1 + erf((z - zz[i]) / sig[i] / numpy.sqrt(2))) / 2
            mask = s < (1 - 1e-10)
            w[i, mask] = s[mask] / (1 - s[mask])
            mask = numpy.logical_not(mask)
            w[i, mask] = 1e10

        c = numpy.ones((nl, nz))
        for i in range(1, nl):
            c[i, :] = w[i-1, :] * c[i-1, :]
        c0 = w[-1, :] * c[-1, :]
        norm = numpy.sum(c, axis=0) + c0

        for i in range(nl):
            c[i] /= norm

        for j in range(nz):
            prof[j] = numpy.sum(dr * c[:, j])

        if plot:
            plt.figure('XU:density_profile', figsize=(5, 3))
            plt.plot(z, prof, 'k-', lw=2, label='electron density')
            plt.xlabel(r'z ($\AA$)')
            plt.ylabel(r'electron density (e$^-$ cm$^{-3}$)')
            plt.tight_layout()

        return z, prof


class DynamicalReflectivityModel(SpecularReflectivityModel):
    """
    model for Dynamical Specular Reflectivity Simulations.
    It uses the transfer Matrix methods as given in chapter 3
    "Daillant, J., & Gibaud, A. (2008). X-ray and Neutron Reflectivity"
    """
    def __init__(self, *args, **kwargs):
        """
        constructor for a reflectivity model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        args :      LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        kwargs:     dict
            optional parameters for the simulation; supported are:
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        sample_width : float, optional
            width of the sample along the beam
        beam_width : float, optional
            beam width in the same units as the sample width
        resolution_width : float, optional
            width of the resolution (deg)
        energy :    float or str
            x-ray energy  in eV
        polarization:   ['P', 'S']
            x-ray polarization
        """
        self.polarization = kwargs.pop('polarization', 'P')
        super(DynamicalReflectivityModel, self).__init__(*args, **kwargs)
        self._init_en_opt
        self._setOpticalConstants()

    def _setOpticalConstants(self):
        if self._init_en_opt != self.energy:
            self.n_indices = numpy.asarray(
                [l.material.idx_refraction(en=self.energy)
                 for l in self.lstack])
            # append n = 1 for vacuum
            self.n_indices = numpy.append(self.n_indices, 1)[::-1]
            self._init_en_opt = self.energy

    def _getTransferMatrices(self, alphai):
        """
        Calculation of Refraction and Translation Matrices per angle per layer.
        """
        # Set heights for each layer
        heights = numpy.asarray([l.thickness for l in self.lstack[1:]])
        heights = numpy.cumsum(heights)[::-1]
        heights = numpy.insert(heights, 0, 0.)  # first interface is at z=0

        # set K-vector in each layer
        kz_angles = -self.exp.k0 * numpy.sqrt(numpy.asarray(
            [n**2 - numpy.cos(numpy.radians(alphai))**2
             for n in self.n_indices]).T)

        # set Roughness for each layer
        roughness = numpy.asarray([l.roughness for l in self.lstack[1:]])[::-1]
        roughness = numpy.insert(roughness, 0, 0.)  # first interface is at z=0

        # Roughness is approximated by a Gaussian Statistics model modification
        # of the transfer matrix elements using Groce-Nevot factors (GNF).

        GNF_factor_P = numpy.asarray(
            [[numpy.exp(-(kz_next - kz)**2 * (rough**2) / 2)
              for (kz, kz_next, rough) in zip(kz_1angle, kz_1angle[1:],
                                              roughness[1:])]
             for kz_1angle in kz_angles])

        GNF_factor_M = numpy.asarray(
            [[numpy.exp(-(kz_next + kz) ** 2 * (rough ** 2) / 2)
              for (kz, kz_next, rough) in zip(kz_1angle, kz_1angle[1:],
                                              roughness[1:])]
             for kz_1angle in kz_angles])

        if self.polarization is 'S':
            p_factor_angles = numpy.asarray(
                [[(kz + kz_next) / (2 * kz)
                  for kz, kz_next in zip(kz_1angle, kz_1angle[1:])]
                 for kz_1angle in kz_angles])

            m_factor_angles = numpy.asarray(
                [[(kz - kz_next) / (2 * kz)
                  for kz, kz_next in zip(kz_1angle, kz_1angle[1:])]
                 for kz_1angle in kz_angles])
        else:
            p_factor_angles = numpy.asarray(
                [[(n_next**2*kz + n**2*kz_next) / (2*n_next**2*kz)
                  for (kz, kz_next, n, n_next) in zip(kz_1angle, kz_1angle[1:],
                                                      self.n_indices,
                                                      self.n_indices[1:])]
                 for kz_1angle in kz_angles])
            m_factor_angles = numpy.asarray(
                [[(n_next**2*kz - n**2*kz_next) / (2*n_next**2*kz)
                  for (kz, kz_next, n, n_next) in zip(kz_1angle, kz_1angle[1:],
                                                      self.n_indices,
                                                      self.n_indices[1:])]
                 for kz_1angle in kz_angles])

        # Translation Matrices dim = (angle, layer, 2, 2)
        T_matrices = numpy.asarray(
            [[([numpy.exp(-1.j*kz*height), 0], [0, numpy.exp(1.j*kz*height)])
              for kz, height in zip(kz_1angle, heights)]
             for kz_1angle in kz_angles])

        R_matrices = numpy.asarray(
            [[([p, m], [m, p]) for p, m in zip(P_fact, M_fact)]
             for (P_fact, M_fact) in zip(p_factor_angles, m_factor_angles)])

        for R_mat, GNF_P, GNF_M in zip(R_matrices, GNF_factor_P, GNF_factor_M):
            R_mat[0, 0] = R_mat[0, 0] * GNF_P
            R_mat[0, 1] = R_mat[0, 1] * GNF_M
            R_mat[1, 0] = R_mat[1, 0] * GNF_M
            R_mat[1, 1] = R_mat[1, 1] * GNF_P

        return T_matrices, R_matrices

    def simulate(self, alphai):
        """
        Simulates the Dynamical Reflectivity as a function of angle of
        incidence

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles

        Returns
        -------
        reflectivity:   array-like
            vector of intensities of the reflectivity signal
        transmitivity: array-like
            vector of intensities of the transmitted signal
        """
        self._setOpticalConstants()
        lai = alphai - self.offset
        # Get Refraction and Translation Matrices for each angle of incidence
        if lai[0] < 1.e-5:
            lai[0] = 1.e-5  # cutoff

        T_matrices, R_matrices = self._getTransferMatrices(lai)

        # Calculate the Transfer Matrix
        M_angles = numpy.zeros((lai.size, 2, 2), dtype=numpy.complex128)
        for (angle, R), T in zip(enumerate(R_matrices), T_matrices):
            pairwiseRT = [numpy.dot(t, r) for r, t in zip(R, T)]
            M = numpy.identity(2, dtype=numpy.complex128)
            for pair in pairwiseRT:
                M = numpy.dot(M, pair)
            M_angles[angle] = M

        # Reflectance and Transmittance
        R = numpy.array([numpy.abs((M[0, 1] / M[1, 1]))**2 for M in M_angles])
        T = numpy.array([numpy.abs((1. / M[1, 1]))**2 for M in M_angles])

        return R, T

    def scanEnergy(self, energies, angle):
        # TODO: this is quite inefficient, too many calls to internal functions
        # TODO: DO not return normalized refelctivity
        """
        Simulates the Dynamical Reflectivity as a function of photon energy at
        fixed angle.

        Parameters
        ----------
        energies: np.ndarray or list
            photon energies (in eV).
        angle : float
            fixed incidence angle

        Returns
        -------
        reflectivity:   array-like
            vector of intensities of the reflectivity signal
        transmitivity: array-like
            vector of intensities of the transmitted signal
        """
        R_energies, T_energies = numpy.array([]), numpy.array([])
        for energy in energies:
            self.energy = energy
            self._setOpticalConstants()
            T_matrices, R_matrices = self._getTransferMatrices([angle, 0])
            T_matrix = T_matrices[0]
            R_matrix = R_matrices[0]
            pairwiseRT = [numpy.dot(t, r) for r, t in zip(R_matrix, T_matrix)]
            M = numpy.identity(2, dtype=numpy.complex128)
            for pair in pairwiseRT:
                M = numpy.dot(M, pair)
            R = numpy.abs(M[0, 1] / M[1, 1]) ** 2
            T = numpy.abs(1. / M[1, 1]) ** 2
            R_energies = numpy.append(R_energies, R)
            T_energies = numpy.append(T_energies, T)
        return R_energies, T_energies


class ResonantReflectivityModel(SpecularReflectivityModel):
    """
    model for specular reflectivity calculations
    CURRENTLY UNDER DEVELOPEMENT! DO NOT USE!
    """
    def __init__(self, *args, **kwargs):
        """
        constructor for a reflectivity model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        args :      LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        kwargs:     dict
            optional parameters for the simulation; supported are:
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        sample_width : float, optional
            width of the sample along the beam
        beam_width : float, optional
            beam width in the same units as the sample width
        resolution_width : float, optional
            width of the resolution (deg)
        energy :    float or str
            x-ray energy  in eV
        polarization:   ['P', 'S']
            x-ray polarization
        """
        self.polarization = kwargs.pop('polarization', 'S')
        super(ResonantReflectivityModel, self).__init__(*args, **kwargs)

    def simulate(self, alphai):
        """
        performs the actual reflectivity calculation for the specified
        incidence angles

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles

        Returns
        -------
        array-like
            vector of intensities of the reflectivity signal
        """
        self.init_cd()
        ns, np = (len(self.lstack), len(alphai))
        lai = alphai - self.offset

        # get layer properties
        t = numpy.asarray([l.thickness for l in self.lstack])
        sig = numpy.asarray([getattr(l, 'roughness', 0) for l in self.lstack])
        rho = numpy.asarray([getattr(l, 'density', 1) for l in self.lstack])
        cd = self.cd
        qzvec = 4 * numpy.pi * numpy.sin(numpy.radians(lai)) /\
            utilities.en2lam(self.energy)
        qvec = numpy.array([[0., 0., qz] for qz in qzvec])
        chihP = numpy.array([[l.material.chih(q, en=self.energy,
                                              polarization=self.polarization)
                              for q in qvec]
                             for l in self.lstack])

        if self.polarization in ['S', 'P']:
            cd = cd + chihP
        else:
            cd = cd

        sai = numpy.sin(numpy.radians(lai))

        if self.beam_width > 0:
            shape = self.sample_width * sai / self.beam_width
            shape[shape > 1] = 1
        else:
            shape = numpy.ones(np)

        ETs = numpy.ones(np, dtype=numpy.complex)
        ERs = numpy.zeros(np, dtype=numpy.complex)
        ks = -self.exp.k0 * numpy.sqrt(sai**2 - 2 * cd[0] * rho[0])

        for i in range(ns):
            if i < ns-1:
                k = -self.exp.k0 * numpy.sqrt(sai**2 - 2 * cd[i+1] * rho[i+1])
                phi = numpy.exp(1j * k * t[i+1])
            else:
                k = -self.exp.k0 * sai
                phi = numpy.ones(np)
            r = (k - ks) / (k + ks) * numpy.exp(-2 * sig[i]**2 * k * ks)
            ET = phi * (ETs + r * ERs)
            ER = (r * ETs + ERs) / phi
            ETs = ET
            ERs = ER
            ks = k

        R = shape * abs(ER / ET)**2
        return self.scale_simulation(self.convolute_resolution(lai, R))


class DiffuseReflectivityModel(SpecularReflectivityModel):
    """
    model for diffuse reflectivity calculations

    The 'simulate' method calculates the diffuse reflectivity on the specular
    rod in coplanar geometry in analogy to the SpecularReflectivityModel.

    The 'simulate_map' method calculates the diffuse reflectivity for a 2D set
    of Q-positions. This method can also calculate the intensity for other
    geometries, like GISAXS with constant incidence angle or a quasi
    omega/2theta scan in GISAXS geometry.
    """

    def __init__(self, *args, **kwargs):
        """
        constructor for a reflectivity model. The arguments consist of a
        LayerStack or individual Layer(s). Optional parameters are specified
        in the keyword arguments.

        Parameters
        ----------
        args :     LayerStack or Layers
            either one LayerStack or several Layer objects can be given
        kwargs :   dict
            optional parameters for the simulation; supported are:
        I0 :        float, optional
            the primary beam intensity
        background : float, optional
            the background added to the simulation
        sample_width : float, optional
            width of the sample along the beam
        beam_width : float, optional
            beam width in the same units as the sample width
        resolution_width : float, optional
            defines the width of the resolution (deg)
        energy :    float, optional
            sets the experimental energy (eV)
        H :         float, optional
            Hurst factor defining the fractal dimension of the roughness (0..1,
            very slow for H != 1 or H != 0.5), default: 1
        vert_correl : float, optional
            vertical correlation length in (Angstrom), 0 means full replication
        vert_nu :   float, optional
            exponent in the vertical correlation function
        method :    int, optional
            1..simple DWBA (default), 2..full DWBA (slower)
        vert_int :  int, optional
            0..no integration over the vertical divergence, 1..with integration
            over the vertical divergence
        qL_zero :   float, optional
            value of inplane q-coordinate which can be considered 0, using
            method 2 it is important to avoid exact 0 and this value will be
            used instead
        """
        if not hasattr(self, 'fit_paramnames'):
            self.fit_paramnames = []
        self.fit_paramnames += ['H', 'vert_correl', 'vert_nu']
        self.H = kwargs.pop('H', 1)
        self.vert_correl = kwargs.pop('vert_correl', 0)
        self.vert_nu = kwargs.pop('vert_nu', 0)
        self.method = kwargs.pop('method', 1)
        self.vert = kwargs.pop('vert_int', 0)
        self.qL_zero = kwargs.pop('qL_zero', 5e-5)
        super(DiffuseReflectivityModel, self).__init__(*args, **kwargs)
        self.lstack_params += ['lat_correl', ]

    def _get_layer_prop(self):
        """
        helper function to obtain layer properties needed for all types of
        simulations
        """
        nl = len(self.lstack)
        self.init_cd()

        t = numpy.asarray([float(l.thickness) for l in self.lstack[nl:0:-1]])
        sig = [float(getattr(l, 'roughness', 0)) for l in self.lstack[nl::-1]]
        rho = [float(getattr(l, 'density', 1)) for l in self.lstack[nl::-1]]
        delta = self.cd * numpy.asarray(rho)
        xiL = [float(getattr(l, 'lat_correl', numpy.inf))
               for l in self.lstack[nl::-1]]

        return t, sig, rho, delta, xiL

    def simulate(self, alphai):
        """
        performs the actual diffuse reflectivity calculation for the specified
        incidence angles. This method always uses the coplanar geometry
        independent of the one set during the initialization.

        Parameters
        ----------
        alphai :    array-like
            vector of incidence angles

        Returns
        -------
        array-like
            vector of intensities of the reflectivity signal
        """
        lai = alphai - self.offset
        # get layer properties
        t, sig, rho, delta, xiL = self._get_layer_prop()

        deltaA = numpy.sum(delta[:-1]*t)/numpy.sum(t)
        lam = utilities.en2lam(self.energy)
        if self.method == 2:
            qL = [-abs(self.qL_zero), abs(self.qL_zero)]
        else:
            qL = [0, ]
        qz = 4 * numpy.pi / lam * numpy.sin(numpy.radians(lai))
        R = self._xrrdiffv2(lam, delta, t, sig, xiL, self.H, self.vert_correl,
                            self.vert_nu, None, qL, qz, self.sample_width,
                            self.beam_width, 1e-4, 1000, deltaA, self.method,
                            1, self.vert)
        R = R.mean(axis=0)
        return self.scale_simulation(self.convolute_resolution(lai, R))

    def simulate_map(self, qL, qz):
        """
        performs diffuse reflectivity calculation for the rectangular grid of
        reciprocal space positions define by qL and qz.  This method uses the
        method and geometry set during the initialization of the class.

        Parameters
        ----------
        qL :    array-like
            lateral coordinate in reciprocal space (vector with NqL components)
        qz :    array-like
            vertical coordinate in reciprocal space (vector with Nqz
            components)

        Returns
        -------
        array-like
            matrix of intensities of the reflectivity signal, with shape
            (len(qL), len(qz))
        """
        # get layer properties
        t, sig, rho, delta, xiL = self._get_layer_prop()

        deltaA = numpy.sum(delta[:-1]*t)/numpy.sum(t)
        lam = utilities.en2lam(self.energy)
        localqL = numpy.copy(qL)
        if self.method == 2:
            localqL[qL == 0] = self.qL_zero

        R = self._xrrdiffv2(lam, delta, t, sig, xiL, self.H, self.vert_correl,
                            self.vert_nu, None, localqL, qz, self.sample_width,
                            self.beam_width, 1e-4, 1000, deltaA, self.method,
                            1, self.vert)
        return self.scale_simulation(R)

    def _xrrdiffv2(self, lam, delta, thick, sigma, xiL, H, xiV, nu, alphai, qL,
                   qz, samplewidth, beamwidth, eps, nmax, deltaA, method, scan,
                   vert):
        """
        simulation of diffuse reflectivity from a rough multilayer. Exact or
        simplified DWBA, fractal roughness model, Ming model of the vertical
        correlation, various scattering geometries

        The used incidence and exit angles are stored in _smap_alphai,
        _smap_alphaf

        Parameters
        ----------
        lam :       float
            x-ray wavelength in Angstrom
        delta :     list or array-like
            vector with the 1-n values (N+1 components, 1st component..layer at
            the free surface, last component..substrate)
        thick :     list or array-like
            vector with thicknesses (N components)
        sigma :     list or array-like
            vector with rms roughnesses (N+1 components)
        xiL :       list or array-like
            vector with lateral correlation lengths (N+1 components)
        H :         float
            Hurst factor (scalar)
        xiV :       float
            vertical correlation: 0..full replication, > 0 and nu > 0.. see
            below, > 0 and nu = 0..vertical correlation length
        nu :        float
            exponent in the vertical correlation function
            exp(-abs(z_m-z_n)*(qL/max(qL))**nu/xiV)
        alphai :    float
            incidence angle (scalar for scan=2, ignored for scan=1, 3)
        qL :        array-like
            lateral coordinate in reciprocal space (vector with NqL components)
        qz :        array-like
            vertical coordinate in reciprocal space (vector with Nqz
            components)
        samplewidth : float
            width of the irradiated sample area (scalar), =0..the irradiated
            are is assumed constant
        beamwidth : float
            width of the primary beam
        eps :       float
            small number
        nmax :      int
            max number of terms in the Taylor series of the lateral correlation
            function
        deltaA :    complex
            effective value of 1-n in simple DWBA (ignored for method=2)
        method :    int
            1..simple DWBA, 2..full DWBA
        scan :      int
            1..standard coplanar geometry, 2..standard GISAXS geometry with
            constant incidence angle, =3..quasi omega/2theta scan in GISAXS
            geometry (incidence and central-exit angles are equal)
        vert :      int
            0..no integration over the vertical divergence, 1..with integration
            over the vertical divergence

        Returns
        -------
        diffint :   array-like
            diffuse reflectivity intensity matrix
        """
        # worker function definitions
        def coherent(alphai, K, delta, thick, N, NqL, Nqz):
            """
            calculate coherent reflection/transmission signal of a multilayer

            Parameters
            ----------
            alphai :    array-like
                matrix of incidence angles in radians (NqL x Nqz components)
            K :         float
                x-ray wave-vector (2*pi/lambda)
            delta :     array-like
                vector with the 1-n values (N+1 components, 1st
                component..layer at the free surface, last
                component..substrate)
            thick :     array-like
                vector with thicknesses (N components)
            N :         int
                number layers in the stack
            NqL :       int
                number of lateral q-points to calculate
            Nqz :       int
                number of vertical q-points to calculate

            Returns
            -------
            T, R, R0, k0, kz : array-like
                transmission, reflection, surface reflection, z-component of
                k-vector, z-component of k-vector in the material.
            """
            k0 = -K * numpy.sin(alphai)
            kz = numpy.zeros((N+1, NqL, Nqz), dtype=numpy.complex)
            T = numpy.zeros((N+1, NqL, Nqz), dtype=numpy.complex)
            R = numpy.zeros((N+1, NqL, Nqz), dtype=numpy.complex)
            for jn in range(N+1):
                kz[jn, ...] = -K * numpy.sqrt(numpy.sin(alphai)**2 -
                                              2 * delta[jn])

            T[N, ...] = numpy.ones((NqL, Nqz), dtype=numpy.complex)
            kzs = kz[N, ...]  # kz in substrate
            for jn in range(N-1, -1, -1):
                kzn = kz[jn, ...]
                tF = 2 * kzn / (kzn + kzs)
                rF = (kzn - kzs) / (kzn + kzs)
                phi = numpy.exp(1j * kzn * thick[jn])
                T[jn, ...] = phi / tF * (T[jn+1, ...] + rF * R[jn+1, ...])
                R[jn, ...] = 1 / phi / tF * (rF * T[jn+1, ...] + R[jn+1, ...])
                kzs = numpy.copy(kzn)

            tF = 2 * k0 / (k0 + kzn)
            rF = (k0 - kzn) / (k0 + kzn)
            T0 = 1 / tF * (T[0, ...] + rF * R[0, ...])
            R0 = 1 / tF * (rF * T[0, ...] + R[0, ...])

            T /= T0
            R /= T0
            R0 /= T0

            return T, R, R0, k0, kz

        def correl(a, b, L, H, eps, nmax, vert, K, NqL, Nqz, isurf):
            """
            correlation function

            Parameters
            ----------
            a :     array-like
                lateral correlation parameter
            b :     array-like
                vertical correlation parameter
            L :     float
                lateral correlation length
            H :     float
                Hurst factor (scalar)
            eps :   float
                small number (decides integration cut-off), typical 1e-3
            nmax :  int
                max number of terms in the Taylor series of the lateral
                correlation function
            vert :  int
                flag to tell decide if integration over vertical divergence is
                used: 0..no integration, 1..with integration
            K :     float
                length of the x-ray wave-vector (2*pi/lambda)
            NqL :   int
                number of lateral q-points to calculate
            Nqz :   int
                number of vertical q-points to calculate
            isurf : array-like
                array with NqL, Nqz flags to tell if there is a positive
                incidence and exit angle

            Returns
            -------
            psi :   array-like
                correlation function
            """
            psi = numpy.zeros((NqL, Nqz), dtype=numpy.complex)
            if H == 0.5 or H == 1:
                dpsi = numpy.zeros_like(psi, dtype=numpy.complex)
                m = isurf > 0
                n = 1
                s = numpy.copy(b)
                errm = numpy.inf
                if H == 1 and vert == 0:
                    def f(a, n):
                        return numpy.exp(-a**2/4/n) / 2 / n**2
                elif H == 0.5 and vert == 0:
                    def f(a, n):
                        return 1. / (n**2 + a**2)**(3/2.)
                elif H == 1 and vert == 1:
                    def f(a, n):
                        return numpy.sqrt(numpy.pi/n**3) * numpy.exp(-a**2/4/n)
                elif H == 0.5 and vert == 1:
                    def f(a, n):
                        return 2. / (n**2 + a**2)

                while errm > eps and n < nmax:
                    dpsi[m] = s[m] * f(a[m], n)
                    if n > 1:
                        errm = abs(numpy.max(dpsi[m] / psi[m]))
                    psi[m] += dpsi[m]
                    s[m] *= b[m]/float(n)
                    n += 1
            else:
                if vert == 0:
                    kern = kernel
                else:
                    kern = kernelvert
                for jL in range(NqL):
                    for jz in range(Nqz):
                        if isurf[jL, jz] == 1:
                            xmax = (-numpy.log(eps / b[jL, jz]))**(1/(2*H))
                            psi[jL, jz] = cquad(kern, 0.0, numpy.real(xmax),
                                                epsrel=eps, epsabs=0,
                                                limit=nmax, args=(a[jL, jz],
                                                b[jL, jz], H))

            if vert == 0:
                psi *= 2 * numpy.pi * L ** 2
            else:
                psi *= 2 * numpy.pi * L / K
            return psi

        def kernelvert(x, a, b, H):
            """
            integration kernel with vertical integration

            Parameters
            ----------
            x :     float or array-like
                independent parameter of the function
            a :     float
                lateral correlation parameter
            b :     complex
                vertical correlation parameter
            H :     float
                Hurst factor (scalar)

            Returns
            -------
            float or arraylike
            """
            w = numpy.exp(b * numpy.exp(-x**(2*H))) - 1
            F = 2 * numpy.cos(a*x) * w
            return F

        def kernel(x, a, b, H):
            """
            integration kernel without vertical integration

            Parameters
            ----------
            x :     float or array-like
                independent parameter of the function
            a :     float
                lateral correlation parameter
            b :     complex
                vertical correlation parameter
            H :     float
                Hurst factor (scalar)

            Returns
            -------
            float or arraylike
            """
            w = numpy.exp(b * numpy.exp(-x**(2*H))) - 1
            F = x * j0(a*x) * w
            return F

        def cquad(func, a, b, **kwargs):
            """
            complex quadrature by spliting real and imaginary part using scipy
            """
            def real_func(*args):
                return numpy.real(func(*args))

            def imag_func(*args):
                return numpy.imag(func(*args))
            real_integral = integrate.quad(real_func, a, b, **kwargs)
            imag_integral = integrate.quad(imag_func, a, b, **kwargs)
            return (real_integral[0] + 1j*imag_integral[0])

        # begin of _xrrdiffv2
        K = 2 * numpy.pi / lam

        N = len(thick)
        NqL = len(qL)
        Nqz = len(qz)

        QZ, QL = numpy.meshgrid(qz, qL)

        # scan types:
        if scan == 1:  # coplanar geometry
            Q = numpy.sqrt(QL**2 + QZ**2)
            QP = numpy.abs(QL)
            th = numpy.arcsin(Q / 2 / K)
            om = numpy.arctan2(QL, QZ)
            ALPHAI = th + om
            ALPHAF = th - om
        elif scan == 2:  # GISAXS geometry with constant incidence angle
            ALPHAI = numpy.radians(alphai) * numpy.ones((NqL, Nqz))
            ALPHAF = numpy.arcsin(QZ / K - numpy.sin(numpy.radians(alphai)))
            PHI = numpy.arcsin(QL / K / numpy.cos(ALPHAF))
            QP = K * numpy.sqrt(numpy.cos(ALPHAF)**2 + numpy.cos(ALPHAI)**2 -
                                2*numpy.cos(ALPHAF) * numpy.cos(ALPHAI) *
                                numpy.cos(PHI))
        elif scan == 3:  # with quasi omega/2theta scan in GISAXS geometry
            ALPHAI = numpy.arcsin(QZ * (K - numpy.sqrt(K**2 - QL**2)) / QL**2)
            ALPHAF = numpy.arcsin(QZ / K - numpy.sin(ALPHAI))
            PHI = numpy.arcsin(QL / K / numpy.cos(ALPHAF))
            QP = K * numpy.sqrt(numpy.cos(ALPHAF)**2 + numpy.cos(ALPHAI)**2 -
                                2*numpy.cos(ALPHAF) * numpy.cos(ALPHAI) *
                                numpy.cos(PHI))
        else:
            raise ValueError("Invalid value of parameter 'scan'")

        # removing the values under the horizon
        isurf = heaviside(ALPHAI) * heaviside(ALPHAF)

        # non-disturbed states:
        if method == 1:
            k01 = -K * numpy.sin(ALPHAI)
            kz1 = -K * numpy.sqrt(numpy.sin(ALPHAI)**2 - 2*deltaA)
            k02 = -K * numpy.sin(ALPHAF)
            kz2 = -K * numpy.sqrt(numpy.sin(ALPHAF)**2 - 2*deltaA)
            T1 = 2 * k01 / (k01 + kz1)
            T2 = 2 * k02 / (k02 + kz2)
            R01 = (k01 - kz1) / (k01 + kz1)
            R02 = (k02 - kz2) / (k02+kz2)
            R1 = numpy.zeros((NqL, Nqz), dtype=numpy.complex)
            R2 = numpy.copy(R1)
            nproc = 1
        else:  # method == 2
            T1, R1, R01, k01, kz1 = coherent(ALPHAI, K, delta, thick, N,
                                             NqL, Nqz)
            T2, R2, R02, k02, kz2 = coherent(ALPHAF, K, delta, thick, N,
                                             NqL, Nqz)
            nproc = 4

        # sample surface
        if beamwidth > 0:
            S = samplewidth * numpy.sin(ALPHAI) / beamwidth
            S[S > 1] = 1
        else:
            S = 1

        # z-coordinates
        z = numpy.zeros(N+1)
        for jn in range(1, N+1):
            z[jn] = z[jn-1] - thick[jn-1]

        # calculation of the deltas
        delt = numpy.zeros(N+1, dtype=numpy.complex)
        for jn in range(N+1):
            if jn == 0:
                delt[jn] = delta[jn]
            if jn > 0:
                delt[jn] = delta[jn] - delta[jn-1]

        # double sum over interfaces
        result = numpy.zeros((NqL, Nqz))
        for jn in range(N+1):
            # if method == 1 and (H == 1 or H == 0.5):
            #     print(jn)
            if nu != 0 or xiV == 0:
                jmdol = 1
            else:
                jmdol = numpy.argmin(numpy.abs(z - (z[jn] -
                                               xiV * numpy.log(eps))))

            for ja in range(nproc):
                if method == 1:
                    Qn = -kz1 - kz2
                    An = T1 * T2 * numpy.exp(-1j*Qn*z[jn])
                else:  # method == 2
                    if ja == 0:
                        An = T1[jn, ...] * T2[jn, ...]
                        Qn = -kz1[jn, ...] - kz2[jn, ...]
                    elif ja == 1:
                        An = T1[jn, ...] * R2[jn, ...]
                        Qn = -kz1[jn, ...] + kz2[jn, ...]
                    elif ja == 2:
                        An = R1[jn, ...] * T2[jn, ...]
                        Qn = kz1[jn, ...] - kz2[jn, ...]
                    elif ja == 3:
                        An = R1[jn, ...] * R2[jn, ...]
                        Qn = kz1[jn, ...] + kz2[jn, ...]
                for jm in range(jmdol, jn+1):
                    if jm == jn:
                        weight = 1
                    else:
                        weight = 2
                    # if method == 1 and (H != 0.5 and H != 1) and ja==1:
                    #     print(jn, jm)
                    # vertical correlation function:
                    if xiV > 0:
                        CV = numpy.exp(-abs(z[jn] - z[jm]) *
                                       (QP/numpy.max(QP))**nu / xiV)
                    else:
                        CV = 1
                    # effective values of sigma and lateral correl. length:
                    try:
                        LP = ((float(xiL[jn])**(-2*H) +
                               float(xiL[jm])**(-2*H)) / 2) ** (-1 / 2 / H)
                    except ZeroDivisionError:
                        LP = 0
                    sig = pymath.sqrt(sigma[jn] * sigma[jm])
                    for jb in range(nproc):
                        if method == 1:
                            Qm = -kz1 - kz2
                            Am = T1 * T2 * numpy.exp(-1j * Qm * z[jm])
                        else:  # method == 2
                            # if H != 0.5 or H != 1:
                            #    print(ja, jb, jn, jm)
                            if jb == 0:
                                Am = T1[jm, ...] * T2[jm, ...]
                                Qm = -kz1[jm, ...] - kz2[jm, ...]
                            elif jb == 1:
                                Am = T1[jm, ...] * R2[jm, ...]
                                Qm = -kz1[jm, ...] + kz2[jm, ...]
                            elif jb == 2:
                                Am = R1[jm, ...] * T2[jm, ...]
                                Qm = kz1[jm, ...] - kz2[jm, ...]
                            elif jb == 3:
                                Am = R1[jm, ...] * R2[jm, ...]
                                Qm = +kz1[jm, ...] + kz2[jm, ...]
                        # lateral correlation function:
                        Psi = correl(QP*LP, Qn*numpy.conj(Qm)*sig**2, LP, H,
                                     eps, nmax, vert, K, NqL, Nqz, isurf)
                        result += numpy.real(CV * delt[jn] *
                                             numpy.exp(-Qn**2 *
                                                       sigma[jn]**2/2) /
                                             Qn * An *
                                             numpy.conj(delt[jm] *
                                             numpy.exp(-Qm**2*sigma[jm]**2/2) /
                                             Qm * Am) * Psi) * weight

        result[isurf == 0] = 0
        self._smap_R01 = R01 * isurf
        self._smap_R02 = R02 * isurf
        self._smap_alphai = numpy.degrees(ALPHAI*isurf)
        self._smap_alphaf = numpy.degrees(ALPHAF*isurf)
        return result * S * K**4 / (16*numpy.pi**2)
