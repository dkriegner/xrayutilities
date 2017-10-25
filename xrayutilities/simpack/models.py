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

import numpy
import scipy.constants as constants
import scipy.interpolate as interpolate
from scipy.special import erf

from . import LayerStack
from .. import config
from ..exception import InputError
from ..experiment import Experiment
from ..math import NormGauss1d, NormLorentz1d, heaviside, solve_quartic


def startdelta(start, delta, num):
    end = start + delta * (num - 1)
    return numpy.linspace(start, end, num)


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
         experiment: Experiment class describing the diffraction geometry,
                     energy and wavelength of the model
         kwargs:     optional keyword arguments specifying model parameters.
                     'resolution_width' defines the width of the resolution
                     'I0' is the primary beam flux/intensity
                     'background' is the background added to the simulation
                     'energy' sets the experimental energy (in eV)
                     'resolution_type' sets the type of resolution function
                                 ('Gauss' (default) or 'Lorentz')
        """
        for kw in kwargs:
            if kw not in ('resolution_width', 'I0', 'background', 'energy',
                          'resolution_type'):
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
            self.exp.energy = kwargs['energy']

    def convolute_resolution(self, x, y):
        """
        convolve simulation result with a resolution function

        Parameters
        ----------
         x:  x-values of the simulation, units of x also decide about the
             unit of the resolution_width parameter
         y:  y-values of the simulation

        Returns
        -------
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
         y:  y-values of the simulation

        Returns
        -------
         scaled y values
        """
        return y * self.I0 + self.background


class LayerModel(Model):
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
         *args:     either one LayerStack or several Layer objects can be given
         *kwargs:   optional parameters for the simulation. supported are:
            'experiment': Experiment class containing geometry and energy of
                          the experiment.
            'surface_hkl': Miller indices of the surface (default: (001))
        """
        exp = kwargs.pop('experiment', None)
        super(LayerModel, self).__init__(exp, **kwargs)
        if len(args) == 1:
            self.lstack = args[0]
        else:
            self.lstack = LayerStack('Stack for %s' % self.__class__.__name__,
                                     *args)

    def _create_return(self, x, E, ai=None, af=None, Ir=None,
                       rettype='intensity'):
        """
        function to create the return value of a simulation. by default only
        the diffracted intensity is returned. However, optionally also the
        incidence and exit angle as well as the reflected intensity can be
        returned.

        Parameters
        ----------
         x:         independent coordinate value for the convolution with the
                    resolution function
         E:         electric field amplitude (complex)
         ai, af:    incidence and exit angle of the XRD beam (in radians)
         Ir:        reflected intensity
         rettype:   type of the return value. 'intensity' (default): returns
                    the diffracted beam flux convoluted with the resolution
                    function; 'field': returns the electric field (complex)
                    without convolution with the resolution function, 'all':
                    returns the electric field, ai, af (both in degree), and
                    the reflected intensity.

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
         *args:     either one LayerStack or several Layer objects can be given
         *kwargs:   optional parameters for the simulation. supported are:
            'experiment': Experiment class containing geometry and energy of
                          the experiment.
        """
        super(KinematicalModel, self).__init__(*args, **kwargs)
        # precalc optical properties
        self.init_chi0()

    def init_chi0(self):
        """
        calculates the needed optical parameters for the simulation. If any of
        the materials/layers is changing its properties this function needs to
        be called again before another correct simulation is made. (Changes of
        thickness does NOT require this!)
        """
        self.chi0 = numpy.asarray([l.material.chi0(en=self.exp.energy)
                                   for l in self.lstack])

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
            fhkl[i] = m.StructureFactor(m.Q(*hkl), en=self.exp.energy) /\
                m.lattice.UnitCellVolume()
            f[i, :] = m.StructureFactorForQ(qv, en0=self.exp.energy) /\
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
         qz:            simulation positions along qz
         hkl:           Miller indices of the Bragg peak whos truncation rod
                        should be calculated
         absorption:    flag to tell if absorption correction should be used
         refraction:    flag to tell if basic refraction correction should be
                        performed. If refraction is True absorption correction
                        is also included independent of the absorption flag.
         rettype:       type of the return value. 'intensity' (default):
                        returns the diffracted beam flux convoluted with the
                        resolution function; 'field': returns the electric
                        field (complex) without convolution with the resolution
                        function, 'all': returns the electric field, ai, af
                        (both in degree), and the reflected intensity.

        Returns
        -------
         vector of the ratios of the diffracted and primary fluxes
        """
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
         *args:     either one LayerStack or several Layer objects can be given
         *kwargs:   optional parameters for the simulation. supported are:
            'experiment': Experiment class containing geometry and energy of
                          the experiment.
            'surface_hkl': Miller indices of the surface (default: (001))
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
         qz:            simulation positions along qz
         hkl:           Miller indices of the Bragg peak whos truncation rod
                        should be calculated
         absorption:    flag to tell if absorption correction should be used
         refraction:    flag to tell if basic refraction correction should be
                        performed. If refraction is True absorption correction
                        is also included independent of the absorption flag.
         rettype:       type of the return value. 'intensity' (default):
                        returns the diffracted beam flux convoluted with the
                        resolution function; 'field': returns the electric
                        field (complex) without convolution with the resolution
                        function, 'all': returns the electric field, ai, af
                        (both in degree), and the reflected intensity.

        Returns
        -------
         vector of the ratios of the diffracted and primary fluxes
        """
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

    Note: This model should not be used in real life scenarios since the made
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
         *args:     either one LayerStack or several Layer objects can be given
         *kwargs:   optional parameters for the simulation; supported are:
                    'I0' is the primary beam intensity
                    'background' is the background added to the simulation
                    'resolution_width' defines the width of the resolution
                                       (deg)
                    'polarization' polarization of the x-ray beam, either 'S',
                                   'P' or 'both'. If set to 'both' also Cmono,
                                   the polarization factor of the monochromator
                                   should be set
                    'Cmono' polarization factor of the monochromator
                    'energy' sets the experimental energy (eV)
                    'experiment': Experiment class containing geometry of the
                                  sample; surface orientation!
        """
        self.polarization = kwargs.pop('polarization', 'S')
        self.Cmono = kwargs.pop('Cmono', 1)
        super(SimpleDynamicalCoplanarModel, self).__init__(*args, **kwargs)
        self.hkl = None
        self.chih = None
        self.chimh = None

    def set_hkl(self, *hkl):
        """
        To speed up future calculations of the same Bragg peak optical
        parameters can be pre-calculated using this function.

        Parameters
        ----------
         hkl:       Miller indices of the Bragg peak for the calculation
        """
        if len(hkl) < 3:
            hkl = hkl[0]
            if len(hkl) < 3:
                raise InputError("need 3 Miller indices")

        self.hkl = numpy.asarray(hkl)

        # calculate chih
        self.chih = {'S': [], 'P': []}
        self.chimh = {'S': [], 'P': []}
        for l in self.lstack:
            q = l.material.Q(self.hkl)
            thetaB = numpy.arcsin(numpy.linalg.norm(q) / 2 / self.exp.k0)
            ch = l.material.chih(q, en=self.exp.energy, polarization='S')
            self.chih['S'].append(-ch[0] + 1j*ch[1])
            self.chih['P'].append((-ch[0] + 1j*ch[1]) *
                                  numpy.abs(numpy.cos(2*thetaB)))
            if not getattr(l, 'inversion_sym', False):
                ch = l.material.chih(-q, en=self.exp.energy, polarization='S')
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
         alphai:    vector of incidence angles (deg)
         hkl:       Miller indices of the diffraction vector (preferable use
                    set_hkl method to speed up repeated calculations of the
                    same peak!)
         geometry:  'hi_lo' for grazing exit (default) and 'lo_hi' for grazing
                    incidence
         idxref:    index of the reference layer. In order to get accurate peak
                    position of the film peak you want this to be the index of
                    the film peak (default: 1). For the substrate use 0.

        Returns
        -------
         vector of intensities of the diffracted signal
        """
        if hkl is not None:
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
         alphai:    vector of incidence angles (deg)
         hkl:       Miller indices of the diffraction vector (preferable use
                    set_hkl method to speed up repeated calculations of the
                    same peak!)
         geometry:  'hi_lo' for grazing exit (default) and 'lo_hi' for grazing
                    incidence
         rettype:   type of the return value. 'intensity' (default): returns
                    the diffracted beam flux convoluted with the resolution
                    function; 'field': returns the electric field (complex)
                    without convolution with the resolution function, 'all':
                    returns the electric field, ai, af (both in degree), and
                    the reflected intensity.

        Returns
        -------
         vector of intensities of the diffracted signal
        """
        if len(self.get_polarizations()) > 1 and rettype != "intensity":
            raise ValueError('XU:DynamicalModel: return type (%s) not '
                             'supported with multiple polarizations!')
            rettype = 'intensity'
        if hkl is not None:
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
            B[:, :, :2] = M[:, :, :2]
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
         *args:     either one LayerStack or several Layer objects can be given
         *kwargs:   optional parameters for the simulation; supported are:
                    'I0' is the primary beam intensity
                    'background' is the background added to the simulation
                    'sample_width' width of the sample along the beam
                    'beam_width' beam width in the same units as the sample
                                 width
                    'resolution_width' defines the width of the resolution
                                       (deg)
                    'energy' sets the experimental energy (eV)
        """
        self.sample_width = kwargs.pop('sample_width', numpy.inf)
        self.beam_width = kwargs.pop('beam_width', 0)
        super(SpecularReflectivityModel, self).__init__(*args, **kwargs)
        # precalc optical properties
        self.init_cd()

    def init_cd(self):
        """
        calculates the needed optical parameters for the simulation. If any of
        the materials/layers is changing its properties this function needs to
        be called again before another correct simulation is made. (Changes of
        thickness and roughness do NOT require this!)
        """
        self.cd = numpy.asarray([-l.material.chi0(en=self.exp.energy)/2
                                 for l in self.lstack])

    def simulate(self, alphai):
        """
        performs the actual reflectivity calculation for the specified
        incidence angles

        Parameters
        ----------
         alphai: vector of incidence angles

        Returns
        -------
        vector of intensities of the reflectivity signal
        """
        ns, np = (len(self.lstack), len(alphai))

        # get layer properties
        t = numpy.asarray([l.thickness for l in self.lstack])
        sig = numpy.asarray([getattr(l, 'roughness', 0) for l in self.lstack])
        rho = numpy.asarray([getattr(l, 'density', 1) for l in self.lstack])
        cd = self.cd

        sai = numpy.sin(numpy.radians(alphai))

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
        return self.scale_simulation(self.convolute_resolution(alphai, R))

    def densityprofile(self, nz, plot=False):
        """
        calculates the electron density of the layerstack from the thickness
        and roughness of the individual layers

        Parameters
        ----------
         nz:    number of values on which the profile should be calculated
         plot:  flag to tell if a plot of the profile should be created

        Returns
        -------
         z, eprof:  coordinates and electron profile. z = 0 corresponds to the
                    surface
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
        zmin = -1 * totT - 10 * sig[0]
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
