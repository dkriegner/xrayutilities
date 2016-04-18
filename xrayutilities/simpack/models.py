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
import scipy.interpolate as interpolate
import scipy.constants as constants
from scipy.special import erf

from . import Layer, LayerStack
from .. import utilities
from .. import config
from ..math import heaviside
from ..math import NormGauss1d
from ..experiment import Experiment


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
        """
        for kw in kwargs:
            if kw not in ('resolution_width', 'I0', 'background', 'energy'):
                raise TypeError('%s is an invalid keyword argument' % kw)

        if experiment:
            self.exp = experiment
        else:
            self.exp = Experiment([1, 0, 0], [0, 0, 1])
        self.resolution_width = kwargs.get('resolution_width', 0)
        self.I0 = kwargs.get('I0', 1)
        self.background = kwargs.get('background', 0)
        if 'energy' in kwargs:
            self.exp.energy = kwargs['energy']

    def convolute_resolution(self, x, y):
        """
        convolve simulation result with a Gaussian resolution function

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
            nres = int(10 * numpy.abs(self.resolution_width / dx))
            xres = startdelta(-5*self.resolution_width, dx, nres + 1)
            # the following works only exactly for equally spaced data points
            resf = NormGauss1d(xres, numpy.mean(xres), self.resolution_width)
            resf /= numpy.sum(resf)  # proper normalization for discrete conv.
            # pad y to avoid edge effects
            interp = interpolate.InterpolatedUnivariateSpline(
                x, y, k=1, ext=0, check_finite=False)
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

    def simulate(self, qz, hkl, absorption=False, refraction=False):
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

        Returns
        -------
         vector of the ratios of the diffracted and primary fluxes
        """
        nl, nq = (len(self.lstack), len(qz))
        rel = constants.physical_constants['classical electron radius'][0]
        rel *= 1e10
        k = self.exp.k0

        # determine q-inplane
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*hkl))
        qinp = numpy.sqrt(ql0[0]**2 + ql0[1]**2)

        # prepare calculation
        qv = numpy.asarray([t.inverse((ql0[0], ql0[1], q)) for q in qz])
        Q = numpy.linalg.norm(qv, axis=1)
        theta = numpy.arcsin(Q / (2 * k))
        domega = numpy.arctan2(qinp, qz)
        alphai, alphaf = (theta + domega, theta - domega)
        valid = heaviside(alphai) * heaviside(alphaf)
        # calculate structure factors
        f = numpy.empty((nl, nq), dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            m = l.material
            f[i, :] = m.StructureFactorForQ(qv, en0=self.exp.energy) /\
                m.lattice.UnitCellVolume()
        # calculate interface positions
        z = numpy.zeros(nl)
        for i, l in enumerate(self.lstack[-1:0:-1]):
            z[-i-2] = z[-i-1] - l.thickness

        # perform kinematical calculation
        E = numpy.zeros(nq, dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            q = qz.astype(numpy.complex)
            if absorption and not refraction:
                q += 1j * k * numpy.imag(self.chi0[i]) / numpy.sin(theta)
            if refraction:
                q = k * (numpy.sqrt(numpy.sin(alphai)**2 + self.chi0[i]) +
                         numpy.sqrt(numpy.sin(alphaf)**2 + self.chi0[i]))
            q -= t(l.material.Q(*hkl))[-1]

            if l.thickness == numpy.inf:
                E += f[i, :] * numpy.exp(-1j * z[i] * q) / (1j * q)
            else:
                E += - f[i, :] * numpy.exp(-1j * q * z[i]) * \
                    (1 - numpy.exp(1j * q * l.thickness)) / (1j * q)

        w = valid * rel**2 / (numpy.sin(alphai) * numpy.sin(alphaf)) *\
            numpy.abs(E)**2
        return self.scale_simulation(self.convolute_resolution(qz, w))


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

    def simulate(self, qz, hkl, absorption=False, refraction=True):
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

        Returns
        -------
         vector of the ratios of the diffracted and primary fluxes
        """
        nl, nq = (len(self.lstack), len(qz))
        rel = constants.physical_constants['classical electron radius'][0]
        rel *= 1e10
        k = self.exp.k0

        # determine q-inplane
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*hkl))
        qinp = numpy.sqrt(ql0[0]**2 + ql0[1]**2)

        # prepare calculation
        qv = numpy.asarray([t.inverse((ql0[0], ql0[1], q)) for q in qz])
        Q = numpy.linalg.norm(qv, axis=1)
        theta = numpy.arcsin(Q / (2 * k))
        domega = numpy.arctan2(qinp, qz)
        alphai, alphaf = (theta + domega, theta - domega)
        valid = heaviside(alphai) * heaviside(alphaf)

        # calculate structure factors
        f = numpy.empty((nl, nq), dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            m = l.material
            f[i, :] = m.StructureFactorForQ(qv, en0=self.exp.energy) /\
                m.lattice.UnitCellVolume()

        # calculate interface positions
        z = numpy.zeros(nl)
        for i, l in enumerate(self.lstack[-1:0:-1]):
            lat = l.material.lattice
            a3 = t(lat.GetPoint(*self.surface_hkl))[-1]
            n3 = l.thickness // a3
            z[-i-2] = z[-i-1] - a3 * n3
            if config.VERBOSITY >= config.INFO_LOW and \
                    numpy.abs(l.thickness/a3 - n3) > 0.01:
                print('XU.KinematicMultiBeamModel: %s thickness changed from'
                      ' %.2fÅ to %.2fÅ (%d UCs)' % (l.name, l.thickness,
                                                    a3 * n3, n3))

        # perform kinematical calculation
        E = numpy.zeros(nq, dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            q = qz.astype(numpy.complex)
            if absorption and not refraction:
                q += 1j * k * numpy.imag(self.chi0[i]) / numpy.sin(theta)
            if refraction:
                q = k * (numpy.sqrt(numpy.sin(alphai)**2 + self.chi0[i]) +
                         numpy.sqrt(numpy.sin(alphaf)**2 + self.chi0[i]))
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

        w = valid * rel**2 / (numpy.sin(alphai) * numpy.sin(alphaf)) *\
            numpy.abs(E)**2
        return self.scale_simulation(self.convolute_resolution(qz, w))


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
         hkl:   Miller indices of the Bragg peak for the calculation
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
            for pol in ('S', 'P'):
                ch = l.material.chih(q, en=self.exp.energy, polarization=pol)
                self.chih[pol].append(-ch[0] + 1j*ch[1])
                ch = l.material.chih(-q, en=self.exp.energy, polarization=pol)
                self.chimh[pol].append(-ch[0] + 1j*ch[1])

        for pol in ('S', 'P'):
            self.chih[pol] = numpy.asarray(self.chih[pol])
            self.chimh[pol] = numpy.asarray(self.chimh[pol])

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

        # determine q-inplane
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*self.hkl))
        hx = numpy.sqrt(ql0[0]**2 + ql0[1]**2)
        if geometry == 'lo_hi':
            hx = -hx

        # calculate vertical diffraction vector components and strain
        hz = numpy.zeros(len(self.lstack))
        for i, l in enumerate(self.lstack):
            hz[i] = t(l.material.Q(*self.hkl))[2]
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
    def simulate(self, alphai, hkl=None, geometry='hi_lo'):
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

        Returns
        -------
         vector of intensities of the diffracted signal
        """
        if hkl is not None:
            self.set_hkl(hkl)

        # return values
        Ih = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}
        Ir = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}

        # determine q-inplane
        t = self.exp._transform
        ql0 = t(self.lstack[0].material.Q(*self.hkl))
        hx = numpy.sqrt(ql0[0]**2 + ql0[1]**2)
        if geometry == 'lo_hi':
            hx = -hx

        # calculate vertical diffraction vector components and strain
        hz = numpy.zeros(len(self.lstack))
        for i, l in enumerate(self.lstack):
            hz[i] = t(l.material.Q(*self.hkl))[2]

        k = self.exp.k0
        kc = k * numpy.sqrt(1 + self.chi0)
        ai = numpy.radians(alphai)
        thetaB = numpy.arcsin(numpy.sqrt(hx**2 + hz[0]**2) / 2 / k)
        Kix = k * numpy.cos(ai)
        Kiz = -k * numpy.sin(ai)
        Khz = numpy.sqrt(k**2 - (Kix + hx)**2)
        pp = Khz / k
        mask = numpy.logical_and(pp > 0, pp < 1)
        # alphah = numpy.zeros(len(ai))  # exit angles
        # alphah[mask] = degrees(numpy.arcsin(pp[mask]))

        P = numpy.zeros((4, 4), dtype=numpy.complex)
        A = numpy.zeros(5, dtype=numpy.complex)
        for pol in self.get_polarizations():
            pom = k**4 * self.chih[pol] * self.chimh[pol]
            if config.VERBOSITY >= config.INFO_ALL:
                print('XU.DynamicalModel: calc. %s-polarization...' % (pol))
            for jal in range(len(mask)):
                if config.VERBOSITY >= config.DEBUG and jal % 1000 == 0:
                    print('%d / %d' % (jal, len(mask)))
                if not mask[jal]:
                    continue
                M = numpy.identity(4, dtype=numpy.complex)

                for i, l in enumerate(self.lstack[-1::-1]):
                    jL = len(self.lstack) - 1 - i
                    A[0] = 1.0
                    A[1] = 2 * hz[jL]
                    A[2] = (Kix[jal] + hx)**2 + hz[jL]**2 + Kix[jal]**2 -\
                        2 * kc[jL]**2
                    A[3] = 2 * hz[jL] * (Kix[jal]**2 - kc[jL]**2)
                    A[4] = (Kix[jal]**2 - kc[jL]**2) *\
                        ((Kix[jal] + hx)**2 + hz[jL]**2 - kc[jL]**2) - pom[jL]
                    X = numpy.roots(A)

                    jneg = numpy.imag(X) <= 0
                    jpos = numpy.imag(X) > 0
                    if numpy.sum(jneg) != 2 or numpy.sum(jpos) != 2:
                        raise ValueError("XU.DynamicalModel: wrong number of "
                                         "pos/neg solutions %d / %d ... "
                                         "aborting!" % (numpy.sum(jpos),
                                                        numpy.sum(jneg)))
                    kz = numpy.zeros(4, dtype=numpy.complex)
                    kz[:2] = X[jneg]
                    kz[2:] = X[jpos]
                    c = (Kix[jal]**2 + kz**2 - kc[jL]**2) / k**2 /\
                        self.chimh[pol][jL]
                    if jL > 0:
                        phi = numpy.diag(numpy.exp(1j * kz * l.thickness))
                    else:
                        phi = numpy.identity(4)

                    P[0, :] = [1, 1, 1, 1]
                    P[1, :] = c
                    P[2, :] = kz
                    P[3, :] = c * (kz + hz[jL])
                    if i == 0:
                        R = numpy.copy(P)
                    else:
                        R = numpy.linalg.inv(Ps).dot(P)
                    M = M.dot(R).dot(phi)
                    Ps = numpy.copy(P)

                B = numpy.asarray([[M[0, 0], M[0, 1], -1, 0],
                                   [M[1, 0], M[1, 1], 0, -1],
                                   [M[2, 0], M[2, 1], Kiz[jal], 0],
                                   [M[3, 0], M[3, 1], 0, -Khz[jal]]])
                C = numpy.asarray([1, 0, Kiz[jal], 0])
                E = numpy.linalg.inv(B).dot(C)
                Ir[pol][jal] = numpy.abs(E[2])**2
                Ih[pol][jal] = numpy.abs(E[3])**2 *\
                    numpy.abs(Khz[jal] / Kiz[jal])

        ret = self.join_polarizations(Ih['S'], Ih['P'])
        return self.scale_simulation(self.convolute_resolution(alphai, ret))


# CODE NOT FULLY TESTED AND ANYHOW USELESS
# class DynamicalSKinematicalLModel(SimpleDynamicalCoplanarModel):
#    """
#    Mixed dynamical and kinematical diffraction model for specular and
#    off-specular qz-scans. Calculation of the flux of reflected and diffracted
#    waves for general asymmetric coplanar diffraction from an arbitrary
#    pseudomorphic multilayer.  Signal from the semi-infinite substrate is
#    calculated using dynamical theory (2-beam theory (2 tiepoints, S and P
#    polarizations)) and the signal from the other layers is calculated
#    kinematically.
#
#    No restrictions are made for the surface orientation.
#
#    The first layer in the model is always assumed to be the semiinfinite
#    substrate indepentent of its given thickness
#    """
#    def simulate(self, alphai, hkl=None, geometry='hi_lo', layerpos='kin'):
#        """
#        performs the actual diffraction calculation for the specified
#        incidence angles.
#
#        Parameters
#        ----------
#         alphai:    vector of incidence angles (deg)
#         hkl:       Miller indices of the diffraction vector (preferable use
#                    set_hkl method to speed up repeated calculations of the
#                    same peak!)
#         geometry:  'hi_lo' for grazing exit (default) and 'lo_hi' for grazing
#                    incidence
#         layerpos:  either 'kin' or 'dyn'. Determines how the diffraction
#                    position of the layers is calculated. default 'kin' for
#                    kinematical. 'dyn' uses a dynamical formular which is more
#                    accurate close to the substrate.
#
#        Returns
#        -------
#         vector of intensities of the diffracted signal
#        """
#        if hkl is not None:
#            self.set_hkl(hkl)
#
#        # return values
#        Ih = {'S': numpy.zeros(len(alphai)), 'P': numpy.zeros(len(alphai))}
#
#        # determine q-inplane
#        t = self.exp._transform
#        ql0 = t(self.lstack[0].material.Q(*self.hkl))
#        hx = numpy.sqrt(ql0[0]**2 + ql0[1]**2)
#        if geometry == 'lo_hi':
#            hx = -hx
#
#        # calculate vertical diffraction vector components and strain
#        hz = numpy.zeros(len(self.lstack))
#        for i, l in enumerate(self.lstack):
#            hz[i] = t(l.material.Q(*self.hkl))[2]
#        epsilon = (hz[0] - hz) / hz
#
#        k = self.exp.k0
#        thetaB = numpy.arcsin(numpy.sqrt(hx**2 + hz[0]**2) / 2 / k)
#        # asymmetry angle
#        asym = numpy.arctan2(hx, hz[0])
#        gamma0 = numpy.sin(asym + thetaB)
#        gammah = numpy.sin(asym - thetaB)
#
#        # deviation of the incident beam from the kinematical maximum
#        ai = numpy.radians(alphai)
#        eta = ai - thetaB - asym
#
#        # calculate Interface positions
#        z = numpy.zeros(len(self.lstack))
#        for i, l in enumerate(self.lstack[-1:0:-1]):
#            z[-i-2] = z[-i-1] - l.thickness
#
#        for pol in self.get_polarizations():
#            x = numpy.zeros(len(alphai), dtype=numpy.complex)
#            for i, l in enumerate(self.lstack):
#                beta = (2 * eta * numpy.sin(2 * thetaB) +
#                        self.chi0[i] * (1 - gammah / gamma0) -
#                        2 * gammah * (gamma0 - gammah) * epsilon[i])
#                if i == 0:  # substrate
#                    y = beta / 2 /\
#                        numpy.sqrt(self.chih[pol][i] * self.chimh[pol][i]) /\
#                        numpy.sqrt(numpy.abs(gammah) / gamma0)
#                    c1 = -numpy.sqrt(self.chih[pol][i] / self.chih[pol][i] *
#                                     gamma0 / numpy.abs(gammah)) *\
#                        (y + numpy.sqrt(y**2 - 1))
#                    c2 = -numpy.sqrt(self.chih[pol][i] / self.chimh[pol][i] *
#                                     gamma0 / numpy.abs(gammah)) *\
#                        (y - numpy.sqrt(y**2 - 1))
#                    pp = numpy.abs(gammah) / gamma0 * numpy.abs(c1)**2
#                    m = pp < 1
#                    x[m] = c1[m]
#                    m = pp >= 1
#                    x[m] = c2[m]
#                else:  # layers
#                    if layerpos == 'dyn':
#                        qz = -k / 2 / gammah * beta
#                    else:  # kinematical alternative
#                        th = (numpy.arccos(hx / k + numpy.cos(ai)) + ai) / 2
#                        qz = 2 * k * numpy.sin(th) * numpy.cos(ai - th) -\
#                            t(l.material.Q(*self.hkl))[-1]
#                    x += k / 2 / gammah * self.chih[pol][i] / qz *\
#                        (1 - numpy.exp(-1j * qz * l.thickness)) *\
#                        numpy.exp(1j * z[i] * qz)
#
#            Ih[pol] = numpy.abs(x)**2
#
#        ret = self.join_polarizations(Ih['S'], Ih['P'])
#        return self.scale_simulation(self.convolute_resolution(alphai, ret))


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
