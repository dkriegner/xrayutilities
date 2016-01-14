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

from . import Layer, LayerStack
from .. import utilities
from ..math import heaviside
from ..math import NormGauss1d
from ..experiment import Experiment


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
            # the following works only exactly for equally spaced data points
            resf = NormGauss1d(x, numpy.mean(x), self.resolution_width)
            resf /= numpy.sum(resf)  # proper normalization for discrete conv.
            return numpy.convolve(y, resf, mode='same')

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
         *kwargs:   optional parameters for the simulation
                    supported are: 'experiment': Experiment class containing
                    geometry and energy of the experiment
        """
        exp = kwargs.get('experiment', None)
        super(LayerModel, self).__init__(exp, **kwargs)
        if len(args) == 1:
            self.lstack = args[0]
        else:
            self.lstack = LayerStack('Stack for %s' % self.__class__.__name__,
                                     *args)


class KinematicalModel(LayerModel):
    """
    Kinematical diffraction model for specular and off-specular qz-scans
    """
    def simulate(self, qz, hkl):
        """
        performs the actual kinematical diffraction calculation on the Qz
        positions specified considering the contribution from a single Bragg
        peak.

        Parameters
        ----------
         qz:    simulation positions along qz
         hkl:   Miller indices of the Bragg peak whos surrounding should be
                calculated

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
        print(ql0)
        qinp = numpy.sqrt(ql0[0]**2 + ql0[1]**2)

        # prepare calculation
        qv = numpy.asarray([t.inverse((ql0[0], ql0[1], q)) for q in qz])
        Q = numpy.linalg.norm(qv, axis=1)
        theta = numpy.arcsin(Q / (2 * k))
        omega = numpy.arctan(qinp / Q)
        alphai, alphaf = (theta + omega, theta-omega)
        valid = heaviside(alphai) * heaviside(alphaf)

        # calculate structure factors
        z = numpy.zeros(nl + 1)
        f = numpy.empty((nl, nq), dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            z[i+1] = z[i] - l.thickness
            f[i, :] = l.material.StructureFactorForQ(qv, en0=self.exp.energy)

        # perform kinematical calculation
        E = numpy.zeros(nq, dtype=numpy.complex)
        for i, l in enumerate(self.lstack):
            q = qz - self.exp.Transform(l.material.Q(*hkl))[-1]
            dE = (numpy.exp(-1j * z[i] * q) / q *
                  (1 - numpy.exp(1j * q * l.thickness)) * f[i, :])
            E += dE

        w = valid * rel**2 / (numpy.sin(alphai) * numpy.sin(alphaf)) *\
            numpy.abs(E)**2
        return self.scale_simulation(self.convolute_resolution(qz, w))


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
                    'beam_width' beam width in the same units as the sample width
                    'resolution_width' defines the width of the resolution (deg)
                    'energy' sets the experimental energy (eV)
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.sample_width = kwargs.get('sample_width', numpy.inf)
        self.beam_width = kwargs.get('beam_width', 0)
        # precalc optical properties
        self.init_chi0()

    def init_chi0(self):
        """
        calculates the needed optical parameters for the simulation. If any of
        the materials/layers is changing its properties this function needs to
        be called again before another correct simulation is made. (Changes of
        thickness and roughness do NOT require this!)
        """
        self.cd = numpy.asarray([-l.material.chi0()/2 for l in self.lstack])

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
