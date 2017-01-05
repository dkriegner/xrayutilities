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

from .smaterials import Powder, PowderList
from .powder import PowderDiffraction
from .. import materials


class PowderModel(object):
    """
    Class to help with powder calculations for multiple materials.  For basic
    calculations the Powder class together with the Fundamental parameters
    approach is used.
    """
    def __init__(self, *args, **kwargs):
        """
        constructor for a powder model. The arguments consist of a PowderList
        or individual Powder(s). Optional parameters are specified in the
        keyword arguments.

        Parameters
        ----------
         *args:    either one PowderList or several Powder objects can be given
         *kwargs:  optional parameters for the simulation. supported are:
           fpclass:  FP_profile derived class with possible convolver mixins.
                     (default: FP_profile)
           fpsettings: settings dictionaries for the convolvers. Default
                       settings are loaded from the config file.
           I0:     scaling factor for the simulation result
           background: NotImplemented

        In particular interesting in fpsettings might be:
        {'displacement': {'specimen_displacement': z-displacement of the sample
                                                   from the rotation center
                          'zero_error_deg': zero error of the 2theta angle}
         'absorption': {'sample_thickness': sample thickness (m),
                        'absorption_coefficient': sample's absorption (m^-1)}
         'axial': {'length_sample': the length of the sample in the axial
                                    direction (m)}
        }
        """
        if len(args) == 1 and isinstance(args[0], PowderList):
            self.materials = args[0]
        else:
            self.materials = PowderList('%s List' % self.__class__.__name__,
                                        *args)
        self.I0 = kwargs.pop('I0', 1.0)
        self.pdiff = []
        for mat in self.materials:
            self.pdiff.append(PowderDiffraction(mat, **kwargs))

    def set_parameters(self, params):
        """
        set simulation parameters of all subobjects

        Parameters
        ----------
         params:    settings dictionaries for the convolvers.
        """

        if 'emission' in params:
            if 'emiss_wavelength' in params['emission']:
                wl = params['emission']['emiss_wavelengths'][0]
                if 'global' in params:
                    params['global']['dominant_wavelength'] = wl
                else:
                    params['global'] = {'dominant_wavelength': wl}

                # set wavelength in base class
                for pd in self.pdiff:
                    pd._set_wavelength(wl*1e10)

        # set parameters for each convolver
        for k in params:
            for pd in self.pdiff:
                pd.fp_profile.set_parameters(convolver=k, **params[k])

    def simulate(self, twotheta, **kwargs):
        """
        calculate the powder diffraction pattern of all materials and sum the
        results based on the relative volume of the materials.

        Parameters
        ----------
         **kwargs: additional keyword arguments are passed to the Convolve
                   function of of the PowderDiffraction objects

        Returns
        -------
         summed powder diffraction intensity of all materials present in the
         model

        Known issue: possibility to add a background is currently missing!
        """
        inte = numpy.zeros_like(twotheta)
        totalvol = sum(pd.mat.volume for pd in self.pdiff)
        for pd in self.pdiff:
            inte += pd.Calculate(twotheta, **kwargs) * pd.mat.volume / totalvol
        return self.I0 * inte

    def __str__(self):
        """
        string representation of the PowderModel
        """
        ostr = "PowderModel {\n"
        ostr += "I0: %f\n" % self.I0
        ostr += str(self.materials)
        ostr += "}"
        return ostr
