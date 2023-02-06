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
# Copyright (C) 2009-2010 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (c) 2009-2023 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2012 Tanja Etzelstorfer <tanja.etzelstorfer@jku.at>

"""
module helping with planning and analyzing experiments.
various classes are provided for describing experimental geometries,
calculationof angular coordinates of Bragg reflections, conversion of angular
coordinates to Q-space and determination of powder diffraction peak positions.

The strength of the module is the versatile QConversion module which can be
configured to describe almost any goniometer geometry.
"""

import copy
import enum
import numbers
import re
import warnings

import numpy
from numpy.linalg import norm

# package internal imports
from . import config, cxrayutilities, math, utilities
from .exception import InputError, UsageError

# regular expression to check goniometer circle syntax
directionSyntax = re.compile("[xyz][+-]")
circleSyntaxDetector = re.compile("([xyz][+-])|(t[xyz])")
circleSyntaxSample = re.compile("[xyzk][+-]")


class QConvFlags(enum.IntFlag):
    NONE = 0
    HAS_TRANSLATIONS = 1
    HAS_SAMPLEDIS = 4
    VERBOSE = 16


class QConversion:

    """
    Class for the conversion of angular coordinates to momentum space for
    arbitrary goniometer geometries and X-ray energy.  Both angular scans
    (where some goniometer angles change during data acquisition) and energy
    scans (where the energy is varied during acquisition) as well as mixed
    cases can be treated.

    the class is configured with the initialization and does provide three
    distinct routines for conversion to momentum space for

    * point detector:     point(...) or __call__()
    * linear detector:    linear(...)
    * area detector:      area(...)

    linear() and area() can only be used after the init_linear()
    or init_area() routines were called
    """

    _valid_init_kwargs = {'en': 'x-ray energy',
                          'wl': 'x-ray wavelength',
                          'UB': 'orientation/orthonormalization matrix'}
    _valid_call_kwargs = {'delta': 'angle offsets',
                          'wl': 'x-ray wavelength',
                          'en': 'x-ray energy',
                          'UB': 'orientation/orthonormalization matrix',
                          'deg': 'True if angles are in degrees',
                          'sampledis': 'sample displacement vector'}
    _valid_linear_kwargs = {'Nav': 'number of channels for block-average',
                            'roi': 'region of interest'}

    def __init__(self, sampleAxis, detectorAxis, r_i, **kwargs):
        """
        initialize QConversion object.
        This means the rotation axis of the sample and detector circles
        need to be given: starting with the outer most circle.

        Parameters
        ----------
        sampleAxis :    list or tuple
            sample circles, e.g. ['x+', 'z+'] would mean two sample circles
            whereas the outer one turns righthanded around the x axis and the
            inner one turns righthanded around z.

        detectorAxis :  list or tuple
            detector circles e.g. ['x-'] would mean a detector arm with a
            single motor turning lefthanded around the x-axis.

        r_i :           array-like
            vector giving the direction of the primary beam (length is relevant
            only if translations are involved)

        kwargs :        dict, optional
            optional keyword arguments
        wl :            float or str, optional
            wavelength of the x-rays in angstrom
        en :            float or str, optional
            energy of the x-rays in electronvolt
        UB :            array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: identity matrix)
        """

        utilities.check_kwargs(kwargs, self._valid_init_kwargs,
                               self.__class__.__name__)

        # initialize some needed variables
        self._kappa_dir = numpy.array((numpy.nan, numpy.nan, numpy.nan))

        # set/check sample and detector axis geometry
        self._set_sampleAxis(sampleAxis)
        self._set_detectorAxis(detectorAxis)

        # r_i: primary beam direction
        if isinstance(r_i, (list, tuple, numpy.ndarray)):
            self.r_i = numpy.array(r_i, dtype=numpy.double)
            if self.r_i.size != 3:
                print("XU.QConversion: warning invalid primary beam "
                      "direction given -> using [0, 1, 0]")
                self.r_i = numpy.array([0, 1, 0], dtype=numpy.double)
        else:
            raise TypeError("QConversion: invalid type of primary beam "
                            "direction r_i, must be tuple, list or "
                            "numpy.ndarray")

        # kwargs
        self._set_wavelength(kwargs.get("wl", config.WAVELENGTH))
        if "en" in kwargs:
            self._set_energy(kwargs["en"])

        self._set_UB(kwargs.get('UB', numpy.identity(3)))

        self._linear_init = False
        self._area_init = False
        self._area_detrotaxis_set = False

    def _set_energy(self, energy):
        self._en = utilities.energy(energy)
        self._wl = utilities.en2lam(self._en)

    def _set_wavelength(self, wl):
        self._wl = utilities.wavelength(wl)
        self._en = utilities.lam2en(self._wl)

    def _get_energy(self):
        return self._en

    def _get_wavelength(self):
        return self._wl

    def _set_sampleAxis(self, sampleAxis):
        """
        property handler for _sampleAxis
        checks if a syntactically correct list of sample circles is given

        Parameters
        ----------
        sampleAxis :    list or tuple
            sample circles, e.g. ['x+', 'z+']
        """

        if isinstance(sampleAxis, (str, list, tuple)):
            if isinstance(sampleAxis, str):
                sAxis = list([sampleAxis])
            else:
                sAxis = list(sampleAxis)
            for circ in sAxis:
                if not isinstance(circ, str) or len(circ) != 2:
                    raise InputError("QConversion: incorrect sample circle "
                                     "type or syntax (%s)" % repr(circ))
                if not circleSyntaxSample.search(circ):
                    raise InputError("QConversion: incorrect sample circle "
                                     "syntax (%s)" % circ)
                if circ[0] == 'k':  # determine kappa rotation axis
                    self._kappa_dir = math.getVector(circ)
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.QConversion: kappa_dir: (%5.3f %5.3f %5.3f)"
                              % tuple(self._kappa_dir))

        else:
            raise TypeError("Qconversion error: invalid type for sampleAxis, "
                            "must be str, list, or tuple")
        self._sampleAxis = sAxis
        self._sampleAxis_str = ''
        for circ in self._sampleAxis:
            self._sampleAxis_str += circ

    def _get_sampleAxis(self):
        """
        property handler for _sampleAxis

        Returns
        -------
        list
            sample axes following the syntax /[xyzk][+-]/
        """
        return self._sampleAxis

    def _set_detectorAxis(self, detectorAxis, detrot=False):
        """
        property handler for _detectorAxis_
        checks if a syntactically correct list of detector circles is given

        Parameters
        ----------
        detectorAxis :     list or tuple
            detector circles, e.g. ['x+']
        detrot :           bool, optional
            flag to tell that the detector rotation is going to be added (used
            internally to avoid double adding of detector rotation axis)
        """
        has_translations = False
        if isinstance(detectorAxis, (str, list, tuple)):
            if isinstance(detectorAxis, str):
                dAxis = list([detectorAxis])
            else:
                dAxis = list(detectorAxis)
            for circ in dAxis:
                if not isinstance(circ, str) or len(circ) != 2:
                    raise InputError("QConversion: incorrect detector circle "
                                     "type or syntax (%s)" % repr(circ))
                if not circleSyntaxDetector.search(circ):
                    raise InputError("QConversion: incorrect detector circle "
                                     "syntax (%s)" % circ)
                if circ[0] == 't':
                    has_translations = True
        else:
            raise TypeError("Qconversion error: invalid type for "
                            "detectorAxis, must be str, list or tuple")
        self._detectorAxis = dAxis
        self._detectorAxis_str = ''
        self._has_translations = has_translations
        for circ in self._detectorAxis:
            self._detectorAxis_str += circ
        if detrot:
            self._area_detrotaxis_set = True
        else:
            self._area_init = False

    def _get_detectorAxis(self):
        """
        property handler for _detectorAxis

        Returns
        -------
        list of detector axis following the syntax /[xyz][+-]/
        """
        return self._detectorAxis

    def _get_UB(self):
        return self._UB

    def _set_UB(self, UB):
        """
        set the orientation matrix used in the Qconversion
        needs to be (3, 3) matrix
        """
        tmp = numpy.array(UB)
        if tmp.shape != (3, 3) and tmp.size != 9:
            raise InputError("QConversion: incorrect shape of UB matrix "
                             "(shape: %s)" % str(tmp.shape))
        self._UB = tmp.reshape((3, 3))

    energy = property(_get_energy, _set_energy)
    wavelength = property(_get_wavelength, _set_wavelength)
    sampleAxis = property(_get_sampleAxis, _set_sampleAxis)
    detectorAxis = property(_get_detectorAxis, _set_detectorAxis)
    UB = property(_get_UB, _set_UB)

    def __str__(self):
        pstr = 'QConversion geometry \n'
        pstr += '---------------------------\n'
        pstr += f'sample geometry({len(self._sampleAxis)}): ' + \
            self._sampleAxis_str + '\n'
        if self._sampleAxis_str.find('k') != -1:
            pstr += ('kappa rotation axis (%5.3f %5.3f %5.3f)\n'
                     % tuple(self._kappa_dir))
        pstr += f'detector geometry({len(self._detectorAxis)}): ' + \
            self._detectorAxis_str + '\n'
        pstr += ('primary beam direction: (%5.2f %5.2f %5.2f) \n'
                 % (self.r_i[0], self.r_i[1], self.r_i[2]))

        if self._linear_init:
            pstr += '\n linear detector initialized:\n'
            pstr += 'linear detector mount direction: ' + \
                self._linear_detdir + '\n'
            pstr += ('number of channels/center channel: %d/%d\n'
                     % (self._linear_Nch, self._linear_cch))
            pstr += ('distance to center of rotation/pixel width: '
                     '%10.4g/%10.4g\n'
                     % (self._linear_distance, self._linear_pixwidth))
            chpdeg = 2 * self._linear_distance / \
                self._linear_pixwidth * numpy.tan(numpy.radians(0.5))
            pstr += f'corresponds to channel per degree: {chpdeg:8.2f}\n'
        if self._area_init:
            pstr += '\n area detector initialized:\n'
            pstr += 'area detector mount directions: %s/%s\n' % (
                self._area_detdir1, self._area_detdir2)
            pstr += ('number of channels/center channels: (%d,%d) / (%d,%d)\n'
                     % (self._area_Nch1, self._area_Nch2,
                        self._area_cch1, self._area_cch2))
            pstr += ('distance to center of rotation/pixel width: '
                     '%10.4g/ (%10.4g,%10.4g) \n'
                     % (self._area_distance, self._area_pwidth1,
                        self._area_pwidth2))
            chpdeg1 = 2 * self._area_distance / \
                self._area_pwidth1 * numpy.tan(numpy.radians(0.5))
            chpdeg2 = 2 * self._area_distance / \
                self._area_pwidth2 * numpy.tan(numpy.radians(0.5))
            pstr += 'corresponds to channel per degree: (%8.2f,%8.2f)\n' % (
                chpdeg1, chpdeg2)

        return pstr

    def _checkInput(self, *args):
        """
        helper function to check that the arguments given to the QConversion
        routines have the correct shape. It determines the number of points in
        the input from the longest array/list given and checks that only inputs
        combatible with this length are given

        Parameters
        ----------
        args :      list
            arguments from the QConversion routine (sample and detector angles)

        Returns
        -------
        Npoints :   int
            integer to tell the number of points given
        """
        np = 1
        for a in args:
            # optain size of input
            if isinstance(a, numpy.ndarray):
                anp = a.size
            elif isinstance(a, (list, tuple)):
                anp = len(a)
            elif isinstance(a, numbers.Number):
                anp = 1
            else:
                raise TypeError('QConversion: Input argument #%d has an '
                                'invalid type.' % args.index(a))
            # check if the input field is valid
            if anp > 1 and np == 1:
                np = anp
            elif anp > 1 and np != anp:
                raise InputError('QConversion: Several input-arrays arguments '
                                 'with different shape are an invalid input!')

        return np

    def _reshapeInput(self, npoints, delta, circles, *args, **kwargs):
        """
        helper function to reshape the input of arguments to
        (len(args), npoints) The input arguments must be either scalars or are
        of length npoints.

        Parameters
        ----------
        npoints :   int
            length of the input arrays
        delta :     list or array-like
            value to substract from the input arguments as array with len(args)
        circles :   list
            list of circle description to decide if degree/radians conversion
            is needed
        args :      list
            input arrays and scalars
        kwargs :    dict, optional
            optional keyword argument to tell if values of rotation axis should
            be converted to radiants (name= 'deg', default=True)

        Returns
        -------
        inarr :     ndarray
            array of shape (len(args), npoints) with the input arguments
        retshape :  tuple
            shape of return values
        """

        inarr = numpy.empty((len(args), npoints), dtype=numpy.double)
        retshape = (npoints,)  # default value
        deg2rad = kwargs.get('deg', True)

        for i in range(len(args)):
            arg = args[i]
            if not isinstance(arg,
                              (numbers.Number, list, tuple, numpy.ndarray)):
                raise TypeError("QConversion: invalid type for one of the "
                                "sample coordinates, must be scalar, list or "
                                "array")
            if isinstance(arg, numbers.Number):
                arg = numpy.ones(npoints, dtype=numpy.double) * arg
            elif isinstance(arg, (list, tuple)):
                arg = numpy.array(arg, dtype=numpy.double)
            else:  # determine return value shape
                retshape = arg.shape
            arg = arg - delta[i]
            if deg2rad and circleSyntaxSample.search(circles[i]):
                inarr[i, :] = numpy.radians(numpy.ravel(arg))
            else:
                inarr[i, :] = numpy.ravel(arg)

        return inarr, retshape

    def _parse_common_kwargs(self, **kwargs):
        """
        parse common keyword arguments to QConversion calls

        Parameters
        ----------
        delta :     list or array-like, optional
            delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of len(*args). used angles are
            than ``*args - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: self.UB)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        en :        float, optional
            x-ray energy in eV (default is converted self._wl). both wavelength
            and energy can also be an array which enables the QConversion for
            energy scans.  Note that the `en` keyword overrules the `wl`
            keyword!
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)
        sampledis : tuple or list or array-like
            sample displacement vector in relative units of the detector
            distance. Applies to parallel beam geometry. (default: (0, 0, 0))
        """
        flags = QConvFlags.NONE
        if self._has_translations:
            flags |= QConvFlags.HAS_TRANSLATIONS

        Ns = len(self.sampleAxis)
        Nd = len(self.detectorAxis)
        if self._area_detrotaxis_set:
            Nd -= 1  # do not consider detector rotation for point detector
        Ncirc = Ns + Nd

        # kwargs
        wl = utilities.wavelength(kwargs.get('wl', self._wl))
        if 'en' in kwargs:
            wl = utilities.lam2en(utilities.energy(kwargs['en']))

        deg = kwargs.get('deg', True)

        delta = numpy.asarray(kwargs.get('delta', numpy.zeros(Ncirc)),
                              dtype=numpy.double)
        if delta.size != Ncirc:
            raise InputError("QConversion: keyword argument delta does "
                             "not have an appropriate shape")

        UB = numpy.asarray(kwargs.get('UB', self.UB))

        sd = numpy.asarray(kwargs.get('sampledis', [0, 0, 0]))
        if 'sampledis' in kwargs:
            flags |= QConvFlags.HAS_SAMPLEDIS

        return Ns, Nd, Ncirc, wl, deg, delta, UB, sd, flags

    def __call__(self, *args, **kwargs):
        """
        wrapper function for point(...)
        """
        return self.point(*args, **kwargs)

    def point(self, *args, **kwargs):
        """
        angular to momentum space conversion for a point detector
        located in direction of self.r_i when detector angles are zero

        Parameters
        ----------
        args :      ndarray, list or Scalars
            sample and detector angles; in total `len(self.sampleAxis) +
            len(detectorAxis)` must be given, always starting with the outer
            most circle. all arguments must have the same shape or length but
            can be mixed with Scalars (i.e. if an angle is always the same it
            can be given only once instead of an array)

             - sAngles :
                sample circle angles, number of arguments must correspond to
                len(self.sampleAxis)
             - dAngles :
                detector circle angles, number of arguments must correspond to
                len(self.detectorAxis)

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list or array-like, optional
            delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of ``len(*args)``. used angles
            are then ``*args - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: self.UB)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        en :        float, optional
            x-ray energy in eV (default is converted self._wl). both wavelength
            and energy can also be an array which enables the QConversion for
            energy scans.  Note that the `en` keyword overrules the `wl`
            keyword!
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)
        sampledis : tuple or list or array-like
            sample displacement vector in relative units of the detector
            distance. Applies to parallel beam geometry. (default: (0, 0, 0))

        Returns
        -------
        ndarray
            reciprocal space positions as numpy.ndarray with shape ``(N , 3)``
            where `N` corresponds to the number of points given in the input
        """

        utilities.check_kwargs(kwargs, self._valid_call_kwargs, 'Ang2Q/point')

        Ns, Nd, Ncirc, wl, deg, delta, UB, sd, flags = \
            self._parse_common_kwargs(**kwargs)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments "
                             "given, number of arguments should be %d"
                             % (len(args), Ncirc))

        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles, retshape = self._reshapeInput(Npoints, delta[:Ns],
                                               self.sampleAxis, *args[:Ns],
                                               deg=deg)
        dAngles = self._reshapeInput(Npoints, delta[Ns:],
                                     self.detectorAxis, *args[Ns:],
                                     deg=deg)[0]
        wl = numpy.ravel(self._reshapeInput(Npoints, (0, ), 'a',
                                            wl, deg=False)[0])

        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        sAxis = self._sampleAxis_str
        dAxis = self._detectorAxis_str

        if self._area_detrotaxis_set:
            # do not consider detector rotation for point detector
            dAxis = self._detectorAxis_str[:-2]
        else:
            dAxis = self._detectorAxis_str

        if config.VERBOSITY >= config.DEBUG:
            print("XU.QConversion: Ns, Nd: %d %d" % (Ns, Nd))
            print(f"XU.QConversion: sAngles / dAngles {str(sAngles)} / "
                  f"{str(dAngles)}")

        qpos = cxrayutilities.ang2q_conversion(
            sAngles, dAngles, self.r_i, sAxis, dAxis,
            self._kappa_dir, UB, sd, wl, config.NTHREADS, flags)

        if Npoints == 1:
            return (qpos[0, 0], qpos[0, 1], qpos[0, 2])
        return numpy.reshape(qpos[:, 0], retshape), \
            numpy.reshape(qpos[:, 1], retshape), \
            numpy.reshape(qpos[:, 2], retshape)

    def init_linear(self, detectorDir, cch, Nchannel, distance=None,
                    pixelwidth=None, chpdeg=None, tilt=0, **kwargs):
        """
        initialization routine for linear detectors
        detector direction as well as distance and pixel size or
        channels per degree must be given.

        Parameters
        ----------
        detectorDir :   str
            direction of the detector (along the pixel array); e.g. 'z+'
        cch :           float
            center channel, in direction of self.r_i at zero detectorAngles
        Nchannel :      int
            total number of detector channels
        distance :      float, optional
            distance of center channel from center of rotation
        pixelwidth :    float, optional
            width of one pixel (same unit as distance)
        chpdeg :        float, optional
            channels per degree (only absolute value is relevant) sign
            determined through detectorDir
        tilt :          float, optional
            tilt of the detector axis from the detectorDir (in degree)
        kwargs:         dict, optional
            optional keyword arguments
        Nav :           int, optional
            number of channels to average to reduce data size (default: 1)
        roi :           tuple or list
            region of interest for the detector pixels; e.g. [100, 900]

            Note:
                Either distance and pixelwidth or chpdeg must be given !!

            Note:
                the channel numbers run from 0 .. Nchannel-1

        """

        utilities.check_kwargs(kwargs, self._valid_linear_kwargs,
                               'init_linear')

        # detectorDir
        if not isinstance(detectorDir, str) or len(detectorDir) != 2:
            raise InputError("QConversion: incorrect detector direction type "
                             "or syntax (%s)" % repr(detectorDir))
        if not directionSyntax.search(detectorDir):
            raise InputError("QConversion: incorrect detector direction "
                             "syntax (%s)" % detectorDir)
        self._linear_detdir = detectorDir

        self._linear_Nch = int(Nchannel)
        self._linear_cch = float(cch)
        self._linear_tilt = numpy.radians(tilt)

        if distance is not None and pixelwidth is not None:
            self._linear_distance = float(distance)
            self._linear_pixwidth = float(pixelwidth)
        elif chpdeg is not None:
            self._linear_distance = 1.0
            self._linear_pixwidth = 2 * self._linear_distance / \
                numpy.abs(float(chpdeg)) * numpy.tan(numpy.radians(0.5))
        else:
            # not all needed values were given
            raise InputError("QConversion: not all mandatory arguments were "
                             "given -> read API doc, need distance and "
                             "pixelwidth or chpdeg")

        # kwargs
        self._linear_roi = kwargs.get('roi', [0, self._linear_Nch])
        self._linear_nav = kwargs.get('Nav', 1)

        # rescale r_i
        self.r_i = math.VecUnit(self.r_i) * self._linear_distance

        self._linear_init = True

    def _get_detparam_linear(self, oroi, nav):
        """
        initialize linear detector geometry for C subroutines. This function
        considers the Nav and roi options.
        """
        cch = self._linear_cch / float(nav)
        pwidth = self._linear_pixwidth * nav
        roi = numpy.array(oroi)
        roi[0] = numpy.floor(oroi[0] / float(nav))
        roi[1] = numpy.ceil((oroi[1] - oroi[0]) / float(nav)) + roi[0]
        roi = roi.astype(numpy.int32)
        return cch, pwidth, roi

    def linear(self, *args, **kwargs):
        """
        angular to momentum space conversion for a linear detector
        the cch of the detector must be in direction of self.r_i when
        detector angles are zero

        the detector geometry must be initialized by the init_linear(...)
        routine

        Parameters
        ----------
        args :      ndarray, list or Scalars
            sample and detector angles; in total `len(self.sampleAxis) +
            len(detectorAxis)` must be given, always starting with the outer
            most circle. all arguments must have the same shape or length but
            can be mixed with Scalars (i.e. if an angle is always the same it
            can be given only once instead of an array)

             - sAngles :
                sample circle angles, number of arguments must correspond to
                len(self.sampleAxis)
             - dAngles :
                detector circle angles, number of arguments must correspond to
                len(self.detectorAxis)

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list or array-like, optional
            delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of ``len(*args)``. used angles
            are then ``*args - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: self.UB)
        Nav :       int, optional
            number of channels to average to reduce data size (default:
            self._linear_nav)
        roi :       list or tuple, optional
            region of interest for the detector pixels; e.g.  [100, 900]
            (default: self._linear_roi)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        en :        float, optional
            x-ray energy in eV (default is converted self._wl). both wavelength
            and energy can also be an array which enables the QConversion for
            energy scans.  Note that the `en` keyword overrules the `wl`
            keyword!
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)
        sampledis : tuple or list or array-like
            sample displacement vector in relative units of the detector
            distance. Applies to parallel beam geometry. (default: (0, 0, 0))

        Returns
        -------
        reciprocal space position of all detector pixels in a numpy.ndarray of
        shape ( (*)*(self._linear_roi[1]-self._linear_roi[0]+1) , 3 )
        """

        if not self._linear_init:
            raise UsageError("QConversion: linear detector not initialized -> "
                             "call Ang2Q.init_linear(...)")

        valid_kwargs = copy.copy(self._valid_call_kwargs)
        valid_kwargs.update(self._valid_linear_kwargs)
        utilities.check_kwargs(kwargs, valid_kwargs, 'Ang2Q/linear')

        Ns, Nd, Ncirc, wl, deg, delta, UB, sd, flags = \
            self._parse_common_kwargs(**kwargs)

        # extra keyword arguments
        nav = kwargs.get('Nav', self._linear_nav)
        oroi = kwargs.get('roi', self._linear_roi)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments "
                             "given, number of arguments should be %d"
                             % (len(args), Ncirc))

        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles, retshape = self._reshapeInput(Npoints, delta[:Ns],
                                               self.sampleAxis, *args[:Ns],
                                               deg=deg)
        dAngles = self._reshapeInput(Npoints, delta[Ns:],
                                     self.detectorAxis, *args[Ns:],
                                     deg=deg)[0]
        wl = numpy.ravel(self._reshapeInput(Npoints, (0, ), 'a',
                                            wl, deg=False)[0])

        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        cch, pwidth, roi = self._get_detparam_linear(oroi, nav)
        sAxis = self._sampleAxis_str
        dAxis = self._detectorAxis_str

        qpos = cxrayutilities.ang2q_conversion_linear(
            sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir,
            cch, pwidth, roi, self._linear_detdir, self._linear_tilt,
            UB, sd, wl, config.NTHREADS, flags)

        # reshape output
        if Npoints == 1:
            qpos.shape = (Npoints * (roi[1] - roi[0]), 3)
            return qpos[:, 0], qpos[:, 1], qpos[:, 2]
        qpos.shape = (Npoints, (roi[1] - roi[0]), 3)
        return qpos[:, :, 0], qpos[:, :, 1], qpos[:, :, 2]

    def init_area(self, detectorDir1, detectorDir2, cch1, cch2, Nch1, Nch2,
                  distance=None, pwidth1=None, pwidth2=None, chpdeg1=None,
                  chpdeg2=None, detrot=0, tiltazimuth=0, tilt=0, **kwargs):
        """
        initialization routine for area detectors
        detector direction as well as distance and pixel size or
        channels per degree must be given. Two separate pixel sizes and
        channels per degree for the two orthogonal directions can be given

        Parameters
        ----------
        detectorDir1 :  str
            direction of the detector (along the pixel direction 1); e.g. 'z+'
            means higher pixel numbers at larger z positions
        detectorDir2 :  str
            direction of the detector (along the pixel direction 2); e.g. 'x+'
        cch1, cch2 :    float
            center pixel, in direction of self.r_i at zero detectorAngles
        Nch1, Nch2 :    int
            number of detector pixels along direction 1, 2
        distance :      float, optional
            distance of center pixel from center of rotation
        pwidth1, pwidth2 : float, optional
            width of one pixel (same unit as distance)
        chpdeg1, chpdeg2 : float, optional
            channels per degree (only absolute value is relevant) sign
            determined through `detectorDir1, detectorDir2`
        detrot :        float, optional
            angle of the detector rotation around primary beam direction (used
            to correct misalignments)
        tiltazimuth :   float, optional
            direction of the tilt vector in the detector plane (in degree)
        tilt :          float, optional
            tilt of the detector plane around an axis normal to the direction
            given by the tiltazimuth

        kwargs :        dict, optional
            optional keyword arguments
        Nav :           tuple or list, optional
            number of channels to average to reduce data size (default: [1, 1])
        roi :           tuple or list, optional
            region of interest for the detector pixels; e.g.
            [100, 900, 200, 800]

            Note:
                Either distance and pwidth1, pwidth2 or chpdeg1, chpdeg2 must
                be given !!

            Note:
                the channel numbers run from 0 .. NchX-1
        """

        utilities.check_kwargs(kwargs, self._valid_linear_kwargs, 'init_area')

        # detectorDir
        if not isinstance(detectorDir1, str) or len(detectorDir1) != 2:
            raise InputError("QConversion: incorrect detector direction1 type "
                             "or syntax (%s)" % repr(detectorDir1))
        if not directionSyntax.search(detectorDir1):
            raise InputError("QConversion: incorrect detector direction1 "
                             "syntax (%s)" % detectorDir1)
        self._area_detdir1 = detectorDir1
        if not isinstance(detectorDir2, str) or len(detectorDir2) != 2:
            raise InputError("QConversion: incorrect detector direction2 type "
                             "or syntax (%s)" % repr(detectorDir2))
        if not directionSyntax.search(detectorDir2):
            raise InputError("QConversion: incorrect detector direction2 "
                             "syntax (%s)" % detectorDir2)
        self._area_detdir2 = detectorDir2

        # other none keyword arguments
        self._area_Nch1 = int(Nch1)
        self._area_Nch2 = int(Nch2)
        self._area_cch1 = float(cch1)
        self._area_cch2 = float(cch2)

        # if detector rotation is present add new motor to consider it in
        # conversion
        self._area_detrot = numpy.radians(detrot)
        if self._area_detrot != 0.:
            if self._area_detrotaxis_set:
                self._set_detectorAxis(
                    self._get_detectorAxis()[:-1] + [math.getSyntax(self.r_i)],
                    detrot=True)
            else:
                self._set_detectorAxis(
                    self._get_detectorAxis() + [math.getSyntax(self.r_i)],
                    detrot=True)

        self._area_tiltazimuth = numpy.radians(tiltazimuth)
        self._area_tilt = numpy.radians(tilt)

        # mandatory keyword arguments
        if (distance is not None and pwidth1 is not None and
                pwidth2 is not None):
            self._area_distance = float(distance)
            self._area_pwidth1 = float(pwidth1)
            self._area_pwidth2 = float(pwidth2)
        elif chpdeg1 is not None and chpdeg2 is not None:
            self._area_distance = 1.0
            self._area_pwidth1 = 2 * self._area_distance / \
                numpy.abs(float(chpdeg1)) * numpy.tan(numpy.radians(0.5))
            self._area_pwidth2 = 2 * self._area_distance / \
                numpy.abs(float(chpdeg2)) * numpy.tan(numpy.radians(0.5))
        else:
            # not all needed values were given
            raise InputError("Qconversion error: not all mandatory arguments "
                             "were given -> read API doc")

        # kwargs
        self._area_roi = kwargs.get('roi', [0, self._area_Nch1,
                                            0, self._area_Nch2])
        self._area_nav = kwargs.get('Nav', [1, 1])

        # rescale r_i
        self.r_i = math.VecUnit(self.r_i) * self._area_distance

        self._area_init = True

    def _get_detparam_area(self, oroi, nav):
        """
        initialize CCD geomtry for C subroutines. This function considers the
        Nav and roi options.
        """
        cch1 = self._area_cch1 / float(nav[0])
        cch2 = self._area_cch2 / float(nav[1])
        pwidth1 = self._area_pwidth1 * nav[0]
        pwidth2 = self._area_pwidth2 * nav[1]
        roi = numpy.array(oroi)
        roi[0] = numpy.floor(oroi[0] / float(nav[0]))
        roi[1] = numpy.ceil((oroi[1] - oroi[0]) / float(nav[0])) + roi[0]
        roi[2] = numpy.floor(oroi[2] / float(nav[1]))
        roi[3] = numpy.ceil((oroi[3] - oroi[2]) / float(nav[1])) + roi[2]
        roi = roi.astype(numpy.int32)
        return cch1, cch2, pwidth1, pwidth2, roi

    def area(self, *args, **kwargs):
        """
        angular to momentum space conversion for a area detector
        the center pixel defined by the init_area routine must be
        in direction of self.r_i when detector angles are zero

        the detector geometry must be initialized by the init_area(...) routine

        Parameters
        ----------
        args :      ndarray, list or Scalars
            sample and detector angles; in total `len(self.sampleAxis) +
            len(detectorAxis)` must be given, always starting with the outer
            most circle. all arguments must have the same shape or length but
            can be mixed with Scalars (i.e. if an angle is always the same it
            can be given only once instead of an array)

             - sAngles :
                sample circle angles, number of arguments must correspond to
                len(self.sampleAxis)
             - dAngles :
                detector circle angles, number of arguments must correspond to
                len(self.detectorAxis)

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list or array-like, optional
            delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of ``len(*args)``. used angles
            are then ``*args - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: self.UB)
        Nav :       tuple or list, optional
            number of channels to average to reduce data size e.g.  [2, 2]
            (default: self._area_nav)
        roi :       list or tuple, optional
            region of interest for the detector pixels; e.g.
            [100, 900, 200, 800] (default: self._area_roi)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        en :        float, optional
            x-ray energy in eV (default is converted self._wl). both wavelength
            and energy can also be an array which enables the QConversion for
            energy scans.  Note that the `en` keyword overrules the `wl`
            keyword!
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)
        sampledis : tuple or list or array-like
            sample displacement vector in relative units of the detector
            distance. Applies to parallel beam geometry. (default: (0, 0, 0))


        Returns
        -------
        reciprocal space position of all detector pixels in a numpy.ndarray of
        shape ((*)*(self._area_roi[1] - self._area_roi[0]+1) *
        (self._area_roi[3] - self._area_roi[2] + 1) , 3) were detectorDir1 is
        the fastest varing
        """

        if not self._area_init:
            raise UsageError("QConversion: area detector not initialized -> "
                             "call Ang2Q.init_area(...)")

        valid_kwargs = copy.copy(self._valid_call_kwargs)
        valid_kwargs.update(self._valid_linear_kwargs)
        utilities.check_kwargs(kwargs, valid_kwargs, 'Ang2Q/area')

        Ns, Nd, Ncirc, wl, deg, delta, UB, sd, flags = \
            self._parse_common_kwargs(**kwargs)

        # extra keyword arguments
        nav = kwargs.get('Nav', self._area_nav)
        oroi = kwargs.get('roi', self._area_roi)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments "
                             "given, number of arguments should be %d"
                             % (len(args), Ncirc))

        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles, retshape = self._reshapeInput(Npoints, delta[:Ns],
                                               self.sampleAxis, *args[:Ns],
                                               deg=deg)
        wl = numpy.ravel(self._reshapeInput(Npoints, (0, ), 'a',
                                            wl, deg=False)[0])

        if self._area_detrotaxis_set:
            Nd = Nd + 1
            if deg:
                a = args[Ns:] + (numpy.degrees(self._area_detrot),)
            else:
                a = args[Ns:] + (self._area_detrot,)
            dAngles = self._reshapeInput(
                Npoints, numpy.append(delta[Ns:], 0),
                self.detectorAxis, *a, deg=deg)[0]
        else:
            dAngles = self._reshapeInput(Npoints, delta[Ns:],
                                         self.detectorAxis, *args[Ns:],
                                         deg=deg)[0]

        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        cch1, cch2, pwidth1, pwidth2, roi = self._get_detparam_area(oroi, nav)

        if config.VERBOSITY >= config.DEBUG:
            print("QConversion.area: roi, number of points per frame: %s, %d"
                  % (str(roi), (roi[1] - roi[0]) * (roi[3] - roi[2])))
            print(f"QConversion.area: cch1, cch2: {cch1:5.2f} {cch2:5.2f}")

        sAxis = self._sampleAxis_str
        dAxis = self._detectorAxis_str

        qpos = cxrayutilities.ang2q_conversion_area(
            sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir,
            cch1, cch2, pwidth1, pwidth2, roi, self._area_detdir1,
            self._area_detdir2, self._area_tiltazimuth, self._area_tilt,
            UB, sd, wl, config.NTHREADS, flags)

        # reshape output
        if Npoints == 1:
            qpos.shape = ((roi[1] - roi[0]), (roi[3] - roi[2]), 3)
            return qpos[:, :, 0], qpos[:, :, 1], qpos[:, :, 2]
        qpos.shape = (Npoints, (roi[1] - roi[0]), (roi[3] - roi[2]), 3)
        return qpos[:, :, :, 0], qpos[:, :, :, 1], qpos[:, :, :, 2]

    def transformSample2Lab(self, vector, *args):
        """
        transforms a vector from the sample coordinate frame to the laboratory
        coordinate system by applying the sample rotations from inner to outer
        circle.

        Parameters
        ----------
        vector :    sequence, list or numpy array
            vector to transform
        args :      list
            goniometer angles (sample angles or full goniometer angles can be
            given. If more angles than the sample circles are given they will
            be ignored)

        Returns
        -------
        ndarray
            rotated vector as numpy.array
        """
        rotvec = vector
        for i in range(len(self.sampleAxis)-1, -1, -1):
            a = args[i]
            axis = self.sampleAxis[i]
            rota = math.getVector(axis)
            rotvec = math.rotarb(rotvec, rota, a)
        return rotvec

    def getDetectorPos(self, *args, **kwargs):
        """
        obtains the detector position vector by applying the detector arm
        rotations.

        Parameters
        ----------
        args :      list
            detector angles. Only detector arm angles as described by the
            detectorAxis attribute must be given.
        kwargs :    dict, optional
            optional keyword arguments
        dim :       int, optional
            dimension of the detector for which the position should be
            determined
        roi :       tuple or list, optional
            region of interest for the detector pixels; (default:
            self._area_roi/self._linear_roi)
        Nav :       tuple or list, optional
            number of channels to average to reduce data size; (default:
            self._area_nav/self._linear_nav)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            numpy array of length 3 with vector components of the detector
            direction. The length of the vector is k.
        """

        valid_kwargs = copy.copy(self._valid_linear_kwargs)
        valid_kwargs['dim'] = 'dimensionality of the detector'
        valid_kwargs['deg'] = 'True if angles are in degrees'
        utilities.check_kwargs(kwargs, valid_kwargs, 'get_detector_pos')

        dim = kwargs.get('dim', 0)

        if dim == 1 and not self._linear_init:
            raise UsageError("QConversion: linear detector not initialized -> "
                             "call Ang2Q.init_linear(...)")
        if dim == 2 and not self._area_init:
            raise UsageError("QConversion: area detector not initialized -> "
                             "call Ang2Q.init_area(...)")

        Nd = len(self.detectorAxis)
        if self._area_detrotaxis_set:
            Nd = Nd - 1

        # kwargs
        deg = kwargs.get('deg', True)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Nd:
            raise InputError(f"QConversion: wrong amount ({len(args)}) of "
                             "arguments given, number of arguments should be "
                             f"{Nd}")

        # determine the number of points and reshape input arguments
        Npoints = self._checkInput(*args)

        if dim == 2 and self._area_detrotaxis_set:
            Nd = Nd + 1
            if deg:
                a = args + (numpy.degrees(self._area_detrot),)
            else:
                a = args + (self._area_detrot,)
            dAngles, retshape = self._reshapeInput(
                Npoints, numpy.append(numpy.zeros(Nd), 0),
                self.detectorAxis, *a, deg=deg)
        else:
            dAngles, retshape = self._reshapeInput(Npoints, numpy.zeros(Nd),
                                                   self.detectorAxis, *args,
                                                   deg=deg)

        dAngles = dAngles.transpose()

        if dim == 2:
            oroi = kwargs.get('roi', self._area_roi)
            nav = kwargs.get('Nav', self._area_nav)
            cch1, cch2, pwidth1, pwidth2, roi = self._get_detparam_area(oroi,
                                                                        nav)
        elif dim == 1:
            oroi = kwargs.get('roi', self._linear_roi)
            nav = kwargs.get('Nav', self._linear_nav)
            cch, pwidth, roi = self._get_detparam_linear(oroi, nav)

        dAxis = self._detectorAxis_str

        if dim == 2:
            cfunc = cxrayutilities.ang2q_detpos_area
            dpos = cfunc(dAngles, self.r_i, dAxis, cch1, cch2, pwidth1,
                         pwidth2, roi, self._area_detdir1, self._area_detdir2,
                         self._area_tiltazimuth, self._area_tilt,
                         config.NTHREADS)

            # reshape output
            if Npoints == 1:
                dpos.shape = ((roi[1] - roi[0]), (roi[3] - roi[2]), 3)
                return dpos[:, :, 0], dpos[:, :, 1], dpos[:, :, 2]
            dpos.shape = (Npoints, (roi[1] - roi[0]), (roi[3] - roi[2]), 3)
            return dpos[:, :, :, 0], dpos[:, :, :, 1], dpos[:, :, :, 2]

        if dim == 1:
            cfunc = cxrayutilities.ang2q_detpos_linear
            dpos = cfunc(dAngles, self.r_i, dAxis, cch, pwidth, roi,
                         self._linear_detdir, self._linear_tilt,
                         config.NTHREADS)

            # reshape output
            if Npoints == 1:
                dpos.shape = (Npoints * (roi[1] - roi[0]), 3)
                return dpos[:, 0], dpos[:, 1], dpos[:, 2]
            dpos.shape = (Npoints, (roi[1] - roi[0]), 3)
            return dpos[:, :, 0], dpos[:, :, 1], dpos[:, :, 2]

        cfunc = cxrayutilities.ang2q_detpos
        dpos = cfunc(dAngles, self.r_i, dAxis, config.NTHREADS)

        if Npoints == 1:
            return (dpos[0, 0], dpos[0, 1], dpos[0, 2])
        return numpy.reshape(dpos[:, 0], retshape), \
            numpy.reshape(dpos[:, 1], retshape), \
            numpy.reshape(dpos[:, 2], retshape)

    def getDetectorDistance(self, *args, **kwargs):
        """
        obtains the detector distance by applying the detector arm movements.
        This is especially interesting for the case of 1 or 2D detectors to
        perform certain geometric corrections.

        Parameters
        ----------
        args :      list
            detector angles. Only detector arm angles as described by the
            detectorAxis attribute must be given.
        kwargs :    dict, optional
            optional keyword arguments
        dim :       int, optional
            dimension of the detector for which the position should be
            determined
        roi :       tuple or list, optional
            region of interest for the detector pixels; (default:
            self._area_roi/self._linear_roi)
        Nav :       tuple or list, optional
            number of channels to average to reduce data size; (default:
            self._area_nav/self._linear_nav)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            numpy array with the detector distance
        """
        x, y, z = self.getDetectorPos(*args, **kwargs)
        return numpy.sqrt(x**2 + y**2 + z**2)


class Experiment:

    """
    base class for describing experiments
    users should use the derived classes: HXRD, GID, PowderExperiment
    """

    _valid_init_kwargs = {'en': 'x-ray energy',
                          'wl': 'x-ray wavelength',
                          'qconv': 'reciprocal space conversion',
                          'sampleor': 'sample orientation'}

    def __init__(self, ipdir, ndir, **keyargs):
        """
        initialization of an Experiment class needs the sample orientation
        given by the samples surface normal and an second not colinear
        direction specifying the inplane reference direction in the crystal
        coordinate system. The orientation of the surface normal in the lab
        coordinate system can also be given or is automatically determined by
        the goniometer type (see argument sampleor).

        Parameters
        ----------
        ipdir :     list or tuple or array-like
            inplane reference direction (ipdir points into the primary beam
            direction at zero angles)
        ndir :      list or tuple or array-like
            surface normal of your sample. ndir points in a direction
            perpendicular to the primary beam, how it is orientated in real
            space is determined by the parameter sampleor (see below).

        keyargs :   dict, optional
            optional keyword arguments
        qconv :     QConversion, optional
            QConversion object to use for the Ang2Q conversion
        sampleor :  {'det', 'sam', '[xyz][+-]'}, optional
            determines the sample surface orientation with respect to the
            coordinate system in which the goniometer rotations are given. You
            can use the [xyz][+-] syntax to specify the nominal surface
            orientation (when all goniometer angles are zero). In addition two
            special values 'det' and 'sam' are available, which will let the
            code determine the orientation from either the inner most detector
            or sample rotation. 'det' means the surface is in the plane spanned
            by the inner most detector rotation (rotation around primary beam
            is ignored) and perpendicular to the primary beam. 'sam' means the
            surface orientation is along the innermost sample circles rotation
            direction (in this case this should be the azimuth motor to yield
            the expected results).  Default is 'det'.
            Restrictions: the given direction can not be along the primary
            beam.  If one needs that case, let the maintainer know. Currently
            this case is caught and a different axis is automatically used as
            z-axis.
        wl :        float or str
            wavelength of the x-rays in angstrom (default: 1.5406A)
        en :        float or str
            energy of the x-rays in eV (default: 8048eV == 1.5406A ).
            the en keyword overrules the wl keyword

            Note:
                The qconv argument does not change the Q2Ang function's
                behavior. See Q2AngFit function in case you want to calculate
                for arbitrary goniometers with some restrictions.
        """

        utilities.check_kwargs(keyargs, self._valid_init_kwargs,
                               self.__class__.__name__)

        if isinstance(ipdir, (list, tuple, numpy.ndarray)):
            self.idir = math.VecUnit(ipdir)
        else:
            raise TypeError("Inplane direction must be list or numpy array")

        if isinstance(ndir, (list, tuple, numpy.ndarray)):
            self.ndir = math.VecUnit(ndir)
        else:
            raise TypeError("normal direction must be list or numpy array")

        # test the given direction to be not parallel and warn if not
        # perpendicular
        if numpy.isclose(norm(numpy.cross(self.idir, self.ndir)), 0):
            raise InputError("given inplane direction is parallel to normal "
                             "direction, they must be linear independent!")
        if not numpy.isclose(numpy.abs(numpy.dot(self.idir, self.ndir)), 0):
            self.idir = numpy.cross(
                numpy.cross(self.ndir, self.idir),
                self.ndir)
            self.idir = self.idir / norm(self.idir)
            warnings.warn("Experiment: given inplane direction is not "
                          "perpendicular to normal direction\n -> Experiment "
                          "class uses the following direction with the same "
                          "azimuth:\n %s" % (' '.join(map(
                                             str, numpy.round(self.idir, 3)))))

        # initialize Ang2Q conversion
        self._A2QConversion = keyargs.get(
            'qconv', QConversion('x+', 'x+', [0, 1, 0]))
        self.Ang2Q = self._A2QConversion

        self._sampleor = keyargs.get('sampleor', 'det')

        # set the coordinate transform for the azimuth used in the experiment
        self.scatplane = math.VecUnit(numpy.cross(self.idir, self.ndir))
        self._set_transform(self.scatplane, self.idir,
                            self.ndir, self._sampleor)

        # calculate the energy from the wavelength
        self._set_wavelength(keyargs.get('wl', config.WAVELENGTH))
        if "en" in keyargs:
            self._set_energy(keyargs["en"])

    def __str__(self):

        ostr = "scattering plane normal: (%f %f %f)\n" % (self.scatplane[0],
                                                          self.scatplane[1],
                                                          self.scatplane[2])
        ostr += "inplane azimuth: (%f %f %f)\n" % (self.idir[0],
                                                   self.idir[1],
                                                   self.idir[2])
        ostr += "second refercence direction: (%f %f %f)\n" % (self.ndir[0],
                                                               self.ndir[1],
                                                               self.ndir[2])
        ostr += f"energy: {self._en:f} (eV)\n"
        ostr += f"wavelength: {self._wl:f} (angstrom)\n"
        ostr += self._A2QConversion.__str__()

        return ostr

    def _set_transform(self, v1, v2, v3, sampleor='det'):
        """
        set new transformation of the coordinate system to use in the
        experimental class.

        The sampleor variable determines the sample surface orientation with
        respect to the coordinate system in which the goniometer rotations are
        given. You can use the [xyz][+-] syntax to specify the nominal surface
        orientation (when all goniometer angles are zero). In addition two
        special values 'det' and 'sam' are available, which will let the code
        determine the orientation from either the inner most detector or sample
        rotation. 'det' means the surface is in the plane spanned by the inner
        most detector rotation (rotation around primary beam is ignored) and
        perpendicular to the primary beam. 'sam' means the surface orientation
        is along the innermost sample circles rotation direction (in this case
        this should be the azimuth motor to yield the expected results).
        Default is 'det'.

        Restrictions: the given direction can not be along the primary beam.
        If one needs that case, let the maintainer know. Currently this case is
        caught and a different axis is automatically used as z-axis.
        """
        # turn idir to Y and ndir to Z
        self._t1 = math.CoordinateTransform(v1, v2, v3)

        if sampleor == 'det':
            yi = self._A2QConversion.r_i
            idc = self._A2QConversion.detectorAxis[-1]
            xi = math.getVector(idc)
            if numpy.isclose(norm(numpy.cross(xi, yi)), 0):
                # this is the case when a detector rotation around the primary
                # beam direction is installed
                idc = self._A2QConversion.detectorAxis[-2]
                xi = math.getVector(idc)
            zi = math.VecUnit(numpy.cross(xi, yi))
        elif sampleor == 'sam':
            yi = self._A2QConversion.r_i
            isc = self._A2QConversion.sampleAxis[-1]
            zi = numpy.abs(math.getVector(isc))
            if numpy.all(numpy.abs(yi) == numpy.abs(zi)):
                zi = numpy.roll(zi, 1)
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.Experiment: Warning, sample orientation "
                          "convention failed. Using (%.3f %.3f %.3f) "
                          "as internal z-axis" % (zi[0], zi[1], zi[2]))
            xi = math.VecUnit(numpy.cross(yi, zi))
        else:
            yi = self._A2QConversion.r_i
            try:
                zi = math.getVector(sampleor)
            except InputError:
                raise InputError('invalid value of sample orientation, use '
                                 'either [xyz][+-] syntax or det/sam!')
            if numpy.all(numpy.abs(yi) == numpy.abs(zi)):
                zi = numpy.roll(zi, 1)
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.Experiment: Warning, sample orientation "
                          "convention failed. Using (%.3f %.3f %.3f) "
                          "as internal z-axis" % (zi[0], zi[1], zi[2]))
            xi = math.VecUnit(numpy.cross(yi, zi))
        # turn r_i to Y and Z defined by detector rotation plane
        self._t2 = math.CoordinateTransform(xi, yi, zi)

        self._transform = math.Transform(
            numpy.dot(numpy.linalg.inv(self._t2.matrix), self._t1.matrix))

    def _set_energy(self, energy):
        self._en = utilities.energy(energy)
        self._wl = utilities.en2lam(self._en)
        self.k0 = numpy.pi * 2. / self._wl
        self._A2QConversion.wavelength = self._wl

    def _set_wavelength(self, wl):
        self._wl = utilities.wavelength(wl)
        self._en = utilities.lam2en(self._wl)
        self.k0 = numpy.pi * 2. / self._wl
        self._A2QConversion.wavelength = self._wl

    def _get_energy(self):
        return self._en

    def _get_wavelength(self):
        return self._wl

    energy = property(_get_energy, _set_energy)
    wavelength = property(_get_wavelength, _set_wavelength)

    def _set_inplane_direction(self, dir):
        self.idir = math.VecUnit(dir)
        v1 = numpy.cross(self.ndir, self.idir)
        self._set_transform(v1, self.idir, self.ndir, self._sampleor)

    def _get_inplane_direction(self):
        return self.idir

    def _set_normal_direction(self, dir):
        self.ndir = math.VecUnit(dir)
        v1 = numpy.cross(self.ndir, self.idir)
        self._set_transform(v1, self.idir, self.ndir, self._sampleor)

    def _get_normal_direction(self):
        return self.ndir

    def Q2Ang(self, qvec):
        pass

    def Ang2HKL(self, *args, **kwargs):
        """
        angular to (h, k, l) space conversion.
        It will set the UB argument to Ang2Q and pass all other parameters
        unchanged.  See Ang2Q for description of the rest of the arguments.

        Parameters
        ----------
        args :      list
            arguments forwarded to Ang2Q
        kwargs :    dict, optional
            optional keyword arguments
        B :         array-like, optional
            reciprocal space conversion matrix of a Crystal. You can specify
            the matrix B (default identiy matrix) shape needs to be (3, 3)
        mat :       Crystal, optional
            Crystal object to use to obtain a B matrix (e.g. xu.materials.Si)
            can be used as alternative to the B keyword argument B is favored
            in case both are given
        U :         array-like, optional
            orientation matrix U can be given. If none is given the orientation
            defined in the Experiment class is used.
        dettype :   {'point', 'linear', 'area'}, optional
            detector type: decides which routine of Ang2Q to call. default
            'point'
        delta :     ndarray, list or tuple, optional
            giving delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of length 2. used angles are
            than ``(om, tt) - delta``
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        en :        float or str, optional
            x-ray energy in eV (default: converted self._wl)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)
        sampledis : tuple or list or array-like
            sample displacement vector in relative units of the detector
            distance. Applies to parallel beam geometry. (default: (0, 0, 0))

        Returns
        -------
        ndarray
            H K L coordinates as numpy.ndarray with shape `(N , 3)` where `N`
            corresponds to the number of points given in the input (args)
        """

        valid_kwargs = {'B': 'orthonormalization matrix',
                        'U': 'orientation matrix',
                        'mat': 'material object',
                        'dettype': 'string with detector type'}
        valid_kwargs.update(QConversion._valid_call_kwargs)
        del valid_kwargs['UB']
        utilities.check_kwargs(kwargs, valid_kwargs, 'Ang2HKL')

        if "B" in kwargs:
            B = numpy.array(kwargs['B'])
            kwargs.pop("B")
        elif "mat" in kwargs:
            mat = kwargs['mat']
            B = mat.B
            kwargs.pop("mat")
        else:
            B = numpy.identity(3)

        if "U" in kwargs:
            U = numpy.array(kwargs['U'])
            kwargs.pop("U")
        else:
            U = self._transform.matrix

        kwargs['UB'] = numpy.dot(U, B)

        if "dettype" in kwargs:
            typ = kwargs['dettype']
            if typ not in ('point', 'linear', 'area'):
                raise InputError("wrong dettype given: needs to be one of "
                                 "'point', 'linear', 'area'")
            kwargs.pop("dettype")
        else:
            typ = 'point'

        if typ == 'linear':
            return self.Ang2Q.linear(*args, **kwargs)
        if typ == 'area':
            return self.Ang2Q.area(*args, **kwargs)
        return self.Ang2Q(*args, **kwargs)

    def Transform(self, v):
        """
        transforms a vector, matrix or tensor of rank 4 (e.g. elasticity
        tensor) to the coordinate frame of the Experiment class. This is for
        example necessary before any Q2Ang-conversion can be performed.

        Parameters
        ----------
        v :     object to transform, list or numpy array of shape
                    (n,) (n, n), (n, n, n, n) where n is the rank of the
                    transformation matrix

        Returns
        -------
         transformed object of the same shape as v
        """
        return self._transform(v)

    def TiltAngle(self, q, deg=True):
        """
        TiltAngle(q, deg=True):
        Return the angle between a q-space position and the surface normal.

        Parameters
        ----------
        q :          list or numpy array with the reciprocal space position

        optional keyword arguments:
        deg :        True/False whether the return value should be in degree or
                     radians (default: True)
        """

        if isinstance(q, list):
            qt = numpy.array(q, dtype=numpy.double)
        elif isinstance(q, numpy.ndarray):
            qt = q
        else:
            raise TypeError("q-space position must be list or numpy array")

        return math.VecAngle(self.ndir, qt, deg)

    def _prepare_qvec(self, Q):
        """
        check and reshape input to have the same q array for all possible types
        of input
        """
        if len(Q) < 3:
            Q = Q[0]
            if len(Q) < 3:
                raise InputError("need 3 q-space vector components")

        if isinstance(Q, (list, tuple, numpy.ndarray)):
            q = numpy.asarray(Q, dtype=numpy.double)
        else:
            raise TypeError("Q vector must be a list, tuple or numpy array")

        if len(q.shape) != 2:
            q = q.reshape(3, -1)
        return q.T


class HXRD(Experiment):

    """
    class describing high angle x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as helps with analyzing measured data

    the class describes a two circle (omega, twotheta) goniometer to
    help with coplanar x-ray diffraction experiments. Nevertheless 3D data
    can be treated with the use of linear and area detectors.
    see "help(HXRDInstance.Ang2Q)"
    """

    def __init__(self, idir, ndir, geometry='hi_lo', **keyargs):
        """
        initialization routine for the HXRD Experiment class

        Parameters
        ----------
        idir, ndir, keyargs :
            same as for the Experiment base class -> please look at the
            docstring of Experiment.__init__ for more details
        geometry :      {'hi_lo', 'lo_hi', 'real'}, optional
            determines the scattering geometry :

                - 'hi_lo' (default) high incidence-low exit
                - 'lo_hi' low incidence - high exit
                - 'real' general geometry - q-coordinates determine
                  high or low incidence

        """
        if "qconv" not in keyargs:
            keyargs['qconv'] = QConversion('x+', 'x+', [0, 1, 0])

        if geometry in ["hi_lo", "lo_hi", "real"]:
            self.geometry = geometry
        else:
            raise InputError("HXRD: invalid value for the geometry "
                             "argument given")

        Experiment.__init__(self, idir, ndir, **keyargs)

        if config.VERBOSITY >= config.DEBUG:
            print(
                "XU.HXRD.__init__: \nEnergy: %s \nGeometry: %s \n%s---" %
                (self._en, self.geometry, str(
                    self.Ang2Q)))

    # pylint: disable-next=method-hidden
    def Ang2Q(self, om, tt, **kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help HXRD.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
        om, tt :    float or array-like
            sample and detector angles as numpy array, lists or Scalars must be
            given. All arguments must have the same shape or length. However,
            if one angle is always the same its enough to give one scalar
            value.

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list or array-like
            giving delta angles to correct the given ones for misalignment.
            delta must be an numpy array or list of length 2. Used angles are
            than om, tt - delta
        UB :        array-like
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: identity matrix)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            reciprocal space positions as numpy.ndarray with shape `(N , 3)`
            where `N` corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine

    def Q2Ang(self, *Q, **keyargs):
        """
        Convert a reciprocal space vector Q to COPLANAR scattering angles.  The
        keyword argument trans determines whether Q should be transformed to
        the experimental coordinate frame or not. The coplanar scattering
        angles correspond to a goniometer with sample rotations
        ['x+', 'y+', 'z-'] and detector rotation 'x+' and primary beam along y.
        This is a standard four circle diffractometer.

        Note:
            The behavior of this function is unchanged if the goniometer
            definition is changed!

        Parameters
        ----------
        Q :         list, tuple or array-like
            array of shape (3) with q-space vector components or 3
            separate lists with qx, qy, qz

        trans :     bool, optional
            apply coordinate transformation on Q (default True)
        deg :       book, optional
            (default True) determines if the angles are returned in radians or
            degrees
        geometry :  {'hi_lo', 'lo_hi', 'real', 'realTilt'}, optional
            determines the scattering geometry (default: self.geometry):

             - 'hi_lo' high incidence and low exit
             - 'lo_hi' low incidence and high exit
             - 'real' general geometry with angles determined by
               q-coordinates (azimuth); this and upper geometries
               return [omega, 0, phi, twotheta]
             - 'realTilt' general geometry with angles determined by
               q-coordinates (tilt); returns [omega, chi, phi, twotheta]

        refrac :    bool, optional
            determines if refraction is taken into account;
            if True then also a material must be given (default: False)
        mat :       Crystal
            Crystal object; needed to obtain its optical properties for
            refraction correction, otherwise not used
        full_output : bool, optional
            determines if additional output is given to determine scattering
            angles more accurately in case refraction is set to True. default:
            False
        fi, fd :    tuple or list
            if refraction correction is applied one can optionally specify the
            facet through which the beam enters (fi) and exits (fd) fi, fd must
            be the surface normal vectors (not transformed & not necessarily
            normalized). If omitted the normal direction of the experiment is
            used.

        Returns
        -------
        ndarray
            **full_output=False**: a numpy array of shape (4) with four
            scattering angles which are [omega, chi, phi, twotheta];

             - omega :      incidence angle with respect to surface
             - chi :        sample tilt for the case of non-coplanar geometry
             - phi :        sample azimuth with respect to inplane reference
                            direction
             - twotheta :   scattering angle/detector angle

            **full_output=True**: a numpy array of shape (6) with five angles
            which are [omega, chi, phi, twotheta, psi_i, psi_d]

             - psi_i : offset of the incidence beam from the scattering plane
                       due to refraction
             - pdi_d : offset ot the diffracted beam from the scattering plane
                       due to refraction
        """

        valid_kwargs = {'trans': 'flag, perform coordinate transformation',
                        'deg': 'flag, return degrees',
                        'geometry': 'geometry string',
                        'refrac': 'flag',
                        'mat': 'Crystal instance',
                        'fi': 'incidence facet',
                        'fd': 'exit facet',
                        'full_output': 'see docstring for details'}
        utilities.check_kwargs(keyargs, valid_kwargs, 'Q2Ang')

        q = self._prepare_qvec(Q)

        # parse keyword arguments
        geom = keyargs.get('geometry', self.geometry)
        if geom not in ["hi_lo", "lo_hi", "real", "realTilt"]:
            raise InputError("HXRD: invalid value for the geometry argument "
                             "given\n valid entries are: hi_lo, lo_hi, real, "
                             "realTilt")
        trans = keyargs.get('trans', True)
        deg = keyargs.get('deg', True)
        mat = keyargs.get('mat', None)  # material for optical properties

        refrac = keyargs.get('refrac', False)
        if refrac and mat is None:  # check if material is available
            raise InputError("keyword argument 'mat' must be set when "
                             "'refrac' is set to True!")

        foutp = keyargs.get('full_output', False)
        fi = keyargs.get('fi', self.ndir)  # incidence facet
        fi = math.VecUnit(self.Transform(fi))

        fd = keyargs.get('fd', self.ndir)  # exit facet
        fd = math.VecUnit(self.Transform(fd))

        # set parameters for the calculation
        z = self.Transform(self.ndir)  # z
        y = self.Transform(self.idir)  # y
        x = self.Transform(self.scatplane)  # x
        if refrac:
            n = numpy.real(
                mat.idx_refraction(
                    self.energy))  # index of refraction
            k = self.k0 * n
        else:
            k = self.k0

        # start calculation for each given Q-point
        if foutp:
            angle = numpy.zeros((6, q.shape[0]))
        else:
            angle = numpy.zeros((4, q.shape[0]))

        if trans:
            q = self.Transform(q)

        if config.VERBOSITY >= config.DEBUG:
            print(f"XU.HXRD.Q2Ang: q= {repr(q)}")

        qa = math.VecNorm(q)
        tth = 2. * numpy.arcsin(qa / 2. / k)

        # calculation of the sample azimuth phi (scattering plane
        # spanned by qvec[1] and qvec[2] directions)

        chi = -numpy.arctan2(math.VecDot(q, x), math.VecDot(q, z))
        if numpy.any(numpy.isclose(numpy.abs(math.VecDot(q, z)), 0)):
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.HXRD: some position is perpendicular to ndir-"
                      "reference direction (might be inplane or "
                      "unreachable)")

        if geom == 'hi_lo':
            # +: high incidence geometry
            om = tth / 2. + math.VecAngle(q, z)
            phi = -numpy.arctan2(math.VecDot(q, x), math.VecDot(q, y))
        elif geom == 'lo_hi':
            # -: low incidence geometry
            om = tth / 2. - math.VecAngle(q, z)
            phi = -numpy.arctan2(-1 * math.VecDot(q, x),
                                 -1 * math.VecDot(q, y))
        elif geom == 'real':
            phi = -numpy.arctan2(math.VecDot(q, x), math.VecDot(q, y))
            sign = numpy.ones(q.shape[0])
            m = numpy.abs(phi) > numpy.pi / 2.
            phi = -numpy.arctan2(math.VecDot(q, x), math.VecDot(q, y))
            sign[m] = -1
            phi[m] = -numpy.arctan2(-1 * math.VecDot(q[m], x),
                                    -1 * math.VecDot(q[m], y))
            om = tth / 2 + sign * math.VecAngle(q, z)
        elif geom == 'realTilt':
            phi = 0.
            om = tth / 2 + numpy.arctan2(
                math.VecDot(q, y),
                numpy.sqrt(math.VecDot(q, z) ** 2 + math.VecDot(q, x) ** 2))

        # refraction correction at incidence and exit facet
        psi_i = numpy.zeros_like(tth)
        psi_d = numpy.zeros_like(tth)  # if refrac is false and full_output
        if refrac:
            beta = tth - om

            ki = k * (numpy.cos(om)[:, numpy.newaxis] * y[numpy.newaxis, :] -
                      numpy.sin(om)[:, numpy.newaxis] * z[numpy.newaxis, :])
            kd = k * (numpy.cos(beta)[:, numpy.newaxis] * y[numpy.newaxis, :] +
                      numpy.sin(beta)[:, numpy.newaxis] * z[numpy.newaxis, :])

            # refraction at incidence facet
            m = math.VecDot(ki, fi) > 0
            if numpy.any(m):
                print("XU.HXRD: Warning, incidence facet not hit by "
                      "primary beam for all positions! check your input!")
                om[m] = numpy.nan
                tth[m] = numpy.nan
            mnot = numpy.logical_not(m)
            cosbi = numpy.abs(math.VecDot(ki, fi) / math.VecNorm(ki))
            cosb0 = numpy.sqrt(1 - n ** 2 * (1 - cosbi ** 2))

            ki0 = self.k0 * (n * math.VecUnit(ki) -
                             (numpy.sign(math.VecDot(ki, fi)) *
                             (n * cosbi - cosb0))[:, numpy.newaxis] *
                             fi[numpy.newaxis, :])
            om[mnot] = math.VecAngle(ki0, y)
            psi_i[mnot] = numpy.arcsin(math.VecDot(ki0, x) / self.k0)
            if config.VERBOSITY >= config.DEBUG:
                print(f"XU.HXRD.Q2Ang: ki, ki0 = {repr(ki)} {repr(ki0)}")

            # refraction at exit facet
            m = math.VecDot(kd, fd) < 0
            if numpy.any(m):
                print("XU.HXRD: Warning, exit facet not hit by "
                      "diffracted beam! check your input!")
                om[m] = numpy.nan
                tth[m] = numpy.nan

            cosbd = numpy.abs(math.VecDot(kd, fd) / math.VecNorm(kd))
            cosb0 = numpy.sqrt(1 - n ** 2 * (1 - cosbd ** 2))

            kd0 = self.k0 * (n * math.VecUnit(kd) -
                             (numpy.sign(math.VecDot(kd, fd)) *
                             (n * cosbd - cosb0))[:, numpy.newaxis] *
                             fd[numpy.newaxis, :])
            tth[mnot] = math.VecAngle(ki0, kd0)
            psi_d[mnot] = numpy.arcsin(numpy.dot(kd0, x) / self.k0)
            if config.VERBOSITY >= config.DEBUG:
                print(f"XU.HXRD.Q2Ang: kd, kd0 = {repr(kd)} {repr(kd0)}")

        if geom == 'realTilt':
            angle[0, :] = om
            angle[1, :] = chi
            angle[3, :] = tth
        else:
            angle[0, :] = om
            angle[2, :] = phi
            angle[3, :] = tth
        if foutp:
            angle[4, :] = psi_i
            angle[5, :] = psi_d

        if q.shape[0] == 1:
            angle = angle.flatten()
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.HXRD.Q2Ang: om, chi, phi, tth,[psi_i, psi_d] = %s"
                      % repr(angle))

        if deg:
            return numpy.degrees(angle)
        return angle


class FourC(HXRD):

    """
    class describing high angle x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as helps with analyzing measured data

    the class describes a four circle (omega, chi, phi, twotheta) goniometer to
    help with coplanar x-ray diffraction experiments. Nevertheless 3D data can
    be treated with the use of linear and area detectors.  see
    "help(FourCInstance.Ang2Q)"
    """

    def __init__(self, idir, ndir, **keyargs):
        """
        initialization routine for the FourC Experiment class

        Parameters
        ----------
        idir, ndir, keyargs :
            same as for the Experiment base class -> please look at the
            docstring of Experiment.__init__ for more details
        geometry :      {'hi_lo', 'lo_hi', 'real'}, optional
            determines the scattering geometry :

                - 'hi_lo' (default) high incidence-low exit
                - 'lo_hi' low incidence - high exit
                - 'real' general geometry - q-coordinates determine
                  high or low incidence

        """
        if "qconv" not in keyargs:
            # 3S+1D goniometer (standard four-circle goniometer,
            # omega, chi, phi, theta)
            keyargs['qconv'] = QConversion(['x+', 'y+', 'z-'],
                                           'x+', [0, 1, 0])

        HXRD.__init__(self, idir, ndir, **keyargs)


class NonCOP(Experiment):

    """
    class describing high angle x-ray diffraction experiments.  The class helps
    with calculating the angles of Bragg reflections as well as helps with
    analyzing measured data for NON-COPLANAR measurements, where the tilt is
    used to align asymmetric peaks, like in the case of a polefigure
    measurement.

    The class describes a four circle (omega, chi, phi, twotheta) goniometer to
    help with x-ray diffraction experiments. Linear and area detectors can be
    treated as described in "help(NonCOPInstance.Ang2Q)"
    """

    def __init__(self, idir, ndir, **keyargs):
        """
        initialization routine for the NonCOP Experiment class

        Parameters
        ----------
        idir, ndir, keyargs :
            same as for the Experiment base class
        """
        if "qconv" not in keyargs:
            # 3S+1D goniometer (standard four-circle goniometer,
            # omega, chi, phi, theta)
            keyargs['qconv'] = QConversion(['x+', 'y+', 'z-'],
                                           'x+', [0, 1, 0])

        Experiment.__init__(self, idir, ndir, **keyargs)

    # pylint: disable-next=method-hidden
    def Ang2Q(self, om, chi, phi, tt, **kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help NonCOP.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
        om, chi, phi, tt : float or array-like
            sample and detector angles as numpy array, lists or Scalars must be
            given. All arguments must have the same shape or length. However,
            if one angle is always the same its enough to give one scalar
            value.

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list, tuple or array-like, optional
            giving delta angles to correct the given ones for misalignment
            delta must be an numpy array or list of length 4. Used angles are
            than om, chi, phi, tt - delta
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: identity matrix)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            reciprocal space positions as numpy.ndarray with shape `(N , 3)`
            where `N` corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine

    def Q2Ang(self, *Q, **keyargs):
        """
        Convert a reciprocal space vector Q to NON-COPLANAR scattering angles.
        The keyword argument trans determines whether Q should be transformed
        to the experimental coordinate frame or not.

        Note:
            The behavior of this function is unchanged if the goniometer
            definition is changed!

        Parameters
        ----------
        Q :         list, tuple or array-like
            array of shape (3) with q-space vector components or 3
            separate lists with qx, qy, qz

        trans :     bool, optional
            apply coordinate transformation on Q (default True)
        deg :       book, optional
            (default True) determines if the angles are returned in radians or
            degrees

        Returns
        -------
        ndarray
            a numpy array of shape (4) with four scattering
            angles which are [omega, chi, phi, twotheta];

             - omega :      incidence angle with respect to surface
             - chi :        sample tilt for the case of non-coplanar geometry
             - phi :        sample azimuth with respect to inplane reference
                            direction
             - twotheta :   scattering angle/detector angle
        """

        valid_kwargs = {'trans': 'coordinate transformation flag',
                        'deg': 'degree-flag'}
        utilities.check_kwargs(keyargs, valid_kwargs, 'Q2Ang')

        q = self._prepare_qvec(Q)

        trans = keyargs.get('trans', True)
        deg = keyargs.get('deg', True)

        angle = numpy.zeros((4, q.shape[0]))
        # set parameters for the calculation
        z = self.Transform(self.ndir)  # z
        y = self.Transform(self.idir)  # y
        x = self.Transform(self.scatplane)  # x

        if trans:
            q = self.Transform(q)

        if config.VERBOSITY >= config.DEBUG:
            print(f"XU.NonCOP.Q2Ang: q= {repr(q)}")

        qa = math.VecNorm(q)
        tth = 2. * numpy.arcsin(qa / 2. / self.k0)
        om = tth / 2.

        # calculation of the sample azimuth
        # the sign depends on the phi movement direction
        phi = -1 * numpy.arctan2(
            math.VecDot(q, x),
            math.VecDot(q, y)) - numpy.pi / 2.

        chi = (math.VecAngle(q, z))

        angle[0, :] = om
        angle[1, :] = chi
        angle[2, :] = phi
        angle[3, :] = tth

        if q.shape[0] == 1:
            angle = angle.flatten()
            if config.VERBOSITY >= config.INFO_ALL:
                print(f"XU.HXRD.Q2Ang: [om, chi, phi, tth] = {repr(angle)}")

        if deg:
            return numpy.degrees(angle)
        return angle


class GID(Experiment):

    """
    class describing grazing incidence x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as it helps with analyzing measured data

    the class describes a four circle (alpha_i, azimuth, twotheta, beta)
    goniometer to help with GID experiments at the ROTATING ANODE.
    3D data can be treated with the use of linear and area detectors.
    see "help(GIDInstance.Ang2Q)"

    Using this class the default sample surface orientation is determined by
    the inner most sample rotation (which is usually the azimuth motor).
    """

    def __init__(self, idir, ndir, **keyargs):
        """
        initialization routine for the GID Experiment class

         - ``idir`` defines the inplane reference direction (idir points into
           the PB direction at zero angles)
         - ``ndir`` defines the surface normal of your sample (ndir points
           along the innermost sample rotation axis)

        Parameters
        ----------
        idir, ndir, keyargs :
            same as for the Experiment base class
        """
        if 'sampleor' not in keyargs:
            keyargs['sampleor'] = 'sam'

        if "qconv" not in keyargs:
            # 2S+2D goniometer
            keyargs['qconv'] = QConversion(['z-', 'x+'], ['x+', 'z-'],
                                           [0, 1, 0])

        Experiment.__init__(self, idir, ndir, **keyargs)

    def Q2Ang(self, qvec, trans=True, deg=True, **kwargs):
        """
        calculate the GID angles needed in the experiment
        the inplane reference direction defines the direction were
        the reference direction is parallel to the primary beam
        (i.e. lattice planes perpendicular to the beam)

        Note:
            The behavior of this function is unchanged if the goniometer
            definition is changed!

        Parameters
        ----------
        qvec :         list, tuple or array-like
            array of shape (3) with q-space vector components or 3
            separate lists with qx, qy, qz

        trans :     bool, optional
            apply coordinate transformation on Q (default True)
        deg :       book, optional
            (default True) determines if the angles are returned in radians or
            degrees

        Returns
        -------
        ndarray
            a numpy array of shape (4) with four GID scattering
            angles which are [alpha_i, azimuth, twotheta, beta];

             - alpha_i :    incidence angle to surface (at the moment always 0)
             - azimuth :    sample rotation with respect to the inplane
                            reference direction
             - twotheta :   scattering angle
             - beta :       exit angle from surface (at the moment always 0)

        """

        valid_kwargs = {'trans': 'coordinate transformation flag',
                        'deg': 'degree-flag'}
        utilities.check_kwargs(kwargs, valid_kwargs, 'Q2Ang')

        if isinstance(qvec, list):
            q = numpy.array(qvec, dtype=numpy.double)
        elif isinstance(qvec, numpy.ndarray):
            q = qvec
        else:
            raise TypeError("Q vector must be a list or numpy array")

        if trans:
            q = self.Transform(q)

        if config.VERBOSITY >= config.INFO_ALL:
            print(f"XU.GID.Q2Ang: q = {repr(q)}")

        # set parameters for the calculation
        z = self.Transform(self.ndir)  # z
        y = self.Transform(self.idir)  # y
        x = self.Transform(self.scatplane)  # x

        # check if reflection is inplane
        if numpy.abs(math.VecDot(q, z)) >= 0.001:
            raise InputError(
                f"Reflection not reachable in GID geometry (Q: {str(q)})")

        # calculate angle to inplane reference direction
        aref = numpy.arctan2(math.VecDot(q, x), math.VecDot(q, y))

        # calculate scattering angle
        qa = math.VecNorm(q)
        tth = 2. * numpy.arcsin(qa / 2. / self.k0)
        azimuth = numpy.pi / 2 + aref + tth / 2.

        if deg:
            ang = [0, numpy.degrees(azimuth), numpy.degrees(tth), 0]
        else:
            ang = [0, azimuth, tth, 0]

        if config.VERBOSITY >= config.INFO_ALL:
            print(f"XU.GID.Q2Ang: [ai, azimuth, tth, beta] = {str(ang)}\n"
                  f"difference to inplane reference which is {aref:5.2f}")

        return ang

    # pylint: disable-next=method-hidden
    def Ang2Q(self, ai, phi, tt, beta, **kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help GID.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
        ai, phi, tt, beta : float or array-like
            sample and detector angles as numpy array, lists or Scalars must be
            given. All arguments must have the same shape or length. However,
            if one angle is always the same its enough to give one scalar
            value.

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list, tuple or array-like, optional
            giving delta angles to correct the given ones for misalignment
            delta must be an numpy array or list of length 4. Used angles are
            then ``ai, phi, tt, beta - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: identity matrix)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            reciprocal space positions as numpy.ndarray with shape `(N , 3)`
            where `N` corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine


class GISAXS(Experiment):

    """
    class describing grazing incidence x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as it helps with analyzing measured data

    the class describes a three circle (alpha_i, twotheta, beta)
    goniometer to help with GISAXS experiments at the ROTATING ANODE.
    3D data can be treated with the use of linear and area detectors.
    see help self.Ang2Q
    """

    def __init__(self, idir, ndir, **keyargs):
        """
        initialization routine for the GISAXS Experiment class

        ``idir`` defines the inplane reference direction (idir points into the
        PB direction at zero angles)

        Parameters
        ----------
        idir, ndir, keyargs :
            same as for the Experiment base class
        """
        if "qconv" not in keyargs:
            # 1S+2D goniometer
            keyargs['qconv'] = QConversion(['x+'], ['x+', 'z-'],
                                           [0, 1, 0])

        Experiment.__init__(self, idir, ndir, **keyargs)

    def Q2Ang(self, Q, trans=True, deg=True, **kwargs):
        pass

    # pylint: disable-next=method-hidden
    def Ang2Q(self, ai, tt, beta, **kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help GISAXS.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
        ai, tt, beta :  float or array-like
            sample and detector angles as numpy array, lists or Scalars must be
            given. all arguments must have the same shape or length. Howevver,
            if one angle is always the same its enough to give one scalar
            value.

        kwargs :    dict, optional
            optional keyword arguments
        delta :     list, tuple or array-like, optional
            giving delta angles to correct the given ones for misalignment
            delta must be an numpy array or list of length 3. Used angles are
            then ``ai, tt, beta - delta``
        UB :        array-like, optional
            matrix for conversion from (hkl) coordinates to Q of sample used to
            determine not Q but (hkl) (default: identity matrix)
        wl :        float or str, optional
            x-ray wavelength in angstrom (default: self._wl)
        deg :       bool, optional
            flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        ndarray
            reciprocal space positions as numpy.ndarray with shape `(N , 3)`
            where `N` corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine


class PowderExperiment(Experiment):
    """
    Experimental class for powder diffraction which helps to convert theta
    angles to momentum transfer space
    """

    def __init__(self, **kwargs):
        """
        class constructor which takes the same keyword arguments as the
        Experiment class

        Parameters
        ----------
        kwargs :     dict, optional
            keyword arguments same as for the Experiment base class
        """
        Experiment.__init__(self, [0, 1, 0], [0, 0, 1], **kwargs)
        self.Ang2Q = self._Ang2Q

    def _Ang2Q(self, th, wl=None, deg=True):
        """
        Converts theta angles to reciprocal space positions
        returns the absolute value of momentum transfer
        """
        if deg:
            lth = numpy.radians(th)
        else:
            lth = th

        if wl:
            k0 = 2 * numpy.pi / wl
        else:
            k0 = self.k0

        qpos = 2 * k0 * numpy.sin(lth)
        return qpos

    def Q2Ang(self, qpos, wl=None, deg=True):
        """
        Converts reciprocal space values to theta angles
        """
        if wl:
            k0 = 2 * numpy.pi / wl
        else:
            k0 = self.k0
        th = numpy.arcsin(numpy.divide(qpos, (2 * k0)))

        if deg:
            th = numpy.degrees(th)

        return th
