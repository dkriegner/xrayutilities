# This file is part of xrayutilities.
#
# xrayutilies is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the additonal notes below for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2015-2017 Marcus H. Mendenhall <marcus.mendenhall@nist.gov>
# Copyright (c) 2017-2023 Dominik Kriegner <dominik.kriegner@gmail.com>

# FP_profile was derived from http://dx.doi.org/10.6028/jres.120.014.c

# Original copyright notice:
# @author Marcus H. Mendenhall (marcus.mendenhall@nist.gov)
# @date March, 2015
# The "Fundamental Parameters Python Code" ("software") is provided by the
# National Institute of Standards and Technology (NIST), an agency of the
# United States Department of Commerce, as a public service.  This software is
# to be used for non-commercial research purposes only and is expressly
# provided "AS IS." Use of this software is subject to your acceptance of these
# terms and conditions.
#
# NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY
# OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA
# ACCURACY.  NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE
# SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
# CORRECTED.  NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE
# USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
# CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
#
# NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY
# INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES
# FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
# INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR
# OTHERWISE, ARISING FROM OR RELATING TO THE SOFTWARE (OR THE USE OF OR
# INABILITY TO USE THIS SOFTWARE), EVEN IF NIST HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGES.
#
# Software developed by NIST employees is not subject to copyright protection
# within the United States.  By using this software, creating derivative works
# or by incorporating this software into another product, you agree that you
# will use the software only for non-commercial research purposes and will
# indemnify and hold harmless the United States Government for any and all
# damages or liabilities that arise out of any use by you.
#
# changelog:
#
# 15 August, 2018 -- MHM
# fixed apparent error in flip of eps0 across twotheta=90 around line 681
#

"""
This module contains the core definitions for the XRD Fundamental Parameneters
Model (FPA) computation in Python.  The main computational class is FP_profile,
which stores cached information to allow it to efficiently recompute profiles
when parameters have been modified.  For the user an Powder class is available
which can calculate a complete powder pattern of a crystalline material.

The diffractometer line profile functions are calculated by methods from Cheary
& Coelho 1998 and Mullen & Cline paper and 'R' package.  Accumulate all
convolutions in Fourier space, for efficiency, except for axial divergence,
which needs to be weighted in real space for I3 integral.

More details about the applied algorithms can be found in the paper by
M. H. Mendelhall et al., `Journal of Research of NIST 120, 223 (2015)
<http://dx.doi.org/10.6028/jres.120.014>`_ to which you should also refer for a
careful definition of all the parameters
"""

# Known bugs/problems:
# in the axial convolver the parameters slit_length_source can not be equal to
# slit_length_target!

# in this file SI units (m) are used for wavelengths, while by default angstrom
# are used in the remaining of the package

import atexit
import copy
import math
import multiprocessing
import numbers
import queue
import sys
import threading
import time
import warnings
from math import cos, pi, sin, sqrt, tan
from multiprocessing.managers import BaseManager

import numpy
from numpy import abs as nabs
from numpy import arcsin as nasin
from numpy import asarray
from numpy import cos as ncos
from numpy import sin as nsin
from numpy import sqrt as nsqrt
from scipy.special import sici  # for the sine and cosine integral

# package internal imports
from .. import config, materials, utilities
from ..experiment import PowderExperiment
from ..math import VecAngle
from .smaterials import Powder

# figure out which FFT package we have, and import it
try:
    from pyfftw.interfaces import cache, numpy_fft

    # recorded variant of real fft that we will use
    best_rfft = numpy_fft.rfft
    # recorded variant of inverse real fft that we will use
    best_irfft = numpy_fft.irfft
    cache.enable()
    cache.set_keepalive_time(1.0)
except ImportError:
    best_rfft = numpy.fft.rfft
    best_irfft = numpy.fft.irfft

# create a table of nice factorizations for the FFT package
#
# this is built once, and shared by all instances
# fftw can handle a variety of transform factorizations
# numpy fft is not too efficient for things other than a power of two,
# although my own measurements says it really does fine.  For now, leave
# all factors available
ft_factors = [
    2*2**i*3**j*5**k for i in range(20) for j in range(10) for k in range(8)
    if 2*2**i*3**j*5**k <= 1000000
]

ft_factors.sort()
ft_factors = numpy.array(ft_factors, int)

# used for debugging moments from FP_profile.axial_helper().
moment_list = []
# if this is *True*, compute and save moment errors
collect_moment_errors = False


class profile_data:
    """
    a skeleton class which makes a combined dict and namespace interface for
    easy pickling and data passing
    """

    def __init__(self, **kwargs):
        """
        initialize the class

        Parameters
        ----------
        kwargs :    dict
            keyword=value list to pre-populate the class
        """
        mydict = {}
        mydict.update(kwargs)
        for k, v in mydict.items():
            setattr(self, k, v)
        # a dictionary which shadows our attributes.
        self.dictionary = mydict

    def add_symbol(self, **kwargs):
        """
        add new symbols to both the attributes and dictionary for the class

        Parameters
        ----------
        kwargs :    dict
            keyword=value pairs
        """
        self.dictionary.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class FP_profile:
    """
    the main fundamental parameters class, which handles a single reflection.
    This class is designed to be highly extensible by inheriting new
    convolvers.  When it is initialized, it scans its namespace for specially
    formatted names, which can come from mixin classes.  If it finds a function
    name of the form conv_xxx, it will call this funtion to create a
    convolver.  If it finds a name of the form info_xxx it will associate the
    dictionary with that convolver, which can be used in UI generation, for
    example.  The class, as it stands, does nothing significant with it.  If it
    finds str_xxx, it will use that function to format a printout of the
    current state of the convolver conv_xxx, to allow improved report
    generation for convolvers.

    When it is asked to generate a profile, it calls all known convolvers.
    Each convolver returns the Fourier transform of its convolvution.  The
    transforms are multiplied together, inverse transformed, and after fixing
    the periodicity issue, subsampled, smoothed and returned.

    If a convolver returns *None*, it is not multipled into the product.

    Parameters
    ----------
    max_history_length :    int
        the number of histories to cache (default=5); can be overridden if
        memory is an issue.
    length_scale_m :        float
        length_scale_m sets scaling for nice printing of parameters.  if the
        units are in mm everywhere, set it to 0.001, e.g.  convolvers which
        implement their own str_xxx method may use this to format their
        results, especially if 'natural' units are not meters.  Typical is
        wavelengths and lattices in nm or angstroms, for example.
    """
    max_history_length = 5  # max number of histories for each convolver
    length_scale_m = 1.0
    # class attribute to tell if convolvers in this class
    # contain anisotropic convolvers
    isotropic = True

    def __init__(self, anglemode,
                 gaussian_smoother_bins_sigma=1.0,
                 oversampling=10):
        """
        initialize the instance

        Parameters
        ----------
        anglemode : {'d', 'twotheta'}
            if setup will be in terms of a d-spacing, otherwise 'twotheta' if
            setup will be at a fixed 2theta value.
        gaussian_smoother_bins_sigma :  float
            the number of bins for post-smoothing of data. 1.0 is good. *None*
            means no final smoothing step.
        oversampling : int
            the number of bins internally which will get computed for each bin
            the the final result.
        """
        if anglemode not in ("d", "twotheta"):
            raise Exception(
                f"invalid angle mode {anglemode}, must be 'd' or 'twotheta'")
        # set to either 'd' for d-spacing based position, or 'twotheta' for
        # angle-based position
        self.anglemode = anglemode
        # sigma, in units of bins, for the final smoother.
        self.gaussian_smoother_bins_sigma = gaussian_smoother_bins_sigma
        # the number of internal bins computed for each bin in the final
        # output. 5-10 is usually plenty.
        self.oversampling = oversampling
        # List of our convolvers, found by introspection of names beginning
        # with 'conv_'
        self.convolvers = convolvers = [
            x for x in dir(self) if x.startswith("conv_")]
        # A dictionary which will store all the parameters local to each
        # convolution
        self.param_dicts = {c: {} for c in convolvers}
        # add global parameters, associated with no specific convolver
        # A dictionary of bound functions to call to compute convolutions
        self.convolver_funcs = {x: getattr(self, x) for x in convolvers}
        # If *True*, print cache hit information
        self.debug_cache = False
        # keep a record of things we don't keep when pickled
        self._clean_on_pickle = set()

    @classmethod
    def isequivalent(cls, hkl1, hkl2, crystalsystem):
        """
        function to determine if according to the convolvers included in this
        class two sets of Miller indices are equivalent. This function is only
        called when the class attribute 'isotropic' is False.

        Parameters
        ----------
        hkl1, hkl2 :    list or tuple
            Miller indices to be checked for equivalence
        crystalsystem : str
            symmetry class of the material which is considered

        Returns
        -------
        bool
        """
        return True

    def get_function_name(self):
        """
        return the name of the function that called this. Useful for convolvers
        to identify themselves

        Returns
        -------
        str
            name of calling function
        """
        return sys._getframe(1).f_code.co_name

    def add_buffer(self, b):
        """
        add a numpy array to the list of objects that can be thrown away on
        pickling.

        Parameters
        ----------
        b :     array-like
            the buffer to add to the list

        Returns
        -------
        b :     array-like
            return the same buffer, to make nesting easy.
        """
        self._clean_on_pickle.add(id(b))
        return b

    def set_window(self, twotheta_window_center_deg,
                   twotheta_window_fullwidth_deg, twotheta_output_points):
        """
        move the compute window to a new location and compute grids, without
        resetting all parameters.  Clears convolution history and sets up many
        arrays.

        Parameters
        ----------
        twotheta_window_center_deg :    float
            the center position of the middle bin of the window, in degrees
        twotheta_window_fullwidth_deg : float
            the full width of the window, in degrees
        twotheta_output_points :        int
            the number of bins in the final output
        """
        # the saved width of the window, in degrees
        self.twotheta_window_fullwidth_deg = twotheta_window_fullwidth_deg
        # the saved center of the window, in degrees
        self.twotheta_window_center_deg = twotheta_window_center_deg
        # the width of the window, in radians
        window_fullwidth = math.radians(twotheta_window_fullwidth_deg)
        self.window_fullwidth = window_fullwidth
        # the center of the window, in radians
        twotheta = math.radians(twotheta_window_center_deg)
        self.twotheta_window_center = twotheta
        # the number of points to compute in the final results
        self.twotheta_output_points = twotheta_output_points
        # the number of points in Fourier space to compute
        nn = self.oversampling * twotheta_output_points // 2 + 1
        self.n_omega_points = nn
        # build all the arrays
        b = self.add_buffer  # shortcut

        # a real-format scratch buffer
        self._rb1 = b(numpy.zeros(nn, float))
        # a real-format scratch buffer
        self._rb2 = b(numpy.zeros(nn, float))
        # a real-format scratch buffer
        self._rb3 = b(numpy.zeros(nn, float))
        # a complex-format scratch buffer
        self._cb1 = b(numpy.zeros(nn, complex))
        # a scratch buffer used by the axial helper
        self._f0buf = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, float))
        # a scratch buffer used for axial divergence
        self._epsb2 = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, float))
        # the I2+ buffer
        self._I2p = b(numpy.zeros(self.oversampling *
                                  twotheta_output_points, float))
        # the I2- buffer
        self._I2m = b(numpy.zeros(self.oversampling *
                                  twotheta_output_points, float))
        # another buffer used for axial divergence
        self._axial = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, float))
        # the largest frequency in Fourier space
        omega_max = self.n_omega_points * 2 * pi / window_fullwidth
        # build the x grid and the complex array that is the convolver
        # omega is in inverse radians in twotheta space (!)
        # i.e. if a transform is written
        # I(ds) = integral(A(L) exp(2 pi i L ds) dL
        # where L is a real-space length, and s=2 sin(twotheta/2)/lambda
        # then ds=2*pi*omega*cos(twotheta/2)/lambda (double check this!)
        # The grid in Fourier space, in inverse radians
        self.omega_vals = b(numpy.linspace(
            0, omega_max, self.n_omega_points, endpoint=True))
        # The grid in Fourier space, in inverse degrees
        self.omega_inv_deg = b(numpy.radians(self.omega_vals))
        # The grid in real space, in radians, with full oversampling
        self.twothetasamples = b(numpy.linspace(
            twotheta - window_fullwidth/2.0, twotheta + window_fullwidth/2.0,
            self.twotheta_output_points * self.oversampling, endpoint=False))
        # The grid in real space, in degrees, with full oversampling
        self.twothetasamples_deg = b(numpy.linspace(
            twotheta_window_center_deg - twotheta_window_fullwidth_deg/2.0,
            twotheta_window_center_deg + twotheta_window_fullwidth_deg/2.0,
            self.twotheta_output_points * self.oversampling, endpoint=False))
        # Offsets around the center of the window, in radians
        self.epsilon = b(self.twothetasamples - twotheta)

        # A dictionary in which we collect recent state for each convolution.
        # whenever the window gets reset, all of these get cleared
        self.convolution_history = {x: [] for x in self.convolvers}

        # A dictionary of Lorentz widths, used for de-periodizing the final
        # result.
        self.lor_widths = {}

    def get_good_bin_count(self, count):
        """
        find a bin count close to what we need, which works well for Fourier
        transforms.

        Parameters
        ----------
        count :     int
            a number of bins.

        Returns
        -------
        int
            a bin count somewhat larger than *count* which is efficient for FFT
        """
        return ft_factors[ft_factors.searchsorted(count)]

    def set_optimized_window(self, twotheta_window_center_deg,
                             twotheta_approx_window_fullwidth_deg,
                             twotheta_exact_bin_spacing_deg):
        """
        pick a bin count which factors cleanly for FFT, and adjust the window
        width to preserve the exact center and bin spacing

        Parameters
        ----------
        twotheta_window_center_deg :            float
            exact position of center bin, in degrees
        twotheta_approx_window_fullwidth_deg:   float
            approximate desired width
        twotheta_exact_bin_spacing_deg:         float
            the exact bin spacing to use
        """
        bins = self.get_good_bin_count(
            int(1 + twotheta_approx_window_fullwidth_deg /
                twotheta_exact_bin_spacing_deg))
        window_actwidth = twotheta_exact_bin_spacing_deg * bins
        self.set_window(twotheta_window_center_deg=twotheta_window_center_deg,
                        twotheta_window_fullwidth_deg=window_actwidth,
                        twotheta_output_points=bins)

    def set_parameters(self, convolver="global", **kwargs):
        """
        update the dictionary of parameters associated with the given convolver

        Parameters
        ----------
        convolver :     str
            the name of the convolver.  name 'global', e.g., attaches to
            function 'conv_global'
        kwargs :        dict
            keyword-value pairs to update the convolvers dictionary.
        """
        self.param_dicts["conv_" + convolver].update(kwargs)

    def get_conv(self, name, key, format=float):
        """
        get a cached, pre-computed convolver associated with the given
        parameters, or a newly zeroed convolver if the cache doesn't contain
        it. Recycles old cache entries.

        This takes advantage of the mutability of arrays.  When the contents of
        the array are changed by the convolver, the cached copy is implicitly
        updated, so that the next time this is called with the same parameters,
        it will return the previous array.

        Parameters
        ----------
        name :      str
            the name of the convolver to seek
        key :       object
            any hashable object which identifies the parameters for the
            computation
        format :    numpy.dtype, optional
            the type of the array to create, if one is not found.

        Returns
        -------
        bool
            flag, which is *True* if valid data were found, or *False* if the
            returned array is zero, and *array*, which must be computed by the
            convolver if *flag* was *False*.
        """

        # previous computed values as a list
        history = self.convolution_history[name]
        for idx, (k, b) in enumerate(history):
            if k == key:
                # move to front to mark recently used
                history.insert(0, history.pop(idx))
                if self.debug_cache:
                    print(name, True, file=sys.stderr)
                return True, b  # True says we got a buffer with valid data
        if len(history) == self.max_history_length:
            buf = history.pop(-1)[1]  # re-use oldest buffer
            buf[:] = 0
        else:
            buf = numpy.zeros(self.n_omega_points, format)
        history.insert(0, (key, buf))
        if self.debug_cache:
            print(name, False, file=sys.stderr)
        return False, buf  # False says buffer is empty, need to recompute

    def get_convolver_information(self):
        """
        return a list of convolvers, and what we know about them. function
        scans for functions named conv_xxx, and associated info_xxx entries.

        Returns
        -------
        list
            list of (convolver_xxx, info_xxx) pairs
        """
        info_list = []
        for k, f in self.convolver_funcs.items():
            info = getattr(self, "info_" + k[5:], {})
            info["docstring"] = f.__doc__
            info_list.append((k, info))

        return info_list

    # A dictionary of default parameters for the global namespace,
    # used to seed a GUI which can harvest this for names, descriptions, and
    # initial values
    info_global = dict(
        group_name="Global parameters",
        help="this should be help information",
        param_info=dict(
            twotheta0_deg=("Bragg center of peak (degrees)", 30.0),
            d=("d spacing (m)", 4.00e-10),
            dominant_wavelength=(
                "wavelength of most intense line (m)", 1.5e-10)
        )
    )

    def __str__(self):
        """
        return a nicely formatted report describing the current state of this
        class. this looks for an str_xxx function associated with each conv_xxx
        name.  If it is found, that function if called to get the state of
        conv_xxx.  Otherwise, this simply formats the dictionary of parameters
        for the convolver, and uses that.

        Returns
        -------
        str
            string of formatted information
        """
        keys = list(self.convolver_funcs)
        keys.sort()  # always return info in the same order
        # global is always first, anyways!
        keys.insert(0, keys.pop(keys.index('conv_global')))
        strings = ["", f"***convolver id 0x{id(self):08x}:"]
        for k in keys:
            strfn = "str_" + k[5:]
            if hasattr(self, strfn):
                strings.append(getattr(self, strfn)())
            else:
                dd = self.param_dicts["conv_" + k[5:]]
                if dd:
                    strings.append(k[5:] + ": " + str(dd))
        return '\n'.join(strings)

    def str_global(self):
        """
        returns a string representation for the global context.

        Returns
        -------
        str
            report on global parameters.
        """
        # in case it's not initialized
        self.param_dicts["conv_global"].setdefault("d", 0)
        return "global: peak center=%(twotheta0_deg).4f, d=%(d).8g, eq. "\
               "div=%(equatorial_divergence_deg).3f" \
               % self.param_dicts["conv_global"]

    def conv_global(self):
        """
        a dummy convolver to hold global variables and information.
        the global context isn't really a convolver, returning *None* means
        ignore result

        Returns
        -------
        *None*
            always returns None
        """
        return None

    def axial_helper(self, outerbound, innerbound, epsvals, destination,
                     peakpos=0, y0=0, k=0):
        """
        the function F0 from the paper.  compute k/sqrt(peakpos-x)+y0 nonzero
        between outer & inner (inner is closer to peak) or k/sqrt(x-peakpos)+y0
        if reversed (i.e. if outer > peak) fully evaluated on a specified eps
        grid, and stuff into destination

        Parameters
        ----------
        outerbound :    float
            the edge of the function farthest from the singularity, referenced
            to epsvals
        innerbound :    float
            the edge closest to the singularity, referenced to epsvals
        epsvals :       array-like
            the array of two-theta values or offsets
        destination :   array-like
            an array into which final results are summed.  modified in place!
        peakpos :       float
            the position of the singularity, referenced to epsvals.
        y0 :            float
            the constant offset
        k :             float
            the scale factor

        Returns
        -------
        lower_index, upper_index :  int
            python style bounds for region of `destination` which has been
            modified.
        """
        if k == 0:
            # nothing to do, point at the middle
            return len(epsvals) // 2, len(epsvals) // 2 + 1

        # bin width for normalizer so sum(result*dx)=exact integral
        dx = epsvals[1] - epsvals[0]
        # flag for whether tail is to the left or right.
        flip = outerbound > peakpos

        delta1 = abs(innerbound - peakpos)
        delta2 = abs(outerbound - peakpos)

        # this is the analytic area the function must have,
        # integral(1/sqrt(eps0-eps)) from lower to upper
        exactintegral = 2 * k * (sqrt(delta2) - sqrt(delta1))
        exactintegral += y0 * (delta2 - delta1)
        # exactintegral=max(0, exactintegral) # can never be < 0, beta out of
        # range.
        exactintegral *= 1 / dx  # normalize so sum is right
        # compute the exact centroid we need for this
        if abs(delta2-delta1) < 1e-12:
            exact_moment1 = 0
        else:
            exact_moment1 = (  # simplified from Mathematica FortranForm
                (4*k*(delta2**1.5-delta1**1.5) + 3*y0*(delta2**2-delta1**2)) /
                (6.*(2*k*(sqrt(delta2)-sqrt(delta1)) + y0*(delta2-delta1)))
            )
            if not flip:
                exact_moment1 = -exact_moment1
        exact_moment1 += peakpos

        # note: because of the way the search is done, this produces a result
        # with a bias of 1/2 channel to the left of where it should be.
        # this is fixed by shifting all the parameters up 1/2 channel
        outerbound += dx / 2
        innerbound += dx / 2
        peakpos += dx / 2  # fix 1/2 channel average bias from search
        # note: searchsorted(side="left") always returns the position of the
        # bin to the right of the match, or exact bin
        idx0, idx1 = epsvals.searchsorted(
            (outerbound, innerbound), side='left')

        # peak has been squeezed out, nothing to do
        if abs(outerbound - innerbound) < (2*dx) or abs(idx1 - idx0) < 2:
            # preserve the exact centroid: requires summing into two channels
            # for a peak this narrow, no attempt to preserve the width.
            # note that x1 (1-f1) + (x1+dx) f1 = mu has solution
            # (mu - x1) / dx  = f1 thus, we want to sum into a channel that has
            # x1<mu by less than dx, and the one to its right
            # pick left edge and make sure we are past it
            idx0 = min(idx0, idx1) - 1
            while exact_moment1 - epsvals[idx0] > dx:
                # normally only one step max, but make it a loop in case of
                # corner case
                idx0 += 1
            f1 = (exact_moment1 - epsvals[idx0]) / dx
            res = (exactintegral * (1 - f1), exactintegral * f1)
            destination[idx0:idx0 + 2] += res
            if collect_moment_errors:
                centroid2 = (res * epsvals[idx0:idx0 + 2]).sum() / sum(res)
                moment_list.append((centroid2 - exact_moment1) / dx)
            return [idx0, idx0 + 2]  # return collapsed bounds

        if not flip:
            if epsvals[idx0] != outerbound:
                idx0 = max(idx0 - 1, 0)
            idx1 = min(idx1 + 1, len(epsvals))
            sign = 1
            deps = self._f0buf[idx0:idx1]
            deps[:] = peakpos
            deps -= epsvals[idx0:idx1]
            deps[-1] = peakpos - min(innerbound, peakpos)
            deps[0] = peakpos - outerbound
        else:
            idx0, idx1 = idx1, idx0
            if epsvals[idx0] != innerbound:
                idx0 = max(idx0 - 1, 0)
            idx1 = min(idx1 + 1, len(epsvals))
            sign = -1
            deps = self._f0buf[idx0:idx1]
            deps[:] = epsvals[idx0:idx1]
            deps -= peakpos
            deps[0] = max(innerbound, peakpos) - peakpos
            deps[-1] = outerbound - peakpos

        dx0 = abs(deps[1] - deps[0])
        dx1 = abs(deps[-1] - deps[-2])

        # make the numerics accurate: compute average on each bin, which is
        # integral of 1/sqrt = 2*sqrt, then difference integral
        # do it in place, return value is actually deps too
        intg = numpy.sqrt(deps, deps)
        intg *= 2 * k * sign

        # do difference in place, running forward to avoid self-trampling
        intg[:-1] -= intg[1:]
        intg[1:-2] += y0 * dx  # add constant
        # handle narrowed bins on ends carefully
        intg[0] += y0 * dx0
        intg[-2] += y0 * dx1

        # intensities are never less than zero!
        if min(intg[:-1]) < -1e-10 * max(intg[:-1]):
            print("bad parameters:", (5 * "%10.4f") %
                  (peakpos, innerbound, outerbound, k, y0))
            print(len(intg), intg[:-1])
            raise ValueError("Bad axial helper parameters")

        # now, make sure the underlying area is the exactly correct
        # integral, without bumps due to discretizing the grid.
        intg *= (exactintegral / (intg[:-1].sum()))

        destination[idx0:idx1 - 1] += intg[:-1]

        # This is purely for debugging.  If collect_moment_errors is *True*,
        #  compute exact vs. approximate moments.
        if collect_moment_errors:
            centroid2 = (intg[:-1] * epsvals[idx0:idx1 - 1]
                         ).sum() / intg[:-1].sum()
            moment_list.append((centroid2 - exact_moment1) / dx)

        return [idx0, idx1 - 1]  # useful info for peak position

    def full_axdiv_I2(self, Lx=None, Ls=None, Lr=None, R=None, twotheta=None,
                      beta=None, epsvals=None):
        """
        return the *I2* function

        Parameters
        ----------
        Lx :        float
            length of the xray filament
        Ls :        float
            length of the sample
        Lr :        float
            length of the receiver slit
        R :         float
            diffractometer length, assumed symmetrical
        twotheta :  float
            angle, in radians, of the center of the computation
        beta :      float
            offset angle
        epsvals :   array-like
            array of offsets from center of computation, in radians

        Returns
        -------
        epsvals :           array-like
            array of offsets from center of computation, in radians
        idxmin, idxmax :    int
            the full python-style bounds of the non-zero region of `I2p`
            and `I2m`
        I2p, I2m :          array-like
            I2+ and I2- from the paper, the contributions to the intensity
        """
        beta1 = (Ls - Lx) / (2 * R)  # Ch&Co after eq. 15abcd
        beta2 = (Ls + Lx) / (2 * R)  # Ch&Co after eq. 15abcd, corrected by KM

        eps0 = beta * beta * tan(twotheta) / 2  # after eq. 26 in Ch&Co

        if -beta2 <= beta < beta1:
            z0p = Lx / 2 + beta * R * (1 + 1 / cos(twotheta))
        elif beta1 <= beta <= beta2:
            z0p = Ls / 2 + beta * R / cos(twotheta)

        if -beta2 <= beta <= -beta1:
            z0m = -1 * Ls / 2 + beta * R / cos(twotheta)
        elif -beta1 < beta <= beta2:
            z0m = -1 * Lx / 2 + beta * R * (1 + 1 / cos(twotheta))

        epsscale = tan(pi / 2 - twotheta) / (2 * R * R)  # =cotan(twotheta)...

        # Ch&Co 18a&18b, KM sign correction
        eps1p = (eps0 - epsscale * ((Lr / 2) - z0p)**2)
        eps2p = (eps0 - epsscale * ((Lr / 2) - z0m)**2)
        # reversed eps2m and eps1m per KM R
        eps2m = (eps0 - epsscale * ((Lr / 2) + z0p)**2)
        eps1m = (eps0 - epsscale * ((Lr / 2) + z0m)**2)  # flip all epsilons

        if twotheta > pi/2:
            # this set of inversions from KM 'R' code, simplified here
            eps1p, eps2p, eps1m, eps2m = eps1m, eps2m, eps1p, eps2p

        # identify ranges per Ch&Co 4.2.2 and table 1 and select parameters
        # note table 1 is full of typos, but the minimized
        # tests from 4.2.2 with redundancies removed seem fine.
        if Lr > z0p - z0m:
            if z0p <= Lr/2 and z0m > -1*Lr/2:  # beam entirely within slit
                rng = 1
                ea = eps1p
                eb = eps2p
                ec = eps1m
                ed = eps2m
            elif (z0p > Lr/2 and z0m < Lr/2) or \
                    (z0m < -1*Lr/2 and z0p > -1*Lr/2):
                rng = 2
                ea = eps2p
                eb = eps1p
                ec = eps1m
                ed = eps2m
            else:
                rng = 3
                ea = eps2p
                eb = eps1p
                ec = eps1m
                ed = eps2m
        else:
            # beam hanging off both ends of slit, peak centered
            if z0m < -1*Lr/2 and z0p > Lr/2:
                rng = 1
                ea = eps1m
                eb = eps2p
                ec = eps1p
                ed = eps2m
            # one edge of beam within slit
            elif (-1*Lr/2 < z0m < Lr/2 and z0p > Lr/2) or \
                    (-1*Lr/2 < z0p < Lr/2 and z0m < -1*Lr/2):
                rng = 2
                ea = eps2p
                eb = eps1m
                ec = eps1p
                ed = eps2m
            else:
                rng = 3
                ea = eps2p
                eb = eps1m
                ec = eps1p
                ed = eps2m

        # now, evaluate function on bounds in table 1 based on ranges
        # note: because of a sign convention in epsilon, the bounds all get
        # switched

        # define them in our namespace so they inherit ea, eb, ec, ed, etc.
        def F1(dst, lower, upper, eea, eeb):
            return self.axial_helper(destination=dst,
                                     innerbound=upper, outerbound=lower,
                                     epsvals=epsvals, peakpos=eps0,
                                     k=sqrt(abs(eps0-eeb))-sqrt(abs(eps0-eea)),
                                     y0=0)

        def F2(dst, lower, upper, eea):
            return self.axial_helper(destination=dst,
                                     innerbound=upper, outerbound=lower,
                                     epsvals=epsvals, peakpos=eps0,
                                     k=sqrt(abs(eps0 - eea)), y0=-1)

        def F3(dst, lower, upper, eea):
            return self.axial_helper(destination=dst,
                                     innerbound=upper, outerbound=lower,
                                     epsvals=epsvals, peakpos=eps0,
                                     k=sqrt(abs(eps0 - eea)), y0=+1)

        def F4(dst, lower, upper, eea):
            # just like F2 but k and y0 negated
            return self.axial_helper(destination=dst,
                                     innerbound=upper, outerbound=lower,
                                     epsvals=epsvals, peakpos=eps0,
                                     k=-sqrt(abs(eps0 - eea)), y0=+1)

        I2p = self._I2p
        I2p[:] = 0
        I2m = self._I2m
        I2m[:] = 0

        indices = []
        if rng == 1:
            indices += F1(dst=I2p, lower=ea, upper=eps0, eea=ea, eeb=eb)
            indices += F2(dst=I2p, lower=eb, upper=ea,   eea=eb)
            indices += F1(dst=I2m, lower=ec, upper=eps0, eea=ec, eeb=ed)
            indices += F2(dst=I2m, lower=ed, upper=ec,   eea=ed)
        elif rng == 2:
            indices += F2(dst=I2p, lower=ea, upper=eps0, eea=ea)
            indices += F3(dst=I2m, lower=eb, upper=eps0, eea=ea)
            indices += F1(dst=I2m, lower=ec, upper=eb, eea=ec, eeb=ed)
            indices += F2(dst=I2m, lower=ed, upper=ec, eea=ed)
        elif rng == 3:
            indices += F4(dst=I2m, lower=eb, upper=ea, eea=ea)
            indices += F1(dst=I2m, lower=ec, upper=eb, eea=ec, eeb=ed)
            indices += F2(dst=I2m, lower=ed, upper=ec, eea=ed)

        idxmin = min(indices)
        idxmax = max(indices)

        return epsvals, idxmin, idxmax, I2p, I2m

    def full_axdiv_I3(self, Lx=None, Ls=None, Lr=None, R=None,
                      twotheta=None, epsvals=None, sollerIdeg=None,
                      sollerDdeg=None, nsteps=10, axDiv=""):
        """
        carry out the integral of *I2* over *beta* and the Soller slits.

        Parameters
        ----------
        Lx :        float
            length of the xray filament
        Ls :        float
            length of the sample
        Lr :        float
            length of the receiver slit
        R :         float
            the (assumed symmetrical) diffractometer radius
        twotheta :  float
            angle, in radians, of the center of the computation
        epsvals :   array-like
            array of offsets from center of computation, in radians
        sollerIdeg : float
            the full-width (both sides) cutoff angle of the incident Soller
            slit
        sollerDdeg : float
            the full-width (both sides) cutoff angle of the detector Soller
            slit
        nsteps :    int
            the number of subdivisions for the integral
        axDiv :     str
            not used

        Returns
        -------
        array-like
            the accumulated integral, a copy of a persistent buffer *_axial*
        """
        beta2 = (Ls + Lx) / (2 * R)  # Ch&Co after eq. 15abcd, corrected by KM

        if sollerIdeg is not None:
            solIrad = math.radians(sollerIdeg) / 2

            def solIfunc(x):
                return numpy.clip(1.0 - abs(x / solIrad), 0, 1)
            beta2 = min(beta2, solIrad)  # no point going beyond Soller
        else:
            def solIfunc(x):
                return numpy.ones_like(x)
        if sollerDdeg is not None:
            solDrad = math.radians(sollerDdeg) / 2

            def solDfunc(x):
                return numpy.clip(1.0 - abs(x / solDrad), 0, 1)
        else:
            def solDfunc(x):
                return numpy.ones_like(x)

        accum = self._axial
        accum[:] = 0

        if twotheta > pi / 2:
            tth1 = pi - twotheta
        else:
            tth1 = twotheta

        for iidx in range(nsteps):
            beta = beta2 * iidx / float(nsteps)

            _, idxmin, idxmax, I2p, I2m = self.full_axdiv_I2(
                Lx=Lx, Lr=Lr, Ls=Ls, beta=beta, R=R,
                twotheta=twotheta, epsvals=epsvals)

            # after eq. 26 in Ch&Co
            eps0 = beta * beta * tan(twotheta) / 2

            gamma0 = beta / cos(tth1)
            deps = self._f0buf[idxmin:idxmax]
            deps[:] = eps0
            deps -= epsvals[idxmin:idxmax]
            deps *= 2 * tan(twotheta)
            # check two channels on each end for negative argument.
            deps[-1] = max(deps[-1], 0)
            deps[0] = max(deps[0], 0)
            if len(deps) >= 2:
                deps[-2] = max(deps[-2], 0)
                deps[1] = max(deps[1], 0)

            gamarg = numpy.sqrt(deps, deps)  # do sqrt in place for speed
            # still need to convert these to in-place
            gamp = gamma0 + gamarg
            gamm = gamma0 - gamarg

            if iidx == 0 or iidx == nsteps - 1:
                weight = 1.0  # trapezoidal rule weighting
            else:
                weight = 2.0

            # sum into the accumulator only channels which can be non-zero
            # do scaling in-place to save a  lot of slow array copying
            I2p[idxmin:idxmax] *= solDfunc(gamp)
            I2p[idxmin:idxmax] *= (weight * solIfunc(beta))
            accum[idxmin:idxmax] += I2p[idxmin:idxmax]
            I2m[idxmin:idxmax] *= solDfunc(gamm)
            I2m[idxmin:idxmax] *= (weight * solIfunc(beta))
            accum[idxmin:idxmax] += I2m[idxmin:idxmax]

        # keep this normalized
        K = 2 * R * R * abs(tan(twotheta))
        accum *= K

        return accum

    def conv_axial(self):
        """
        compute the Fourier transform of the axial divergence component

        Returns
        -------
        array-like
            the transform buffer, or *None* if this is being ignored
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        if self.param_dicts[me].get("axDiv", None) is None:
            return None
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        kwargs.update(self.param_dicts["conv_global"])
        if "equatorial_divergence_deg" in kwargs:
            del kwargs["equatorial_divergence_deg"]  # not used

        flag, axfn = self.get_conv(me, kwargs, complex)
        if flag:
            return axfn  # already up to date if first return is True

        # no axial divergence, transform of delta fn
        if kwargs["axDiv"] != "full" or kwargs["twotheta0_deg"] == 90.0:
            axfn[:] = 1
            return axfn
        axbuf = self.full_axdiv_I3(
            nsteps=kwargs["n_integral_points"],
            epsvals=self.epsilon,
            Lx=kwargs["slit_length_source"],
            Lr=kwargs["slit_length_target"],
            Ls=kwargs["length_sample"],
            sollerIdeg=kwargs["angI_deg"],
            sollerDdeg=kwargs["angD_deg"],
            R=kwargs["diffractometer_radius"],
            twotheta=kwargs["twotheta0"]
        )
        axfn[:] = best_rfft(axbuf)

        return axfn

    def conv_tube_tails(self):
        """
        compute the Fourier transform of the rectangular tube tails function

        Returns
        -------
        array-like
            the transform buffer, or *None* if this is being ignored
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None  # no convolver
        # we also need the diffractometer radius from the global space
        kwargs["diffractometer_radius"] = self.param_dicts[
            "conv_global"]["diffractometer_radius"]
        flag, tailfn = self.get_conv(me, kwargs, complex)
        if flag:
            return tailfn  # already up to date

        # tube_tails is (main width, left width, right width, intensity),
        # so peak is raw peak + tophat centered at (left width+ right width)
        # with area intensity*(right width-left width)/main_width
        # check this normalization!
        # note: this widths are as defined by Topas... really I think it should
        # be
        # x/(2*diffractometer_radius) since the detector is 2R from the source,
        # but since this is just a fit parameter, we'll defin it as does Topas

        tail_eps = (kwargs["tail_right"] - kwargs["tail_left"]) / \
            kwargs["diffractometer_radius"]
        main_eps = kwargs["main_width"] / kwargs["diffractometer_radius"]
        tail_center = (kwargs["tail_right"] + kwargs["tail_left"]) / \
            kwargs["diffractometer_radius"] / 2.0
        tail_area = kwargs["tail_intens"] * \
            (kwargs["tail_right"] - kwargs["tail_left"]) / kwargs["main_width"]

        cb1 = self._cb1
        rb1 = self._rb1

        cb1.real = 0
        cb1.imag = self.omega_vals
        cb1.imag *= tail_center  # sign is consistent with Topas definition
        numpy.exp(cb1, tailfn)  # shifted center, computed into tailfn

        rb1[:] = self.omega_vals
        rb1 *= (tail_eps / 2 / pi)
        rb1 = numpy.sinc(rb1)
        tailfn *= rb1
        tailfn *= tail_area  # normalize

        rb1[:] = self.omega_vals
        rb1 *= (main_eps / 2 / pi)
        rb1 = numpy.sinc(rb1)
        tailfn += rb1  # add central peak
        return tailfn

    def general_tophat(self, name="", width=None):
        """
        a utility to compute a transformed tophat function and save it in a
        convolver buffer

        Parameters
        ----------
        name :      str
            the name of the convolver cache buffer to update
        width :     float
            the width in 2-theta space of the tophat

        Returns
        -------
        array-like
            the updated convolver buffer, or *None* if the width was *None*
        """
        if width is None:
            return None  # no convolver
        flag, conv = self.get_conv(name, width, float)
        if flag:
            return conv  # already up to date
        rb1 = self._rb1
        rb1[:] = self.omega_vals
        rb1 *= (width / 2 / pi)
        conv[:] = numpy.sinc(rb1)
        return conv

    # A dictionary of default parameters for conv_emissions,
    # used to seed a GUI which can harvest this for names, descriptions, and
    # initial values
    info_emission = dict(
        group_name="Incident beam and crystal size",
        help="this should be help information",
        param_info=dict(
            emiss_wavelengths=("wavelengths (m)", (1.58e-10,)),
            emiss_intensities=("relative intensities", (1.00,)),
            emiss_lor_widths=("Lorenztian emission fwhm (m)", (1e-13,)),
            emiss_gauss_widths=("Gaussian emissions fwhm (m)", (1e-13,)),
            crystallite_size_gauss=(
                "Gaussian crystallite size fwhm (m)", 1e-6),
            crystallite_size_lor=(
                "Lorentzian crystallite size fwhm (m)", 1e-6),
        )
    )

    def str_emission(self):
        """
        format the emission spectrum and crystal size information

        Returns
        -------
        str
            the formatted information
        """
        dd = self.param_dicts["conv_emission"]
        if not dd:
            return "No emission spectrum"
        dd.setdefault("crystallite_size_lor", 1e10)
        dd.setdefault("crystallite_size_gauss", 1e10)
        dd.setdefault("strain_lor", 0)
        dd.setdefault("strain_gauss", 0)
        spect = numpy.array((
            dd["emiss_wavelengths"], dd["emiss_intensities"],
            dd["emiss_lor_widths"], dd["emiss_gauss_widths"]))
        # convert to angstroms, like Topas
        spect[0] *= 1e10 * self.length_scale_m
        spect[2] *= 1e13 * self.length_scale_m  # milli-angstroms
        spect[3] *= 1e13 * self.length_scale_m  # milli-angstroms
        nm = 1e9 * self.length_scale_m
        items = ["emission and broadening:"]
        items.append("spectrum=\n" + str(spect.transpose()))
        items.append("crystallite_size_lor (nm): "
                     f"{dd['crystallite_size_lor']*nm:.5g}")
        items.append("crystallite_size_gauss (nm): "
                     f"{dd['crystallite_size_gauss']*nm:.5g}")
        items.append(f"strain_lor: {dd['strain_lor']:.5g}")
        items.append(f"strain_gauss: {dd['strain_gauss']:.5g}")
        return '\n'.join(items)

    def conv_emission(self):
        """
        compute the emission spectrum and (for convenience) the particle size
        widths

        Returns
        -------
        array-like
            the convolver for the emission and particle sizes

        Note:
            the particle size and strain stuff here is just to be consistent
            with *Topas* and to be vaguely efficient about the computation,
            since all of these have the same general shape.
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        kwargs.update(self.param_dicts["conv_global"])
        # if the crystallite size and strain parameters are not set, set them
        # to values that make their corrections disappear
        kwargs.setdefault("crystallite_size_lor", 1e10)
        kwargs.setdefault("crystallite_size_gauss", 1e10)
        kwargs.setdefault("strain_lor", 0)
        kwargs.setdefault("strain_gauss", 0)

        # convert arrays to lists for key checking
        key = {}
        key.update(kwargs)
        for k, v in key.items():
            if hasattr(v, 'tolist'):
                key[k] = v.tolist()

        flag, emiss = self.get_conv(me, key, complex)
        if flag:
            return emiss  # already up to date

        epsilon0s = (
            2 * nasin(asarray(kwargs["emiss_wavelengths"])/(2.0*kwargs["d"])) -
            kwargs["twotheta0"]
        )
        theta = kwargs["twotheta0"] / 2
        # Emission profile FWHM + crystallite broadening (scale factors are
        # Topas choice!) (Lorentzian)
        # note: the strain broadenings in Topas are expressed in degrees
        # 2theta, must convert to radians(theta) with pi/360
        widths = (
            (asarray(kwargs["emiss_lor_widths"]) /
             asarray(kwargs["emiss_wavelengths"])) *
            tan(theta) + math.radians(kwargs["strain_lor"]) / 2 * tan(theta) +
            (asarray(kwargs["emiss_wavelengths"]) /
             (2*kwargs["crystallite_size_lor"]*cos(theta)))
        )
        # save weighted average width for future reference in periodicity fixer
        self.lor_widths[me] = sum(widths * kwargs["emiss_intensities"]) / \
            sum(kwargs["emiss_intensities"])
        # gaussian bits add in quadrature
        gfwhm2s = (
            ((2 * asarray(kwargs["emiss_gauss_widths"]) /
              asarray(kwargs["emiss_wavelengths"])) *
             tan(theta))**2 +
            (math.radians(kwargs["strain_gauss"]) / 2 * tan(theta))**2 +
            (asarray(kwargs["emiss_wavelengths"]) /
             (kwargs["crystallite_size_gauss"]*cos(theta)))**2
        )

        # note that the Fourier transform of a lorentzian with FWHM 2a
        # is exp(-abs(a omega))
        # now, the line profiles in Fourier space have to have phases
        # carefully handled to put the lines in the right places.
        # note that the transform of f(x+dx)=exp(i omega dx) f~(x)
        omega_vals = self.omega_vals
        for wid, gfwhm2, eps, intens in zip(widths, gfwhm2s, epsilon0s,
                                            kwargs["emiss_intensities"]):
            xvals = numpy.clip(omega_vals * (-wid), -100, 0)
            sig2 = gfwhm2 / (8 * math.log(2.0))  # convert fwhm**2 to sigma**2
            gxv = numpy.clip((sig2 / -2.0) * omega_vals * omega_vals, -100, 0)
            emiss += numpy.exp(xvals + gxv + complex(0, -eps) *
                               omega_vals) * intens
        return emiss

    def conv_flat_specimen(self):
        """
        compute the convolver for the flat-specimen correction

        Returns
        -------
        array-like
            the convolver
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        equatorial_divergence_deg = self.param_dicts[
            "conv_global"].get("equatorial_divergence_deg", None)
        if not equatorial_divergence_deg:
            return None
        twotheta0 = self.param_dicts["conv_global"]["twotheta0"]
        key = (twotheta0, equatorial_divergence_deg)
        flag, conv = self.get_conv(me, key, complex)
        if flag:
            return conv  # already up to date

        # Flat-specimen error, from Cheary, Coelho & Cline 2004 NIST eq. 9 & 10
        # compute epsm in radians from eq. divergence in degrees
        # to make it easy to use the axial_helper to compute the function
        epsm = math.radians(equatorial_divergence_deg)**2 /\
            tan(twotheta0/2.0) / 2.0
        eqdiv = self._epsb2
        eqdiv[:] = 0
        dtwoth = (self.twothetasamples[1] - self.twothetasamples[0])
        self.axial_helper(destination=eqdiv,
                          outerbound=-epsm,
                          innerbound=0,
                          epsvals=self.epsilon,
                          peakpos=0, k=dtwoth/(2.0*sqrt(epsm)))

        conv[:] = best_rfft(eqdiv)
        conv[1::2] *= -1  # flip center
        return conv

    def conv_absorption(self):
        """
        compute the sample transparency correction, including the
        finite-thickness version

        Returns
        -------
        array-like
            the convolver
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None
        kwargs["twotheta0"] = self.param_dicts["conv_global"]["twotheta0"]
        kwargs["diffractometer_radius"] = self.param_dicts[
            "conv_global"]["diffractometer_radius"]

        flag, conv = self.get_conv(me, kwargs, complex)
        if flag:
            return conv  # already up to date

        # absorption, from Cheary, Coelho & Cline 2004 NIST eq. 12,
        # EXCEPT delta = 1/(2*mu*R) instead of 2/(mu*R)
        # from Mathematica, unnormalized transform is
        # (1-exp(epsmin*(i w + 1/delta)))/(i w + 1/delta)
        delta = sin(kwargs["twotheta0"]) / \
            (2 * kwargs["absorption_coefficient"] *
             kwargs["diffractometer_radius"])
        # arg=(1/delta)+complex(0,-1)*omega_vals
        cb = self._cb1
        cb.imag = self.omega_vals
        cb.imag *= -1
        cb.real = 1 / delta
        numpy.reciprocal(cb, conv)  # limit for thick samples=1/(delta*arg)
        conv *= 1.0 / delta  # normalize
        # rest of transform of function with cutoff
        if kwargs.get("sample_thickness", None) is not None:
            epsmin = -2.0 * kwargs["sample_thickness"] * \
                cos(kwargs["twotheta0"] / 2.0) / \
                kwargs["diffractometer_radius"]
            cb *= epsmin
            numpy.expm1(cb, cb)
            cb *= -1
            conv *= cb
        return conv

    def conv_displacement(self):
        """
        compute the peak shift due to sample displacement and the *2theta* zero
        offset

        Returns
        -------
        array-like
            the convolver
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        kwargs = self.param_dicts[me]
        twotheta0 = self.param_dicts["conv_global"]["twotheta0"]
        diffractometer_radius = self.param_dicts[
            "conv_global"]["diffractometer_radius"]
        specimen_displacement = kwargs.get("specimen_displacement", 0.0)
        if specimen_displacement is None:
            specimen_displacement = 0.0
        zero_error_deg = kwargs.get("zero_error_deg", 0.0)
        if zero_error_deg is None:
            zero_error_deg = 0.0

        flag, conv = self.get_conv(me,
                                   (twotheta0, diffractometer_radius,
                                    specimen_displacement, zero_error_deg),
                                   complex)
        if flag:
            return conv  # already up to date

        delta = -2 * cos(twotheta0 / 2.0) * \
            specimen_displacement / diffractometer_radius
        conv.real = 0
        conv.imag = self.omega_vals
        # convolver *= numpy.exp(complex(0, -delta-zero_error_deg*pi/180.0) *
        #                        omega_vals)
        conv.imag *= (-delta - math.radians(zero_error_deg) -
                      (twotheta0 - self.twotheta_window_center))
        numpy.exp(conv, conv)
        return conv

    def conv_receiver_slit(self):
        """
        compute the rectangular convolution for the receiver slit or SiPSD
        pixel size

        Returns
        -------
        array-like
            the convolver
        """
        me = self.get_function_name()  # the name of the convolver, as a string
        # The receiver slit convolution is a top-hat of angular half-width
        # a=(slit_width/2)/diffractometer_radius
        # which has Fourier transform of sin(a omega)/(a omega)
        # NOTE! numpy's sinc(x) is sin(pi x)/(pi x), not sin(x)/x
        if self.param_dicts[me].get("slit_width", None) is None:
            return None

        epsr = (self.param_dicts["conv_receiver_slit"]["slit_width"] /
                self.param_dicts["conv_global"]["diffractometer_radius"])
        return self.general_tophat(me, epsr)

    def conv_si_psd(self):
        """
        compute the convolver for the integral of defocusing of the face of an
        Si PSD

        Returns
        -------
        array-like
            the convolver
        """
        # omega offset defocussing from Cheary, Coelho & Cline 2004 eq. 15
        # expressed in terms of a Si PSD looking at channels with vertical
        # offset from the center between psd_window_lower_offset and
        # psd_window_upper_offset do this last, because we may ultimately take
        # a list of bounds, and return a list of convolutions, for efficiency
        me = self.get_function_name()  # the name of the convolver, as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None
        kwargs.update(self.param_dicts["conv_global"])

        flag, conv = self.get_conv(me, kwargs, float)
        if flag:
            return conv  # already up to date

        if not (kwargs["equatorial_divergence_deg"] and
                kwargs["si_psd_window_bounds"]):
            # if either of these is zero or None, convolution is trivial
            conv[:] = 1
            return conv

        psd_lower_win_pos, psd_upper_win_pos = kwargs["si_psd_window_bounds"]
        dthl = psd_lower_win_pos / kwargs["diffractometer_radius"]
        dthu = psd_upper_win_pos / kwargs["diffractometer_radius"]
        alpha = math.radians(kwargs["equatorial_divergence_deg"])
        argscale = alpha / (2.0 * tan(kwargs["twotheta0"] / 2))
        # WARNING si(x)=integral(sin(x)/x), not integral(sin(pi x)/(pi x))
        # i.e. they sinc function is not consistent with the si function
        # whence the missing pi in the denominator of argscale
        rb1 = self._rb1
        rb2 = self._rb2
        rb3 = self._rb3
        rb1[:] = self.omega_vals
        rb1 *= argscale * dthu
        sici(rb1, conv, rb3)  # gets both sine and cosine integral, si in conv
        if dthl:  # no need to do this if the lower bound is 0
            rb1[:] = self.omega_vals
            rb1 *= argscale * dthl
            sici(rb1, rb2, rb3)  # gets sine and cosine integral, si in rb2
            conv -= rb2
        conv[1:] /= self.omega_vals[1:]
        conv *= 1 / argscale
        conv[0] = dthu - dthl  # fix 0/0 with proper area
        return conv

    def conv_smoother(self):
        """
        compute the convolver to smooth the final result with a Gaussian before
        downsampling.

        Returns
        -------
        array-like
            the convolver
        """
        # create a smoother for output result, independent of real physics, if
        # wanted
        me = self.get_function_name()  # the name of the convolver, as a string
        if not self.gaussian_smoother_bins_sigma:
            return None  # no smoothing
        flag, buf = self.get_conv(me, self.gaussian_smoother_bins_sigma,
                                  format=float)
        if flag:
            return buf  # already computed
        buf[:] = self.omega_vals
        buf *= (self.gaussian_smoother_bins_sigma * (
            self.twothetasamples[1] - self.twothetasamples[0]))
        buf *= buf
        buf *= -0.5
        numpy.exp(buf, buf)
        return buf

    def compute_line_profile(self, convolver_names=None,
                             compute_derivative=False,
                             return_convolver=False):
        """
        execute all the convolutions; if convolver_names is None, use
        everything we have, otherwise, use named convolutions.

        Parameters
        ----------
        convolver_names:    list
            a list of convolvers to select. If *None*, use all found
            convolvers.
        compute_derivative: bool
            if *True*, also return d/dx(function) for peak position fitting

        Returns
        -------
        object
            a profile_data object with much information about the peak
        """

        # create a function which is the Fourier transform of the
        # combined convolutions of all the factors

        # ="d" if we are using 'd' space, "twotheta" if using twotheta
        anglemode = self.anglemode

        # the rough center of the spectrum, used for things which need it.
        # Copied from global convolver.
        self.dominant_wavelength = dominant_wavelength = self.param_dicts[
            "conv_global"].get("dominant_wavelength", None)

        if anglemode == "twotheta":
            twotheta0_deg = self.param_dicts["conv_global"]["twotheta0_deg"]
            twotheta0 = math.radians(twotheta0_deg)
            d = dominant_wavelength / (2 * sin(twotheta0 / 2.0))
        else:
            d = self.param_dicts["conv_global"]["d"]
            twotheta0 = 2 * math.asin(dominant_wavelength / (2.0 * d))
            twotheta0_deg = math.degrees(twotheta0)

        # set these in global namespace
        self.set_parameters(d=d, twotheta0=twotheta0,
                            twotheta0_deg=twotheta0_deg)

        if convolver_names is None:
            convolver_names = self.convolver_funcs.keys()  # get all names

        # run through the name list, and call the convolver to harvest its
        # result
        conv_list = [self.convolver_funcs[x]() for x in convolver_names]

        # now, multiply everything together
        convolver = self._cb1  # get a complex scratch buffer
        convolver[:] = 1  # initialize
        for c in conv_list:  # accumulate product
            if c is not None:
                convolver *= c

        if convolver[1].real > 0:  # recenter peak!
            convolver[1::2] *= -1

        peak = best_irfft(convolver)

        # now, use the trick from Mendenhall JQSRT Voigt paper to remove
        # periodic function correction
        # JQSRT 105 number 3 July 2007 p. 519 eq. 7
        # total lor widths, created by the various colvolvers
        correction_width = 2 * sum(self.lor_widths.values())

        d2p = 2.0 * pi / self.window_fullwidth
        alpha = correction_width / 2.0  # be consistent with convolver
        mu = (peak * self.twothetasamples).sum() / peak.sum()  # centroid
        dx = self.twothetasamples - mu
        eps_corr1 = (math.sinh(d2p * alpha) / self.window_fullwidth) / \
            (math.cosh(d2p * alpha) - ncos(d2p * dx))
        eps_corr2 = (alpha / pi) / (dx * dx + alpha * alpha)
        corr = (convolver[0].real / numpy.sum(eps_corr2)) * \
            (eps_corr1 - eps_corr2)
        peak -= corr

        peak *= self.window_fullwidth / \
            (self.twotheta_output_points / self.oversampling)  # scale to area

        if compute_derivative:
            # this is useful
            convolver *= self.omega_vals
            convolver *= complex(0, 1)
            deriv = best_irfft(convolver)
            deriv *= self.window_fullwidth / \
                (self.twotheta_output_points / self.oversampling)
            deriv = deriv[::self.oversampling]
        else:
            deriv = None

        result = profile_data(twotheta0_deg=math.degrees(twotheta0),
                              twotheta=self.twothetasamples[
                                  ::self.oversampling],
                              omega_inv_deg=self.omega_inv_deg[
                                  :self.twotheta_output_points // 2 + 1],
                              twotheta_deg=self.twothetasamples_deg[
                                  ::self.oversampling],
                              peak=peak[::self.oversampling],
                              derivative=deriv
                              )

        if return_convolver:
            result.add_symbol(
                convolver=convolver[:self.twotheta_output_points//2+1])

        return result

    def self_clean(self):
        """
        do some cleanup to make us more compact;
        Instance can no longer be used after doing this, but can be pickled.
        """
        clean = self._clean_on_pickle
        pd = dict()
        pd.update(self.__dict__)
        for thing in pd:
            x = getattr(self, thing)
            if id(x) in clean:
                delattr(self, thing)
        # delete extra attributes cautiously, in case we have already been
        # cleaned
        for k in ('convolver_funcs', 'convolvers',
                  'factors', 'convolution_history'):
            if pd.pop(k, None) is not None:
                delattr(self, k)

    def __getstate__(self):
        """
        return information for pickling.  Removes transient data from cache of
        shadow copy so resulting object is fairly compact.  This does not
        affect the state of the actual instance.

        Returns
        -------
        dict
            dictionary of sufficient information to reanimate instance.
        """
        #  do some cleanup on state before we get pickled
        # (even if main class not cleaned)
        clean = self._clean_on_pickle
        pd = dict()
        pd.update(self.__dict__)
        for thing in pd:
            x = getattr(self, thing)
            if id(x) in clean:
                del pd[thing]
        # delete extra attributes cautiously, in case we have already been
        # cleaned
        for k in ('convolver_funcs', 'convolvers',
                  'factors', 'convolution_history'):
            pd.pop(k, None)
        return pd

    def __setstate__(self, setdict):
        """
        reconstruct class from pickled information
        This rebuilds the class instance so it is ready to use on unpickling.

        Parameters
        ----------
        self :      FP_Profile
            an empty class instance
        setdict :   dict
            dictionary from FP_profile.__getstate__()
        """
        self.__init__(anglemode=setdict["anglemode"],
                      gaussian_smoother_bins_sigma=setdict[
                          "gaussian_smoother_bins_sigma"],
                      oversampling=setdict["oversampling"]
                      )
        for k, v in setdict.items():
            setattr(self, k, v)
        try:
            s = self
            self.set_window(
                twotheta_window_center_deg=s.twotheta_window_center_deg,
                twotheta_window_fullwidth_deg=s.twotheta_window_fullwidth_deg,
                twotheta_output_points=s.twotheta_output_points
            )
            # override clearing of this by set_window
            self.lor_widths = setdict["lor_widths"]
        except AttributeError:
            pass


class convolver_handler:
    """
    manage the convolvers of on process
    """

    def __init__(self):
        self.convolvers = []

    def add_convolver(self, convolver):
        self.convolvers.append(convolver)

    def update_parameters(self, parameters):
        for c in self.convolvers:
            for k, v in parameters.items():
                if k == 'classoptions':
                    continue
                c.set_parameters(convolver=k, **v)

    def set_windows(self, centers, npoints, flag, width):
        for c, cen, np, f in zip(self.convolvers, centers, npoints, flag):
            if f:
                c.set_window(twotheta_output_points=np,
                             twotheta_window_center_deg=cen,
                             twotheta_window_fullwidth_deg=width)

    def calc(self, run, ttpeaks):
        """
        calculate profile function for selected convolvers

        Parameters
        ----------
        run :   list
            list of flags of length of convolvers to tell which convolver needs
            to be run
        ttpeaks : array-like
            peak positions for the convolvers

        Returns
        -------
        list
            list of profile_data result objects
        """
        results = []
        for c, flag, tt in zip(self.convolvers, run, ttpeaks):
            if flag:
                c.set_parameters(twotheta0_deg=tt)
                res = c.compute_line_profile()
                del res.twotheta, res.omega_inv_deg
                del res.dictionary, res.derivative
                results.append(res)
            else:
                results.append(None)
        return results


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


class manager(BaseManager):
    pass


class PowderDiffraction(PowderExperiment):

    """
    Experimental class for powder diffraction. This class calculates the
    structure factors of powder diffraction lines and uses instances of
    FP_profile to perform the convolution with experimental resolution function
    calculated by the fundamental parameters approach. This class uses
    multiprocessing to speed up calculation. Set config.NTHREADS=1 to restrict
    this to one worker process.
    """

    _valid_init_kwargs = copy.copy(PowderExperiment._valid_init_kwargs)
    _valid_init_kwargs['tt_cutoff'] = '2Theta cut off value in degree'
    _valid_init_kwargs['fpclass'] = 'FP_profile derived class'
    _valid_init_kwargs['fpsettings'] = 'settings dictionaries'
    _valid_init_kwargs['enable_simulation'] = 'flag to enable simulation mode'
    del _valid_init_kwargs['sampleor']

    def __init__(self, mat, **kwargs):
        """
        the class is initialized with a xrayutilities.materials.Crystal
        instance and calculates the powder intensity and peak positions of the
        Crystal up to an angle of tt_cutoff. Results are stored in

            data .... array with intensities
            ang ..... Bragg angles of the peaks (Theta!)
            qpos .... reciprocal space position of intensities

        If `enable_simulation` is set to True the `Convolve` and `Calculate`
        methods can be used to calculated a powder pattern. Upon such
        initialization it is strongly recommend to call the `close` method to
        clean up the multiprocessing threads when no further calculations will
        be performed.

        Parameters
        ----------
        mat :           Crystal or Powder
            material for the powder calculation
        kwargs :        dict
            optional keyword arguments same as for the Experiment base class
            and:
        tt_cutoff :     float, optional
            Powder peaks are calculated up to an scattering angle of tt_cutoff
            (deg)
        enable_simulation:  False, optional
            enables the initialization of the convolvers. Call `close()` after
            you are finished with your instance!
        fpclass :       FP_profile
            FP_profile derived class with possible convolver mixins.
            (default: FP_profile)
        fpsettings :    dict
            settings dictionaries for the convolvers. Default settings are
            loaded from the config file.
        """
        utilities.check_kwargs(kwargs, self._valid_init_kwargs,
                               self.__class__.__name__)
        if isinstance(mat, materials.Crystal):
            self.mat = Powder(mat, 1)
        elif isinstance(mat, Powder):
            self.mat = mat
        else:
            raise TypeError("mat must be an instance of class "
                            "xrayutilities.materials.Crystal or "
                            "xrayutilities.simpack.Powder")

        self.data = dict()
        self._tt_cutoff = kwargs.pop('tt_cutoff', 180)
        self.fpclass = kwargs.pop('fpclass', FP_profile)
        self.settings = self.load_settings_from_config(
            kwargs.pop('fpsettings', {}))
        self.set_sample_parameters()  # initialize local settings
        self._enable_sim = kwargs.pop('enable_simulation', False)

        PowderExperiment.__init__(self, **kwargs)

        # number of significant digits, needed to identify equal floats
        self.digits = 5

        # determine if convolvers are isotropic
        self.isotropic = self.fpclass.isotropic

        # calculate powder lines position and intensities
        self.init_powder_lines(self._tt_cutoff)

        # initialize FP_profile instances (add field to the data dictionary)
        for h in self.data:
            self.data[h]['conv'] = self._init_fpprofile(self.fpclass)
        # set settings in all convolvers
        self.update_settings(self.settings)
        self.set_sample_parameters()

        # set some other class properties
        self.__tt = None
        self.__ww = None

        # initialize multiprocessing
        if self._enable_sim:
            self._init_multiprocessing()
        # set wavelength from class constructor
        if 'wl' in kwargs:
            self._set_wavelength_pd(kwargs['wl'])
        if 'en' in kwargs:
            self._set_energy_pd(kwargs['en'])

        if self.settings['global']['geometry'] != 'symmetric':
            warnings.warn("PowderDiffraction: geometry '%s' not fully "
                          "supported, proceed with caution!"
                          % self.settings['global']['geometry'])

    def _init_multiprocessing(self):
        """
        initialize multiprocessing for powder pattern calculation
        """
        # The structure of the multiprocessing code is as follows:
        # There are nproc "manager"s which handle the actual convolver code.
        # Additionally there are 4 daemon threads which listen for work to be
        # distributed to the managers.
        np = config.NTHREADS
        self.nproc = np if np != 0 else multiprocessing.cpu_count()
        self.chunks = chunkify(list(self.data), self.nproc)
        self.next_proc = len(self.data) % self.nproc
        manager.register("conv", convolver_handler)
        self.managers = [manager() for idx in range(self.nproc)]
        self.conv_handlers = []
        self.threads = []
        self.output_queue = queue.Queue()
        for idx, mg in enumerate(self.managers):
            mg.start()
            m = mg.conv()
            for h in self.chunks[idx]:
                m.add_convolver(self.data[h]['conv'])
            self.conv_handlers.append(m)
            self.threads.append((
                threading.Thread(target=self._send_work, args=(idx, )),
                queue.Queue(), self.output_queue))
        self._running = True
        for th, _, _ in self.threads:
            th.daemon = True
            th.start()
        atexit.register(self.__stop__)

    def close(self):
        self.__stop__()

    def __stop__(self):
        self._running = False
        try:  # try/except needed only for python2 compatibility
            # end daemon threads which distribute the work load
            for th, q1, _ in self.threads:
                q1.put(None)
                th.join()
            delattr(self, 'threads')
            # end managers which handle the convolvers
            delattr(self, 'conv_handlers')
            for m in self.managers:
                m.shutdown()
            delattr(self, 'managers')
        except AttributeError:
            pass
        atexit.unregister(self.__stop__)

    def load_settings_from_config(self, settings):
        """
        load parameters from the config and update these settings with the
        options from the settings parameter
        """
        params = dict()
        for k in config.POWDER:
            params[k] = dict()
            params[k].update(config.POWDER[k])
            if k in settings:
                params[k].update(settings[k])
        for k in settings:
            if k not in config.POWDER:
                params[k] = dict()
                params[k].update(settings[k])
        return params

    def _init_fpprofile(self, fpclass):
        """
        configure the default parameters of the FP_profile class and return an
        instance with these settings

        Parameters
        ----------
        fpclass :   FP_profile
            class with possible mixins which implement more convolvers

        Returns
        -------
        fpclass
            instance of fpclass
        """
        classparams = dict()
        classparams.update(self.settings['classoptions'])
        classparams.pop('window_width')
        p = fpclass(**classparams)
        p.debug_cache = False
        return p

    def _set_wavelength_pd(self, wl):
        PowderExperiment._set_wavelength(self, wl)
        s = {'emission': {'emiss_wavelengths': self.wavelength*1e-10}}
        self.update_settings(s)

    def _set_energy_pd(self, energy):
        PowderExperiment._set_energy(self, energy)
        s = {'emission': {'emiss_wavelengths': self.wavelength*1e-10}}
        self.update_settings(s)

    energy = property(PowderExperiment._get_energy, _set_energy_pd)
    wavelength = property(PowderExperiment._get_wavelength, _set_wavelength_pd)

    def set_wavelength_from_params(self):
        """
        sets the wavelenth in the base class from the settings dictionary of
        the FP_profile classes and also set it in the 'global' part of the
        parameters
        """
        if 'emission' in self.settings:
            pem = self.settings['emission']
            if 'emiss_wavelengths' in pem:
                wl = pem['emiss_wavelengths'][0]
                self.settings['global']['dominant_wavelength'] = wl
                for d in self.data.values():
                    fp = d['conv']
                    fp.set_parameters(convolver='global',
                                      **self.settings['global'])
                # set wavelength in base class
                PowderExperiment._set_wavelength(self, wl*1e10)

    def set_sample_parameters(self):
        """
        load sample parameters from the Powder class and use them in all
        FP_profile instances of this object
        """
        samplesettings = {}
        for prop, default in (('crystallite_size_lor', 1e10),
                              ('crystallite_size_gauss', 1e10),
                              ('strain_lor', 0),
                              ('strain_gauss', 0),
                              ('preferred_orientation', (0, 0, 0)),
                              ('preferred_orientation_factor', 1)):
            samplesettings[prop] = getattr(self.mat, prop, default)

        self.settings['emission'].update(samplesettings)
        for d in self.data.values():
            fp = d['conv']
            fp.set_parameters(convolver='emission', **samplesettings)

    def update_settings(self, newsettings=None):
        """
        update settings of all instances of FP_profile

        Parameters
        ----------
        newsettings :   dict, optional
            dictionary with new settings. It has to include one subdictionary
            for every convolver which should have its settings changed.
        """
        if newsettings is None:
            return
        if 'global' in newsettings:
            if 'dominant_wavelength' in newsettings['global']:
                print('PowderDiffraction: dominant wavelength is a read only'
                      'setting \n -> use emission: emiss_wavelength instead')
        if 'emission' in newsettings:
            nem = newsettings['emission']
            for k in ('emiss_wavelengths', 'emiss_intensities',
                      'emiss_gauss_widths', 'emiss_lor_widths'):
                if k in nem:
                    if isinstance(nem[k], numbers.Number):
                        nem[k] = (nem[k], )
        for k in newsettings:
            if k == 'classoptions':
                continue
            for d in self.data.values():
                fp = d['conv']
                fp.set_parameters(convolver=k, **newsettings[k])
            if k not in self.settings:
                self.settings[k] = dict()
            self.settings[k].update(newsettings[k])
        self.set_wavelength_from_params()

    @property
    def twotheta(self):
        return self.__tt

    @twotheta.setter
    def twotheta(self, tt):
        oldtt = self.__tt
        self.__tt = tt
        if oldtt is None:
            self.set_window()
        elif len(oldtt) != len(self.__tt):
            self.set_window(force=True)
        elif not numpy.all(numpy.equal(oldtt, self.__tt)):
            self.set_window(force=True)

    @property
    def window_width(self):
        return self.__ww

    @window_width.setter
    def window_width(self, ww):
        oldww = self.__ww
        if ww == 'config':
            self.__ww = config.POWDER['classoptions']['window_width']
        else:
            self.__ww = ww
        if oldww != self.__ww:
            self.set_window(force=True)

    def set_window(self, force=False):
        """
        sets the calcultion window for all convolvers
        """
        ww = self.window_width
        tt = self.twotheta
        if not ww or tt is None:  # not all necessary information is set up
            return
        npoints = dict()
        nset = dict()
        ttmax = tt.max()
        ttmin = tt.min()
        for h, d in self.data.items():
            ttpeak = 2 * d['ang']
            if ttpeak - ww/2 > ttmax or ttpeak + ww/2 < ttmin:
                continue
            idx = numpy.argwhere(numpy.logical_and(tt > ttpeak - ww/2,
                                                   tt < ttpeak + ww/2))
            try:
                np = int(math.ceil(len(idx) /
                                   (tt[idx[-1, 0]]-tt[idx[0, 0]]) * ww))
            except OverflowError:
                np = 1
            npoints[h] = np
            if hasattr(d['conv'], 'twotheta_window_center_deg'):
                fptt = d['conv'].twotheta_window_center_deg
                if abs(ttpeak-fptt) / ww < 0.25 and not force:
                    continue
                nset[h] = True
            else:
                nset[h] = True
            # set window in local instances
            d['conv'].set_window(twotheta_output_points=np,
                                 twotheta_window_center_deg=ttpeak,
                                 twotheta_window_fullwidth_deg=ww)
        # set multiprocessing instances
        for chunk, handler in zip(self.chunks, self.conv_handlers):
            handler.set_windows([2 * self.data[h]['ang'] for h in chunk],
                                [npoints.get(h, 0) for h in chunk],
                                [nset.get(h, False) for h in chunk], ww)

    def _send_work(self, idx):
        """
        a threaded block which watches for data and runs computation
        """
        _, qinput, qoutput = self.threads[idx]
        while self._running:
            try:
                settings, run, ttpeaks = qinput.get(True)
            except TypeError:
                break
            handler = self.conv_handlers[idx]
            handler.update_parameters(settings)
            results = handler.calc(run, ttpeaks)
            qoutput.put((idx, results))  # put results on output queue
        self._running = False

    def reflection_strength(self, tt_cutoff):
        """
        determine structure factors/reflection strength of all Bragg peaks up
        to tt_cutoff. This function also implements the March-Dollase model for
        preferred orientation in the symmetric reflection mode. Note that
        although this means the sample has anisotropic properties the various
        lines can still be merged together since at the moment no anisotropic
        crystal shape is supported.

        Parameters
        ----------
        tt_cutoff : float
            upper cutoff value of 2theta until which the reflection strength
            are calculated

        Returns
        -------
        ndarray
            numpy array with field for 'hkl' (Miller indices of the peaks), 'q'
            (q-position), and 'r' (reflection strength) of the Bragg peaks
        """
        mat = self.mat.material
        pref_or = self.settings['emission']['preferred_orientation']
        r = self.settings['emission']['preferred_orientation_factor']
        mode = self.settings['global']['geometry']
        ai = self.settings['global']['geometry_incidence_angle']
        qmax = 2 * self.k0 * sin(math.radians(tt_cutoff/2))

        # get allowed Bragg peaks
        hkl = tuple(mat.lattice.get_allowed_hkl(qmax))
        q = mat.Q(hkl)
        qnorm = numpy.linalg.norm(q, axis=1)

        # March-Dollase model for preferred orientation
        # see http://www.crl.nitech.ac.jp/ar/2013/0711_acrc_ar2013_review.pdf
        def fdsum(alpha, delta, r, N=8):
            alpha[alpha == 0] += config.EPSILON  # alpha = 0 unstable
            alpha[alpha == pi] -= config.EPSILON  # alpha = pi unstable
            r3m1 = r**3 - 1
            xi0 = -ncos(alpha-delta)/nsqrt(1+(r3m1)*ncos(alpha-delta)**2)
            xi1 = -ncos(alpha+delta)/nsqrt(1+(r3m1)*ncos(alpha+delta)**2)
            sad2 = (nsin(alpha) * nsin(delta))**2
            cad = ncos(alpha) * ncos(delta)

            def h(xi):
                return 1 / nsqrt(sad2 - (cad + xi/nsqrt(1-(r3m1)*xi**2))**2)

            def w(j, N):
                return pi/(2*N)*sin((j+0.5)*pi/N)

            def xi(j, N):
                return (xi1+xi0)/2 + (xi1-xi0)/2*cos((j+0.5)*pi/N)

            s = numpy.sum([w(j, N)*h(xi(j, N)) for j in range(N)], axis=0)
            return r**(3/2) / pi * (xi1-xi0) * s

        if numpy.all(pref_or == (0, 0, 0)) or r == 1:
            f = 1
        else:
            alpha = VecAngle(q, mat.Q(pref_or))
            if mode == 'symmetric':
                f = ((r * ncos(alpha))**2 + nsin(alpha)**2/r)**(-3/2)
            elif mode == 'capillary':
                f = fdsum(alpha, pi/2, r)
            elif mode == 'asymmetric':
                if not isinstance(ai, numbers.Number):
                    raise ValueError("'geometry_incidence_angle' must be a "
                                     "number")
                th = self.Q2Ang(qnorm, deg=False)
                f = fdsum(alpha, nabs(th-math.radians(ai)), r)
            else:
                raise ValueError("xu.simpack.PowderDiffraction: invalid "
                                 "geometry mode (%s)" % mode)

        # assemble return value
        data = numpy.zeros(len(hkl), dtype=[('q', numpy.double),
                                            ('r', numpy.double),
                                            ('hkl', numpy.ndarray)])
        data['q'] = qnorm
        data['r'] = nabs(mat.StructureFactorForQ(q, self.energy)) ** 2 * f
        data['hkl'] = hkl

        return data

    def merge_lines(self, data):
        """
        if calculation is isotropic lines at the same q-position can be merged
        to one line to reduce the calculational effort

        Parameters
        ----------
        data :  ndarray
            numpy field array with values of 'hkl' (Miller indices of the
            peaks), 'q' (q-position), and 'r' (reflection strength) as produced
            by the `reflection_strength` method

        Returns
        -------
        hkl, q, ang, r : array-like
            Miller indices, q-position, diffraction angle (Theta), and
            reflection strength of the material
        """
        data.sort(order=['q', 'hkl'])
        qpos = []
        refstrength = []
        hkl = []

        def add_lines(q, ref, chkl):
            for R, m in zip(ref, chkl):
                qpos.append(q)
                refstrength.append(R)
                hkl.append(m)

        currq = -1
        curref = []
        currhkl = []
        for r in data:
            if not numpy.isclose(r[0] - currq, 0):
                add_lines(currq, curref, currhkl)
                currq = r[0]
                curref = [r[1], ]
                currhkl = [r[2], ]
            else:
                if self.isotropic:
                    curref[-1] += r[1]
                    currhkl[-1] = r[2]
                else:
                    # merge lines which are equal according to the crystal
                    # and convolver symmetries
                    added = False
                    for i, m in enumerate(currhkl):
                        if self.mat.material.lattice.isequivalent(m, r[2]):
                            if self.fpclass.isequivalent(
                                    m, r[2],
                                    self.mat.material.lattice.crystal_system):
                                curref[i] += r[1]
                                added = True
                    if not added:
                        curref.append(r[1])
                        currhkl.append(r[2])
        # add remaining lines
        add_lines(currq, curref, currhkl)

        return (hkl,
                numpy.array(qpos, dtype=numpy.double),
                self.Q2Ang(qpos),
                numpy.array(refstrength, dtype=numpy.double))

    def correction_factor(self, ang):
        """
        calculate the correction factor for the diffracted intensities. This
        contains the polarization effects and the Lorentz factor

        Parameters
        ----------
        ang :   aray-like
            theta diffraction angles for which the correction should be
            calculated

        Returns
        -------
        f :     array-like
            array of the same shape as ang containing the correction factors
        """
        # correct data for polarization and lorentzfactor and unit cell volume
        # see L.S. Zevin : Quantitative X-Ray Diffractometry
        # page 18ff
        polarization_factor = (1 +
                               ncos(numpy.radians(2 * ang)) ** 2) / 2
        lorentz_factor = 1. / (nsin(numpy.radians(ang)) ** 2 *
                               ncos(numpy.radians(ang)))
        unitcellvol = self.mat.material.lattice.UnitCellVolume()
        return polarization_factor * lorentz_factor / unitcellvol ** 2

    def init_powder_lines(self, tt_cutoff):
        """
        calculates the powder intensity and positions up to an angle of
        tt_cutoff (deg) and stores the result in the data dictionary whose
        structure is as follows:

            The data dictionary has one entry per line with a unique identifier
            as key of the entry. The entries themself are dictionaries which
            have the following entries:

            * hkl :   (h, k, l), Miller indices of the Bragg peak
            * r :     reflection strength of the line
            * ang :   Bragg angle of the peak (theta = 2theta/2!)
            * qpos :  reciprocal space position
        """

        tmp_data = self.reflection_strength(tt_cutoff)
        hkl, qpos, ang, rs = self.merge_lines(tmp_data)
        corrfact = self.correction_factor(ang)
        rs *= corrfact
        ids = [tuple(idx) for idx in hkl]
        for i, q, a, r in zip(ids, qpos, ang, rs):
            active = True if r/rs.max() > config.EPSILON else False
            self.data[i] = {'qpos': q, 'ang': a, 'r': r,
                            'active': active}

    def update_powder_lines(self, tt_cutoff):
        """
        calculates the powder intensity and positions up to an angle of
        tt_cutoff (deg) and updates the values in:

            * ids:  list of unique identifiers of the powder line
            * data: array with intensities
            * ang:  bragg angles of the peaks (theta=2theta/2!)
            * qpos: reciprocal space position of intensities
        """
        tmp_data = self.reflection_strength(tt_cutoff)
        hkl, qpos, ang, rs = self.merge_lines(tmp_data)
        corrfact = self.correction_factor(ang)
        rs *= corrfact
        ids = [tuple(idx) for idx in hkl]
        rsmax = rs.max()
        for h, q, a, r in zip(ids, qpos, ang, rs):
            active = True if r/rsmax > config.EPSILON else False
            if h in self.data:
                self.data[h]['qpos'] = q
                self.data[h]['ang'] = a
                self.data[h]['r'] = r
                self.data[h]['active'] = active
            else:
                # new peak needs a new convolver
                fp = self._init_fpprofile(self.fpclass)
                for k, v in self.settings.items():
                    if k == 'classoptions':
                        continue
                    fp.set_parameters(convolver=k, **v)
                self.data[h] = {'qpos': q, 'ang': a, 'r': r,
                                'conv': fp, 'active': active}
                if self._enable_sim:
                    self.conv_handlers[self.next_proc].add_convolver(fp)
                    self.chunks[self.next_proc].append(h)
                    self.next_proc = (self.next_proc + 1) % self.nproc
        for h in self.data:
            if h not in ids:
                # make entry inactive
                self.data[h]['active'] = False

    def Convolve(self, twotheta, window_width='config', mode='multi'):
        """
        convolute the powder lines with the resolution function and map them
        onto the twotheta positions. This calculates the powder pattern
        excluding any background contribution

        Parameters
        ----------
        twotheta :      array-like
            two theta values at which the powder pattern should be calculated.
        window_width :  float, optional
            width of the calculation window of a single peak
        mode :          {'multi, 'local'}, optional
            multiprocessing mode, either 'multi' to use multiple processes or
            'local' to restrict the calculation to a single process

        Note:
            Bragg peaks are only included up to tt_cutoff set in the class
            constructor!

        Returns
        -------
         output intensity values for the twotheta values given in the input
        """
        if self._enable_sim:
            t_start = time.time()

            out = numpy.zeros_like(twotheta)
            tt = self.twotheta = twotheta
            self.window_width = window_width
            ww = self.window_width
            ttmax = tt.max()
            ttmin = tt.min()

            # check if twotheta range extends above tt_cutoff
            if ttmax > self._tt_cutoff:
                warnings.warn('twotheta range is larger then tt_cutoff. '
                              'Possibly Bragg peaks in the convolution range '
                              'are not considered!')

            if mode == 'local':
                for h, d in self.data.items():
                    if not d['active']:
                        continue
                    ttpeak = 2 * d['ang']
                    # check if peak is in data range to be calculated
                    if ttpeak - ww/2 > ttmax or ttpeak + ww/2 < ttmin:
                        continue
                    idx = numpy.argwhere(numpy.logical_and(tt > ttpeak - ww/2,
                                                           tt < ttpeak + ww/2))
                    d['conv'].set_parameters(twotheta0_deg=ttpeak)
                    result = d['conv'].compute_line_profile()
                    out[idx] += numpy.interp(tt[idx], result.twotheta_deg,
                                             result.peak*d['r'], left=0,
                                             right=0)
            else:
                # prepare multiprocess calculation
                for idx, chunk in enumerate(self.chunks):
                    run = []
                    ttpeaks = []
                    for h in chunk:
                        ttpeak = 2 * self.data[h]['ang']
                        ttpeaks.append(ttpeak)
                        if (ttpeak - ww/2 > ttmax or
                                ttpeak + ww/2 < ttmin):
                            run.append(False)
                        else:
                            run.append(True)
                        if not self.data[h]['active']:
                            run[-1] = False
                    # start calculation in other processes
                    self.threads[idx][1].put((self.settings, run, ttpeaks))
                gotit = set(range(self.nproc))
                while gotit:
                    # receive ready calculations
                    idx, res = self.output_queue.get(True)
                    chunk = self.chunks[idx]
                    for h, r in zip(chunk, res):
                        if r is None:
                            continue
                        ttpeak = 2 * self.data[h]['ang']
                        mask = numpy.argwhere(
                            numpy.logical_and(tt > ttpeak - ww/2,
                                              tt < ttpeak + ww/2))

                        out[mask] += numpy.interp(tt[mask], r.twotheta_deg,
                                                  r.peak*self.data[h]['r'],
                                                  left=0, right=0)
                    gotit.discard(idx)  # got that result, don't expect more

            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.Powder.Convolute: exec time=", time.time() - t_start)
            return out
        print("XU.Powder: not initialized for calculation -> exiting!")
        return None

    def Calculate(self, twotheta, **kwargs):
        """
        calculate the powder diffraction pattern including convolution with the
        resolution function and map them onto the twotheta positions. This also
        performs the calculation of the peak intensities from the internal
        material object

        Parameters
        ----------
        twotheta :      array-like
            two theta values at which the powder pattern should be calculated.
        kwargs :        dict
            additional keyword arguments are passed to the Convolve function

        Returns
        -------
        array-like
            output intensity values for the twotheta values given in the input

        Notes
        -----
        Bragg peaks are only included up to tt_cutoff set in the class
        constructor!
        """
        if self._enable_sim:
            self.set_sample_parameters()
            self.update_powder_lines(self._tt_cutoff)
            self.set_window()
            return self.Convolve(twotheta, **kwargs)
        print("XU.Powder: not initialized for calculation -> exiting!")
        return None

    def __str__(self):
        """
        Prints out available information about the material and reflections
        """
        ostr = "Powder diffraction object\n"
        ostr += "-------------------------\n"
        ostr += self.mat.__repr__() + "\n"
        ostr += "Lattice:\n" + self.mat.material.lattice.__str__()
        rmax = 0
        for d in self.data.values():
            if d['r'] > rmax:
                rmax = d['r']
        ostr += "\nReflections:\n"
        ostr += "------------\n"
        ostr += ("      h k l     |    tth    |    |Q|    |"
                 "Int     |   Int (%)\n")
        ostr += ("   ------------------------------------"
                 "---------------------------\n")
        for h, d in sorted(self.data.items(), key=lambda t: t[1]['ang']):
            if d['active']:
                ostr += ("%15s   %8.4f   %8.3f   %10.2f  %10.2f\n"
                         % (h.__str__(), 2 * d['ang'],
                            d['qpos'], d['r'], d['r'] / rmax * 100.))

        if self._enable_sim:
            ostr += "Settings: " + str(self.settings)
        return ostr
