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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2015-2016 Marcus H. Mendenhall <marcus.mendenhall@nist.gov>

# This file was derived from http://dx.doi.org/10.6028/jres.120.014.c

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

# dependency on Experiment baseclass is very loose now and the
# energy/wavelength handling is not working as in the other experiment classes

from __future__ import absolute_import, print_function

import math
import os
import sys
import warnings
from math import cos, pi, sin, sqrt, tan

import numpy
from numpy import arcsin as nasin
from numpy import cos as ncos
from numpy import asarray
from numpy.linalg import norm
from scipy.special import sici  # for the sine and cosine integral

# package internal imports
from .smaterials import Powder
from .. import math as xumath
from .. import config, materials
from ..experiment import PowderExperiment

# figure out which FFT package we have, and import it
try:
    from pyfftw.interfaces import numpy_fft, cache
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
ft_factors = numpy.array(ft_factors, numpy.int)

# used for debugging moments from FP_profile.axial_helper().
moment_list = []
# if this is *True*, compute and save moment errors
collect_moment_errors = False


class profile_data(object):
    """
    a skeleton class which makes a combined dict and namespace interface for
    easy pickling and data passing
    """

    def __init__(self, **kwargs):
        """
        initialize the class

        Parameters
        ----------
         kwargs keyword=value list to pre-populate the class
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
         kwargs keyword=value pairs
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

    Noteable class parameters:
     max_history_length: the number of histories to cache (default=5); can be
                         overridden if memory is an issue.
     length_scale_m: length_scale_m sets scaling for nice printing of
                     parameters.  if the units are in mm everywhere, set it to
                     0.001, e.g.  convolvers which implement their own str_xxx
                     method may use this to format their results, especially if
                     'natural' units are not meters.  Typical is wavelengths
                     and lattices in nm or angstroms, for example.

    """
    max_history_length = 5  # max number of histories for each convolver
    length_scale_m = 1.0

    def __init__(self, anglemode,
                 output_gaussian_smoother_bins_sigma=1.0,
                 oversampling=10):
        """
        initialize the instance

        Parameters
        ----------
         anglemode: 'd' if setup will be in terms of a d-spacing,
                    otherwise 'twotheta' if setup will be at a fixed 2theta
                    value.
         output_gaussian_smoother_bins_sigma: the number of bins for
                                              post-smoothing of data. 1.0 is
                                              good. *None* means no final
                                              smoothing step.
         oversampling: the number of bins internally which will get computed
                       for each bin the the final result.
        """
        if anglemode not in ("d", "twotheta"):
            raise Exception(
                "invalid angle mode %s, must be 'd' or 'twotheta'" % anglemode)
        # set to either 'd' for d-spacing based position, or 'twotheta' for
        # angle-based position
        self.anglemode = anglemode
        # sigma, in units of bins, for the final smoother.
        self.output_gaussian_smoother_bins_sigma = output_gaussian_smoother_bins_sigma
        # the number of internal bins computed for each bin in the final
        # output. 5-10 is usually plenty.
        self.oversampling = oversampling
        # List of our convolvers, found by introspection of names beginning
        # with 'conv_'
        self.convolvers = convolvers = [
            x for x in dir(self) if x.startswith("conv_")]
        # A dictionary which will store all the parameters local to each
        # convolution
        self.param_dicts = dict([(c, {}) for c in convolvers])
        # add global parameters, associated with no specific convolver
        # A dictionary of bound functions to call to compute convolutions
        self.convolver_funcs = dict(
            [(x, getattr(self, x)) for x in convolvers])
        # If *True*, print cache hit information
        self.debug_cache = False

    def get_function_name(self):
        """
        return the name of the function that called this. Useful for convolvers
        to identify themselves

        Returns
        ----------
        name of calling function
        """
        return sys._getframe(1).f_code.co_name

    def add_buffer(self, b):
        """
        add a numpy array to the list of objects that can be thrown away on
        pickling.

        Parameters
        ----------
         b: the buffer to add to the list

        Returns
        -------
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
         twotheta_window_center_deg: the center position of the middle bin of
                                     the window, in degrees
         twotheta_window_fullwidth_deg: the full width of the window, in
                                        degrees
         twotheta_output_points: the number of bins in the final output
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
        # keep a record of things we don't keep when pickled
        x = self._clean_on_pickle = set()
        b = self.add_buffer  # shortcut

        # a real-format scratch buffer
        self._rb1 = b(numpy.zeros(nn, numpy.float))
        # a real-format scratch buffer
        self._rb2 = b(numpy.zeros(nn, numpy.float))
        # a real-format scratch buffer
        self._rb3 = b(numpy.zeros(nn, numpy.float))
        # a complex-format scratch buffer
        self._cb1 = b(numpy.zeros(nn, numpy.complex))
        # a scratch buffer used by the axial helper
        self._f0buf = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, numpy.float))
        # a scratch buffer used for axial divergence
        self._epsb2 = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, numpy.float))
        # the I2+ buffer
        self._I2p = b(numpy.zeros(self.oversampling *
                                  twotheta_output_points, numpy.float))
        # the I2- buffer
        self._I2m = b(numpy.zeros(self.oversampling *
                                  twotheta_output_points, numpy.float))
        # another buffer used for axial divergence
        self._axial = b(numpy.zeros(self.oversampling *
                                    twotheta_output_points, numpy.float))
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
        self.convolution_history = dict([(x, []) for x in self.convolvers])

        # A dictionary of Lorentz widths, used for de-periodizing the final
        # result.
        self.lor_widths = {}

    def get_good_bin_count(self, count):
        """
        find a bin count close to what we need, which works well for Fourier
        transforms.

        Parameters
        ----------
         count: a number of bins.

        Returns
        -------
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
         twotheta_window_center_deg: exact position of center bin, in degrees
         twotheta_approx_window_fullwidth_deg: approximate desired width
         twotheta_exact_bin_spacing_deg: the exact bin spacing to use
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
         convolver: the name of the convolver.  name 'global', e.g., attaches
                    to function 'conv_global'
         kwargs: keyword-value pairs to update the convolvers dictionary.
        """
        self.param_dicts["conv_" + convolver].update(kwargs)

    def get_conv(self, name, key, format=numpy.float):
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
         name: the name of the convolver to seek
         key: any hashable object which identifies the parameters for the
              computation
         format: the type of the array to create, if one is not found.

        Returns
        -------
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
        string of formatted information
        """
        keys = list(self.convolver_funcs.keys())
        keys.sort()  # always return info in the same order
        # global is always first, anyways!
        keys.insert(0, keys.pop(keys.index('conv_global')))
        strings = ["", "***convolver id 0x%08x:" % id(self)]
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
        *None*, always
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
         outerbound: the edge of the function farthest from the singularity,
                     referenced to epsvals
         innerbound: the edge closest to the singularity, referenced to
                     epsvals
         epsvals: the array of two-theta values or offsets
         destination: an array into which final results are summed.  modified
                      in place!
         peakpos: the position of the singularity, referenced to epsvals.
         y0: the constant offset
         k: the scale factor

        Returns
        -------
         (*lower_index*, *upper_index*) python style bounds
         for region of *destination* which has been modified.
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
        # exactintegral=max(0,exactintegral) # can never be < 0, beta out of
        # range.
        exactintegral *= 1 / dx  # normalize so sum is right
        # compute the exact centroid we need for this
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
        if abs(outerbound - innerbound) < 2 * dx:
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
         Lx: length of the xray filament
         Ls: length of the sample
         Lr: length of the receiver slit
         R: diffractometer length, assumed symmetrical
         twotheta: angle, in radians, of the center of the computation
         beta: offset angle
         epsvals: array of offsets from center of computation, in radians

        Returns
        -------
         (*epsvals*, *idxmin*, *idxmax*, *I2p*, *I2m*).
         *idxmin* and *idxmax* are the full python-style bounds of the non-zero
         region of *I2p* and *I2m*.
         *I2p* and *I2m* are I2+ and I2- from the paper, the contributions to
         the intensity.
        """
        beta1 = (Ls - Lx) / (2 * R)  # Ch&Co after eq. 15abcd
        beta2 = (Ls + Lx) / (2 * R)  # Ch&Co after eq. 15abcd, corrected by KM

        eps0 = beta * beta * tan(twotheta) / 2  # after eq. 26 in Ch&Co

        if -beta2 <= beta < beta1:
            z0p = Lx / 2 + beta * R * (1 + 1 / cos(twotheta))
        elif beta1 <= beta <= beta2:
            z0p = Ls / 2 + beta * R / cos(twotheta)

        if -beta2 <= beta <= -beta1:
            z0m = -Ls / 2 + beta * R / cos(twotheta)
        elif -beta1 < beta <= beta2:
            z0m = -Lx / 2 + beta * R * (1 + 1 / cos(twotheta))

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
            if z0p <= Lr/2 and z0m > -Lr/2:  # beam entirely within slit
                rng = 1
                ea = eps1p
                eb = eps2p
                ec = eps1m
                ed = eps2m
            elif (z0p > Lr/2 and z0m < Lr/2) or (z0m < -Lr/2 and z0p > -Lr/2):
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
            if z0m < -Lr/2 and z0p > Lr/2:
                rng = 1
                ea = eps1m
                eb = eps2p
                ec = eps1p
                ed = eps2m
            # one edge of beam within slit
            elif (-Lr/2 < z0m < Lr/2 and z0p > Lr/2) or \
                    (-Lr/2 < z0p < Lr/2 and z0m < -Lr/2):
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
         Lx: length of the xray filament
         Ls: length of the sample
         Lr: length of the receiver slit
         R: the (assumed symmetrical) diffractometer radius
         twotheta: angle, in radians, of the center of the computation
         epsvals: array of offsets from center of computation, in radians
         sollerIdeg: the full-width (both sides) cutoff angle of the incident
                     Soller slit
         sollerDdeg: the full-width (both sides) cutoff angle of the detector
                     Soller slit
         nsteps: the number of subdivisions for the integral
         axDiv: not used

        Returns
        -------
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

            eps, idxmin, idxmax, I2p, I2m = self.full_axdiv_I2(
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
         the transform buffer, or *None* if this is being ignored
        """
        me = self.get_function_name()  # the name of this convolver,as a string
        if self.param_dicts[me].get("axDiv", None) is None:
            return None
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        kwargs.update(self.param_dicts["conv_global"])
        if "equatorial_divergence_deg" in kwargs:
            del kwargs["equatorial_divergence_deg"]  # not used

        flag, axfn = self.get_conv(me, kwargs, numpy.complex)
        if flag:
            return axfn  # already up to date if first return is True

        # no axial divergence, transform of delta fn
        if kwargs["axDiv"] != "full":
            axfn[:] = 1
            return axfn
        else:
            xx = type("data", (), kwargs)
            axbuf = self.full_axdiv_I3(
                nsteps=xx.n_integral_points,
                epsvals=self.epsilon,
                Lx=xx.slit_length_source,
                Lr=xx.slit_length_target,
                Ls=xx.length_sample,
                sollerIdeg=xx.angI_deg,
                sollerDdeg=xx.angD_deg,
                R=xx.diffractometer_radius,
                twotheta=xx.twotheta0
            )
        axfn[:] = best_rfft(axbuf)

        return axfn

    def conv_tube_tails(self):
        """
        compute the Fourier transform of the rectangular tube tails function

        Returns
        -------
         the transform buffer, or *None* if this is being ignored
        """
        me = self.get_function_name()  # the name of this convolver,as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None  # no convolver
        # we also need the diffractometer radius from the global space
        kwargs["diffractometer_radius"] = self.param_dicts[
            "conv_global"]["diffractometer_radius"]
        flag, tailfn = self.get_conv(me, kwargs, numpy.complex)
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
        xx = type("data", (), kwargs)  # allow dotted notation

        tail_eps = (xx.tail_right - xx.tail_left) / xx.diffractometer_radius
        main_eps = xx.main_width / xx.diffractometer_radius
        tail_center = (xx.tail_right + xx.tail_left) / \
            xx.diffractometer_radius / 2.0
        tail_area = xx.tail_intens * \
            (xx.tail_right - xx.tail_left) / xx.main_width

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
         name: the name of the convolver cache buffer to update
         width: the width in 2-theta space of the tophat

        Returns
        -------
         the updated convolver buffer, or *None* if the width was *None*
        """
        if width is None:
            return  # no convolver
        flag, conv = self.get_conv(name, width, numpy.float)
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
         the formatted information
        """
        dd = self.param_dicts["conv_emission"]
        if not dd:
            return "No emission spectrum"
        dd.setdefault("crystallite_size_lor", 1e10)
        dd.setdefault("crystallite_size_gauss", 1e10)
        dd.setdefault("strain_lor", 0)
        dd.setdefault("strain_gauss", 0)
        xx = type("data", (), dd)
        spect = numpy.array((
            xx.emiss_wavelengths, xx.emiss_intensities,
            xx.emiss_lor_widths, xx.emiss_gauss_widths))
        # convert to Angstroms, like Topas
        spect[0] *= 1e10 * self.length_scale_m
        spect[2] *= 1e13 * self.length_scale_m  # milli-Angstroms
        spect[3] *= 1e13 * self.length_scale_m  # milli-Angstroms
        nm = 1e9 * self.length_scale_m
        items = ["emission and broadening:"]
        items.append("spectrum=\n" + str(spect.transpose()))
        items.append("crystallite_size_lor (nm): %.5g" %
                     (xx.crystallite_size_lor * nm))
        items.append("crystallite_size_gauss (nm): %.5g" %
                     (xx.crystallite_size_gauss * nm))
        items.append("strain_lor: %.5g" % xx.strain_lor)
        items.append("strain_gauss: %.5g" % xx.strain_lor)
        return '\n'.join(items)

    def conv_emission(self):
        """
        compute the emission spectrum and (for convenience) the particle size
        widths

        Returns
        -------
         the convolver for the emission and particle sizes

        Note: the particle size and strain stuff here is just to be consistent
              with *Topas* and to be vaguely efficient about the computation,
              since all of these have the same general shape.
        """
        me = self.get_function_name()  # the name of this convolver,as a string
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

        flag, emiss = self.get_conv(me, key, numpy.complex)
        if flag:
            return emiss  # already up to date

        xx = type("data", (), kwargs)  # make it dot-notation accessible

        epsilon0s = 2 * nasin(xx.emiss_wavelengths/(2.0*xx.d)) - xx.twotheta0
        theta = xx.twotheta0 / 2
        # Emission profile FWHM + crystallite broadening (scale factors are
        # Topas choice!) (Lorentzian)
        widths = (
            (xx.emiss_lor_widths / xx.emiss_wavelengths) * tan(theta) +
            xx.strain_lor * tan(theta) +
            (xx.emiss_wavelengths / (2 * xx.crystallite_size_lor * cos(theta)))
        )
        # save weighted average width for future reference in periodicity fixer
        self.lor_widths[me] = sum(
            widths * xx.emiss_intensities) / sum(xx.emiss_intensities)
        # gaussian bits add in quadrature
        gfwhm2s = (
            ((2*xx.emiss_gauss_widths/xx.emiss_wavelengths) * tan(theta))**2 +
            (xx.strain_gauss * tan(theta))**2 +
            (xx.emiss_wavelengths / (xx.crystallite_size_gauss*cos(theta)))**2
        )

        # note that the Fourier transform of a lorentzian with FWHM 2a
        # is exp(-abs(a omega))
        # now, the line profiles in Fourier space have to have phases
        # carefully handled to put the lines in the right places.
        # note that the transform of f(x+dx)=exp(i omega dx) f~(x)
        omega_vals = self.omega_vals
        for wid, gfwhm2, eps, intens in zip(widths, gfwhm2s, epsilon0s,
                                            xx.emiss_intensities):
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
         the convolver
        """
        me = self.get_function_name()  # the name of this convolver,as a string
        equatorial_divergence_deg = self.param_dicts[
            "conv_global"].get("equatorial_divergence_deg", None)
        if not equatorial_divergence_deg:
            return None
        twotheta0 = self.param_dicts["conv_global"]["twotheta0"]
        key = (twotheta0, equatorial_divergence_deg)
        flag, conv = self.get_conv(me, key, numpy.complex)
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
        idx0, idx1 = self.axial_helper(destination=eqdiv,
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
         the convolver
        """
        me = self.get_function_name()  # the name of this convolver,as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None
        kwargs["twotheta0"] = self.param_dicts["conv_global"]["twotheta0"]
        kwargs["diffractometer_radius"] = self.param_dicts[
            "conv_global"]["diffractometer_radius"]

        flag, conv = self.get_conv(me, kwargs, numpy.complex)
        if flag:
            return conv  # already up to date
        xx = type("data", (), kwargs)  # make it dot-notation accessible

        # absorption, from Cheary, Coelho & Cline 2004 NIST eq. 12,
        # EXCEPT delta = 1/(2*mu*R) instead of 2/(mu*R)
        # from Mathematica, unnormalized transform is
        # (1-exp(epsmin*(i w + 1/delta)))/(i w + 1/delta)
        delta = sin(xx.twotheta0) / (2 * xx.absorption_coefficient *
                                     xx.diffractometer_radius)
        # arg=(1/delta)+complex(0,-1)*omega_vals
        cb = self._cb1
        cb.imag = self.omega_vals
        cb.imag *= -1
        cb.real = 1 / delta
        numpy.reciprocal(cb, conv)  # limit for thick samples=1/(delta*arg)
        conv *= 1.0 / delta  # normalize
        # rest of transform of function with cutoff
        if kwargs.get("sample_thickness", None) is not None:
            epsmin = -2.0 * xx.sample_thickness * \
                cos(xx.twotheta0 / 2.0) / xx.diffractometer_radius
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
         the convolver
        """
        me = self.get_function_name()  # the name of this convolver,as a string
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
                                   numpy.complex)
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
         the convolver
        """
        me = self.get_function_name()  # the name of this convolver,as a string
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
         the convolver
        """
        # omega offset defocussing from Cheary, Coelho & Cline 2004 eq. 15
        # expressed in terms of a Si PSD looking at channels with vertical
        # offset from the center between psd_window_lower_offset and
        # psd_window_upper_offset do this last, because we may ultimately take
        # a list of bounds, and return a list of convolutions, for efficiency
        me = self.get_function_name()  # the name of this convolver,as a string
        kwargs = {}
        kwargs.update(self.param_dicts[me])  # get all of our parameters
        if not kwargs:
            return None
        kwargs.update(self.param_dicts["conv_global"])

        flag, conv = self.get_conv(me, kwargs, numpy.float)
        if flag:
            return conv  # already up to date

        xx = type("data", (), kwargs)

        if not xx.equatorial_divergence_deg or not xx.si_psd_window_bounds:
            # if either of these is zero or None, convolution is trivial
            conv[:] = 1
            return conv

        psd_lower_window_pos, psd_upper_window_pos = xx.si_psd_window_bounds
        dthl = psd_lower_window_pos / xx.diffractometer_radius
        dthu = psd_upper_window_pos / xx.diffractometer_radius
        alpha = math.radians(xx.equatorial_divergence_deg)
        argscale = alpha / (2.0 * tan(xx.twotheta0 / 2))
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
         the convolver
        """
        # create a smoother for output result, independent of real physics, if
        # wanted
        me = self.get_function_name()  # the name of this convolver,as a string
        if not self.output_gaussian_smoother_bins_sigma:
            return  # no smoothing
        flag, buf = self.get_conv(me, self.output_gaussian_smoother_bins_sigma,
                                  format=numpy.float)
        if flag:
            return buf  # already computed
        buf[:] = self.omega_vals
        buf *= (self.output_gaussian_smoother_bins_sigma * (
            self.twothetasamples[1] - self.twothetasamples[0]))
        buf *= buf
        buf *= -0.5
        numpy.exp(buf, buf)
        return buf

    def compute_line_profile(self, convolver_names=None,
                             compute_derivative=False):
        """
        execute all the convolutions; if convolver_names is None, use
        everything we have, otherwise, use named convolutions.

        Parameters
        ----------
         convolver_names: a list of convolvers to select. If *None*, use all
                          found convolvers.
         compute_derivative: if *True*, also return d/dx(function) for peak
                             position fitting

        Returns
        -------
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

        return result

    def self_clean(self):
        """
        do some cleanup to make us more compact;
        Instance can no longer be used after doing this, but can be pickled.
        """
        clean = self._clean_on_pickle
        pd = dict()
        pd.update(self.__dict__)
        for thing in pd.keys():
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
        dictionary of sufficient information to reanimate instance.
        """
        #  do some cleanup on state before we get pickled
        # (even if main class not cleaned)
        clean = self._clean_on_pickle
        pd = dict()
        pd.update(self.__dict__)
        for thing in pd.keys():
            x = getattr(self, thing)
            if id(x) in clean:
                del pd[thing]
        # delete extra attributes cautiously, in case we have alread been
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
         self: an empty class instance
         setdict: dictionary from FP_profile.__getstate__()
        """
        self.__init__(anglemode=setdict["anglemode"],
                      output_gaussian_smoother_bins_sigma=setdict[
                          "output_gaussian_smoother_bins_sigma"],
                      oversampling=setdict["oversampling"]
                      )
        for k, v in setdict.items():
            setattr(self, k, v)
        self.set_window(
            twotheta_window_center_deg=self.twotheta_window_center_deg,
            twotheta_window_fullwidth_deg=self.twotheta_window_fullwidth_deg,
            twotheta_output_points=self.twotheta_output_points
        )
        # override clearing of this by set_window
        self.lor_widths = setdict["lor_widths"]


class PowderDiffraction(PowderExperiment):

    """
    Experimental class for powder diffraction. This class calculates the
    structure factors of powder diffraction lines and uses an instance of
    FP_profile to perform the convolution with experimental resolution function
    calculated by the fundamental parameters approach.
    """

    def __init__(self, mat, **kwargs):
        """
        the class is initialized with a xrayutilities.materials.Crystal
        instance and calculates the powder intensity and peak positions of the
        Crystal up to an angle of tt_cutoff. Results are stored in

            data .... array with intensities
            ang ..... Bragg angles of the peaks (Theta!)
            qpos .... reciprocal space position of intensities

        Parameters
        ----------
         mat:        xrayutilities.material.Crystal or
                     xrayutilities.simpack.Powder instance specifying the
                     material for the powder calculation
         kwargs:     optional keyword arguments
                     same as for the Experiment base class +
          tt_cutoff: Powder peaks are calculated up to an scattering angle of
                     tt_cutoff (deg)
          fpclass:   FP_profile derived class with possible convolver mixins.
                     (default: FP_profile)
          fpsettings: settings dictionaries for the convolvers. Default
                      settings are loaded from the config file.
        """
        if isinstance(mat, materials.Crystal):
            self.mat = Powder(mat, 1)
        elif isinstance(mat, Powder):
            self.mat = mat
        else:
            raise TypeError("mat must be an instance of class "
                            "xrayutilities.materials.Crystal or "
                            "xrayutilities.simpack.Powder")

        self._tt_cutoff = kwargs.pop('tt_cutoff', 180)
        fpclass = kwargs.pop('fpclass', FP_profile)
        settings = kwargs.pop('fpsettings', {})
        PowderExperiment.__init__(self, **kwargs)

        self.fp_profile = self._init_fpprofile(fpclass, settings)
        self.set_sample_parameters()

        # number of significant digits, needed to identify equal floats
        self.digits = 5
        self.qpos = None

        self.PowderIntensity(self._tt_cutoff)

    def set_sample_parameters(self):
        """
        load sample parameters from the Powder class and use them in fp_profile
        """
        samplesettings = {}
        for prop, default in zip(('crystallite_size_lor',
                                  'crystallite_size_gauss',
                                  'strain_lor', 'strain_gauss'),
                                 (1e10, 1e10, 0, 0)):
            samplesettings[prop] = getattr(self.mat, prop, default)

        self.fp_profile.set_parameters(convolver='emission', **samplesettings)

    def _init_fpprofile(self, fpclass, settings={}):
        """
        configure the default parameters of the FP_profile class

        Parameters
        ----------
         fpclass: class with possible mixins which implement more convolvers
         settings: user-supplied settings, default values are loaded from the
                   config file
        """

        params = dict()
        for k in config.POWDER:
            params[k] = dict()
            params[k].update(config.POWDER[k])
            if k in settings:
                params[k].update(settings[k])
        for k in settings:
            if k not in config.POWDER:
                params[k] = dict().update(settings[k])

        params['classoptions'].pop('window_width')
        wl = params['emission']['emiss_wavelengths'][0]
        params['global']['dominant_wavelength'] = wl
        # set wavelength in base class
        self._set_wavelength(wl*1e10)

        p = fpclass(**params['classoptions'])
        p.debug_cache = False
        # set parameters for each convolver
        params.pop('classoptions')
        for k in params:
            p.set_parameters(convolver=k, **params[k])

        self.profile_params = params

        return p

    def PowderIntensity(self, tt_cutoff=180):
        """
        Calculates the powder intensity and positions up to an angle of
        tt_cutoff (deg) and stores the result in:

            data .... array with intensities
            ang ..... Bragg angles of the peaks (Theta!)
            qpos .... reciprocal space position of intensities
        """
        mat = self.mat.material
        # calculate maximal Bragg indices
        hmax = int(numpy.ceil(norm(mat.lattice.a1) * self.k0 / numpy.pi *
                   numpy.sin(numpy.radians(tt_cutoff / 2.))))
        hmin = -hmax
        kmax = int(numpy.ceil(norm(mat.lattice.a2) * self.k0 / numpy.pi *
                   numpy.sin(numpy.radians(tt_cutoff / 2.))))
        kmin = -kmax
        lmax = int(numpy.ceil(norm(mat.lattice.a3) * self.k0 / numpy.pi *
                   numpy.sin(numpy.radians(tt_cutoff / 2.))))
        lmin = -lmax

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.Powder.PowderIntensity: tt_cutoff; (hmax,kmax,lmax): "
                  "%6.2f (%d,%d,%d)" % (tt_cutoff, hmax, kmax, lmax))

        qlist = []
        qabslist = []
        hkllist = []
        qmax = numpy.sqrt(2) * self.k0 * numpy.sqrt(
            1 - numpy.cos(numpy.radians(tt_cutoff)))
        # calculate structure factor for each reflex
        for h in range(hmin, hmax + 1):
            for k in range(kmin, kmax + 1):
                for l in range(lmin, lmax + 1):
                    q = mat.Q(h, k, l)
                    if norm(q) < qmax:
                        qlist.append(q)
                        hkllist.append([h, k, l])
                        qabslist.append(numpy.round(norm(q), self.digits))

        qabs = numpy.array(qabslist, dtype=numpy.double)
        s = mat.StructureFactorForQ(qlist, self.energy)
        r = numpy.absolute(s) ** 2

        _tmp_data = numpy.zeros(r.size, dtype=[('q', numpy.double),
                                               ('r', numpy.double),
                                               ('hkl', list)])
        _tmp_data['q'] = qabs
        _tmp_data['r'] = r
        _tmp_data['hkl'] = hkllist
        # sort the list and compress equal entries
        _tmp_data.sort(order='q')

        self.qpos = [0]
        self.data = [0]
        self.hkl = [[0, 0, 0]]
        for r in _tmp_data:
            if r[0] == self.qpos[-1]:
                self.data[-1] += r[1]
            elif numpy.round(r[1], self.digits) != 0.:
                self.qpos.append(r[0])
                self.data.append(r[1])
                self.hkl.append(r[2])

        # cat first element to get rid of q = [0,0,0] divergence
        self.qpos = numpy.array(self.qpos[1:], dtype=numpy.double)
        self.ang = self.Q2Ang(self.qpos)
        self.data = numpy.array(self.data[1:], dtype=numpy.double)
        self.hkl = self.hkl[1:]

        # correct data for polarization and lorentzfactor and unit cell volume
        # see L.S. Zevin : Quantitative X-Ray Diffractometry
        # page 18ff
        polarization_factor = (1 +
                               numpy.cos(numpy.radians(2 * self.ang)) ** 2) / 2
        lorentz_factor = 1. / (numpy.sin(numpy.radians(self.ang)) ** 2 *
                               numpy.cos(numpy.radians(self.ang)))
        unitcellvol = mat.lattice.UnitCellVolume()
        self.data = (self.data * polarization_factor *
                     lorentz_factor / unitcellvol ** 2)

    def Convolve(self, twotheta, window_width='config'):
        """
        convolute the powder lines with the resolution function and map them
        onto the twotheta positions. This calculates the powder pattern
        excluding any background contribution

        Parameters
        ----------
         twotheta:  two theta values at which the powder pattern should be
                    calculated.
                    Note: Bragg peaks are only included up to tt_cutoff set in
                          the class constructor!
         window_width: width of the calculation window of a single peak

        Returns
        -------
         output intensity values for the twotheta values given in the input
        """
        import time
        t_start = time.time()
        p = self.fp_profile

        ttmax = numpy.max(twotheta)
        ttmin = numpy.min(twotheta)
        # check if twotheta range extends above tt_cutoff
        if ttmax > self._tt_cutoff:
            warnings.warn('twotheta range is larger then tt_cutoff. Possibly '
                          'Bragg peaks in the convolution range are not '
                          'considered!')

        outint = numpy.zeros_like(twotheta)
        if window_width == 'config':
            ww = config.POWDER['classoptions']['window_width']
        else:
            ww = window_width
        tt = twotheta
        for twotheta_x, sfact in zip(2*self.ang, self.data):
            # check if peak is in data range to be calculated
            if twotheta_x - ww/2 > ttmax or twotheta_x + ww/2 < ttmin:
                continue
            idx = numpy.argwhere(numpy.logical_and(tt > 20, tt < 40))
            npoints = int(math.ceil(len(idx) / (tt[idx[-1]]-tt[idx[0]]) * ww))
            # put the compute window in the right place and clear all histories
            p.set_window(twotheta_output_points=npoints,
                         twotheta_window_center_deg=twotheta_x,
                         twotheta_window_fullwidth_deg=ww)
            # set parameters which are shared by many things
            p.set_parameters(twotheta0_deg=twotheta_x)
            result = p.compute_line_profile()

            outint += numpy.interp(tt, result.twotheta_deg,
                                   result.peak*sfact, left=0, right=0)
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.Powder.Convolute: exec time=", time.time() - t_start)
        return outint

    def Calculate(self, twotheta, **kwargs):
        """
        calculate the powder diffraction pattern including convolution with the
        resolution function and map them onto the twotheta positions. This also
        performs the calculation of the peak intensities from the internal
        material object

        Parameters
        ----------
         twotheta:  two theta values at which the powder pattern should be
                    calculated.
                    Note: Bragg peaks are only included up to tt_cutoff set in
                          the class constructor!
         **kwargs: additional keyword arguments are passed to the Convolve
                   function

        Returns
        -------
         output intensity values for the twotheta values given in the input
        """
        self.set_sample_parameters()
        self.PowderIntensity()
        return self.Convolve(twotheta, **kwargs)

    def __str__(self):
        """
        Prints out available information about the material and reflections
        """
        ostr = "\nPowder diffraction object \n"
        ostr += "-------------------------\n"
        ostr += self.mat.__repr__() + "\n"
        ostr += "Lattice:\n" + self.mat.material.lattice.__str__()
        if self.qpos is not None:
            max = self.data.max()
            ostr += "\nReflections: \n"
            ostr += "--------------\n"
            ostr += ("      h k l     |    tth    |    |Q|    |"
                     "Int     |   Int (%)\n")
            ostr += ("   ------------------------------------"
                     "---------------------------\n")
            for i in range(self.qpos.size):
                ostr += ("%15s   %8.4f   %8.3f   %10.2f  %10.2f\n"
                         % (self.hkl[i].__str__(),
                            2 * self.ang[i],
                            self.qpos[i],
                            self.data[i],
                            self.data[i] / max * 100.))
        try:
            ostr += str(self.fp_profile)
        except:
            pass

        return ostr
