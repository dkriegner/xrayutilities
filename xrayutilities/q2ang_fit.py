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
# Copyright (C) 2015-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
Module provides functions to convert a q-vector from reciprocal space to
angular space. a simple implementation uses scipy optimize routines to perform
a fit for a arbitrary goniometer.

The user is, however, expected to use the bounds variable to put restrictions
to the number of free angles to obtain reproducible results. In general only 3
angles are needed to fit an arbitrary q-vector (2 sample + 1 detector angles or
1 sample + 2 detector). More complicated restrictions can be implemented using
the lmfit package. (done upon request!)

The function is based on a fitting routine. For a specific goniometer also
analytic expressions from literature can be used as they are implemented in the
predefined experimental classes HXRD, NonCOP, and GID.
"""

import scipy.optimize
import numpy
import numbers

from . import config
from . import math
from .exception import InputError


def _makebounds(boundsin):
    """
    generate proper bounds for scipy.optimize.minimize function
    from a list/tuple of more convenient bounds.

    Parameters
    ----------
     boundsin:   list/tuple/array of bounds, or fixed values. the number of
                 entries needs to be equal to the number of angle in the
                 goniometer given to the q2ang_general function
                 example input for four gonimeter angles:
                 ((0, 90), 0, (0, 180), (0, 90))

    Returns
    -------
     bounds to be handed over to the scipy.minimize routine. The function will
     expand
    """
    boundsout = []
    for b in boundsin:
        if isinstance(b, (tuple, list, numpy.ndarray)):
            if len(b) == 2:
                boundsout.append((b[0], b[1]))
            elif len(b) == 1:
                boundsout.append((b[0], b[0]))
            else:
                raise InputError('bound values must have two or one elements')
        elif isinstance(b, numbers.Number):
            boundsout.append((b, b))  # variable fixed
        elif b is None:
            boundsout.append((None, None))  # no bound
        else:
            raise InputError('bound value is of invalid type (%s)' % type(b))

    return tuple(boundsout)


def _errornorm_q2ang(angles, qvec, hxrd, U=numpy.identity(3)):
    """
    function to determine the offset in the qposition calculated from
    a set of experimental angles and the given vector

    Parameters
    ----------
     angles   iterable object with angles of the goniometer
     qvec     vector with three q-coordinates
     hxrd     experiment class to be used for the q calculation
     U        orientation matrix

    Returns
    -------
     q-space error between the current fit-guess and the user-specified
     position
    """

    qcalc = hxrd.Ang2Q.point(*angles, UB=U)
    dq = numpy.linalg.norm(qcalc - qvec)
    return dq


def exitAngleConst(angles, alphaf, hxrd):
    """
    helper function for an pseudo-angle constraint for the Q2AngFit-routine.

    Parameters
    ----------
     angles:    fit parameters of Q2AngFit
     alphaf:    the exit angle which should be fixed
     hxrd:      the Experiment object to use for qconversion
    """
    qconv = hxrd._A2QConversion
    # calc kf
    detangles = [a for a in angles[-len(qconv.detectorAxis):]]
    kf = qconv.getDetectorPos(*detangles)
    if numpy.linalg.norm(kf) == 0:
        af = 0
    else:
        ndirlab = qconv.transformSample2Lab(hxrd.Transform(hxrd.ndir), *angles)
        af = 90 - math.VecAngle(kf, ndirlab, deg=True) - alphaf
    return af


def Q2AngFit(qvec, expclass, bounds=None, ormat=numpy.identity(3),
             startvalues=None, constraints=()):
    """
    Functions to convert a q-vector from reciprocal space to angular space.
    This implementation uses scipy optimize routines to perform a fit for a
    goniometer with arbitrary number of goniometer angles.

    The user *must* use the bounds variable to put
    restrictions to the number of free angles to obtain reproducible results.
    In general only 3 angles are needed to fit an arbitrary q-vector (2 sample
    + 1 detector angles or 1 sample + 2 detector).

    Parameters
    ----------
     qvec:      q-vector for which the angular positions should be calculated
     expclass:  experimental class used to define the goniometer for which the
                angles should be calculated.

     keyword arguments(optional):
      bounds:   list of bounds of the goniometer angles. The number of bounds
                must correspond to the number of goniometer angles in the
                expclass.  Angles can also be fixed by supplying only one value
                for a particular angle. e.g.:
                ((low, up), fix, (low2, up2), (low3, up3))
      ormat:    orientation matrix of the sample to be used in the conversion
      startvalues:  start values for the fit, which can significantly speed up
                    the conversion. The number of values must correspond to the
                    number of angles in the goniometer of the expclass
      constraints:  sequence of constraint dictionaries. This allows applying
                    arbitrary (e.g. pseudo-angle) contraints by supplying
                    according constraint functions. (see
                    scipy.optimize.minimize). The supplied function will be
                    called with the arguments (angles, qvec, Experiment, U).

    Returns
    -------
     fittedangles, qerror, errcode:
        list of fitted goniometer angles, the error in reciprocal space and the
        errcode of the scipy minimize function. for a successful fit the error
        code should be <=2
    """

    # check input parameters
    if len(qvec) != 3:
        raise ValueError("XU.Q2AngFit: length of given q-vector is not 3 "
                         "-> invalid")
    lqvec = numpy.asarray(qvec)

    qconv = expclass._A2QConversion
    nangles = len(qconv.sampleAxis) + len(qconv.detectorAxis)

    # generate starting position for optimization
    if startvalues is None:
        start = numpy.zeros(nangles)
    else:
        start = startvalues

    # check bounds
    if bounds is None:
        bounds = numpy.zeros(2 * nangles) - 180.
        bounds[::2] = 180.
        bounds.shape = (nangles, 2)
    elif len(bounds) != nangles:
        raise ValueError("XU.Q2AngFit: number of specified bounds invalid")

    # perform optimization
    res = scipy.optimize.minimize(_errornorm_q2ang, start,
                                  args=(lqvec, expclass, ormat),
                                  method='SLSQP', bounds=_makebounds(bounds),
                                  constraints=constraints,
                                  options={'maxiter': 1000,
                                           'eps': config.EPSILON,
                                           'ftol': config.EPSILON})

    x, errcode, qerror = (res.x, res.status, res.fun)
    if qerror >= 1e-7:
        if config.VERBOSITY >= config.DEBUG:
            print("XU.Q2AngFit: info: need second run")
        # make a second run
        res = scipy.optimize.minimize(_errornorm_q2ang, res.x,
                                      args=(lqvec, expclass, ormat),
                                      method='SLSQP',
                                      bounds=_makebounds(bounds),
                                      constraints=constraints,
                                      options={'maxiter': 1000,
                                               'eps': config.EPSILON,
                                               'ftol': config.EPSILON})
        x, errcode, qerror = (res.x, res.status, res.fun)

    if ((config.VERBOSITY >= config.DEBUG) or (qerror > 10*config.EPSILON and
                                               config.VERBOSITY >=
                                               config.INFO_LOW)):
        print("XU.Q2AngFit: q-error=%.4g with error-code %d (%s)"
              % (qerror, errcode, res.message))

    return x, qerror, errcode
