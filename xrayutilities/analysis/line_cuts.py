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
# Copyright (C) 2018 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from .. import config
from ..experiment import HXRD
from ..gridder import FuzzyGridder1D


def _get_cut(pos_along, pos_perp, intensity, dis, npoints):
    """
    obtain a line cut from 2D data using a FuzzyGridder1D to do the hard work.
    Data points with value of pos_perp smaller than `dis` will be considered in
    the line cut

    Parameters
    ----------
    pos_along :     array-like
        position along the cut which should be taken
    pos_perp :      array-like
        distance from the line cut axis. only data points with distance < `dis`
        will be considered
    intensity :     array-like
        data points, `pos_along`, `pos_perp`, and `intensity` must have the
        same shape
    dis :           float
        maximum distance to be allowed for contributing data points
    npoints :       int
        number of points in the output data

    Returns
    -------
    x :     ndarray
        gridded position along the cut axis
    d :     ndarray
        gridded data values for every position `x` along the cut line
    mask:   ndarray
        mask which is 1 for every used data point and 0 for the rest
    """
    mask = numpy.abs(pos_perp) < dis
    g1d = FuzzyGridder1D(npoints)
    width = (numpy.max(pos_along[mask]) - numpy.min(pos_along[mask])) /\
        float(npoints)
    g1d(pos_along[mask], intensity[mask], width=width)
    return g1d.xaxis, g1d.data, mask.astype(numpy.int8)


def get_qz_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    r"""
    extracts a qz scan from reciprocal space map data with integration along
    either, the perpendicular plane in q-space, omega (sample rocking angle) or
    2theta direction. For the integration in angular space (omega, or 2theta)
    the coplanar diffraction geometry with qy and qz as diffraction plane is
    assumed. This is consistent with the coplanar geometry implemented in the
    HXRD-experiment class.

    This function works for 2D and 3D input data in the same way!

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of y, z (list with two components) or x, y, z (list with three
        components) momentum transfers
    intensity : array-like
        2D or 3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    float or tuple/list
        x/y-position at which the line scan should be extracted. this must be a
        float for 2D data and a tuple with two values for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir`, either in 1/\AA (`q`) or degree
        ('omega', or '2theta'). data will be integrated from
        `-intrange/2 .. +intrange/2`

    intdir :    {'q', 'omega', '2theta'}, optional
        integration direction: 'q': perpendicular Q-plane (default), 'omega':
        sample rocking angle, or '2theta': scattering angle.
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        For 3D data the angular integration directions although applicable for
        any set of data only makes sense when the data are aligned into the
        y/z-plane.

    Returns
    -------
    qz, qzint :     ndarray
        qz scan coordinates and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> qzcut, qzcut_int, mask = get_qz_scan([qy, qz], inten, 3.0, 200,
                                             intrange=0.3)
    """
    intdir = kwargs.get('intdir', 'q')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make all data 3D
    if len(qpos) == 2:
        lqpos = [numpy.zeros_like(qpos[0]), qpos[0], qpos[1]]
        lcut = [0, cutpos]
    else:
        lqpos = qpos
        lcut = cutpos

    # make line cuts with selected integration direction
    if intdir == 'q':
        qperp = numpy.sqrt((lqpos[0]-lcut[0])**2 + (lqpos[1]-lcut[1])**2)
        ret = _get_cut(lqpos[2], qperp, intensity, intrange/2, npoints)
    elif intdir == 'omega':
        om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
        q = 4 * numpy.pi / lam * numpy.sin(numpy.radians(tt/2))
        ocut = numpy.degrees(numpy.arcsin(lcut[1]/q)) + tt / 2
        qzpos = q * numpy.cos(numpy.radians(ocut - tt/2))
        ret = _get_cut(qzpos, om-ocut, intensity, intrange/2, npoints)
    elif intdir == '2theta':
        om, chi, phi, tt = hxrd.Q2Ang(*qpos, trans=False, geometry='realTilt')
        q = 4 * numpy.pi / lam * numpy.sin(numpy.radians(tt/2))
        ttcut = 2 * (om - numpy.degrees(numpy.arcsin(lcut[1]/q)))
        qzpos = 4 * numpy.pi / lam * numpy.sin(numpy.radians(ttcut/2)) *\
            numpy.cos(numpy.radians(om - ttcut/2))
        ret = _get_cut(qzpos, tt-ttcut, intensity, intrange/2, npoints)

    return ret


def get_qy_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    r"""
    extracts a qy scan from reciprocal space map data with integration along
    either, the perpendicular plane in q-space, omega (sample rocking angle) or
    2theta direction. For the integration in angular space (omega, or 2theta)
    the coplanar diffraction geometry with qy and qz as diffraction plane is
    assumed. This is consistent with the coplanar geometry implemented in the
    HXRD-experiment class.

    This function works for 2D and 3D input data in the same way!

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of y, z (list with two components) or x, y, z (list with three
        components) momentum transfers
    intensity : array-like
        2D or 3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    float or tuple/list
        x/z-position at which the line scan should be extracted. this must be a
        float for 2D data (z-position) and a tuple with two values for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir`, either in 1/\AA (`q`) or degree
        ('omega', or '2theta'). data will be integrated from
        `-intrange .. +intrange`

    intdir :    {'q', 'omega', '2theta'}, optional
        integration direction: 'q': perpendicular Q-plane (default), 'omega':
        sample rocking angle, or '2theta': scattering angle.
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        For 3D data the angular integration directions although applicable for
        any set of data only makes sense when the data are aligned into the
        y/z-plane.

    Returns
    -------
    qy, qyint :     ndarray
        qy scan coordinates and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> qycut, qycut_int, mask = get_qy_scan([qy, qz], inten, 5.0, 250,
                                             intrange=0.02, intdir='2theta')
    """
    intdir = kwargs.get('intdir', 'q')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make all data 3D
    if len(qpos) == 2:
        lqpos = [numpy.zeros_like(qpos[0]), qpos[0], qpos[1]]
        lcut = [0, cutpos]
    else:
        lqpos = qpos
        lcut = cutpos

    # make line cuts with selected integration direction
    if intdir == 'q':
        qperp = numpy.sqrt((lqpos[0]-lcut[0])**2 + (lqpos[2]-lcut[1])**2)
        ret = _get_cut(lqpos[1], qperp, intensity, intrange/2, npoints)
    elif intdir == 'omega':
        om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
        q = 4 * numpy.pi / lam * numpy.sin(numpy.radians(tt/2))
        ocut = tt / 2 + (numpy.sign(lqpos[1]) *
                         numpy.degrees(numpy.arccos(lcut[1]/q)))
        qypos = q * numpy.sin(numpy.radians(ocut - tt/2))
        ret = _get_cut(qypos, om-ocut, intensity, intrange/2, npoints)
    elif intdir == '2theta':
        om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
        ttcut = om - numpy.degrees(numpy.arcsin(numpy.sin(numpy.radians(om)) -
                                   lcut[1]*lam/(2*numpy.pi)))
        q = 4 * numpy.pi / lam * numpy.sin(numpy.radians(ttcut/2))
        qypos = q * numpy.sin(numpy.radians(om - ttcut/2))
        ret = _get_cut(qypos, tt-ttcut, intensity, intrange/2, npoints)

    return ret


def get_qx_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    r"""
    extracts a qx scan from reciprocal space map data with integration along
    either, the perpendicular plane in q-space, omega (sample rocking angle) or
    2theta direction. For the integration in angular space (omega, or 2theta)
    the coplanar diffraction geometry with qy and qz as diffraction plane is
    assumed. This is consistent with the coplanar geometry implemented in the
    HXRD-experiment class.

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of x, y, z (list with three components) momentum transfers
    intensity : array-like
        3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    float or tuple/list
        y/z-position at which the line scan should be extracted. this must be a
        float for 2D data (z-position) and a tuple with two values for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir`, either in 1/\AA (`q`) or degree
        ('omega', or '2theta'). data will be integrated from
        `-intrange .. +intrange`

    intdir :    {'q', 'omega', '2theta'}, optional
        integration direction: 'q': perpendicular Q-plane (default), 'omega':
        sample rocking angle, or '2theta': scattering angle.
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        The angular integration directions although applicable for
        any set of data only makes sense when the data are aligned into the
        y/z-plane.

    Returns
    -------
    qx, qxint :     ndarray
        qy scan coordinates and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> qxcut, qxcut_int, mask = get_qy_scan([qx, qy, qz], inten, [0, 2.0],
                                             250, intrange=0.01)
    """
    intdir = kwargs.get('intdir', 'q')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make line cuts with selected integration direction
    if intdir == 'q':
        qperp = numpy.sqrt((qpos[1]-cutpos[0])**2 + (qpos[2]-cutpos[1])**2)
        ret = _get_cut(qpos[0], qperp, intensity, intrange/2, npoints)
    elif intdir == 'omega':
        # needs testing
        om, chi, phi, tt = hxrd.Q2Ang(*qpos, trans=False, geometry='realTilt')
        ocut, dmy, dmy, ttcut = hxrd.Q2Ang(qpos[0], cutpos[0], cutpos[1],
                                           trans=False, geometry='realTilt')
        ret = _get_cut(qpos[0], om-ocut, intensity, intrange/2, npoints)
    elif intdir == '2theta':
        # needs testing
        om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
        ocut, dmy, dmy, ttcut = hxrd.Q2Ang(qpos[0], cutpos[0], cutpos[1],
                                           trans=False, geometry='realTilt')
        ret = _get_cut(qpos[0], tt-ttcut, intensity, intrange/2, npoints)

    return ret


def get_omega_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    """
    extracts an omega scan from reciprocal space map data with integration
    along either the 2theta, or radial (omega-2theta) direction. The coplanar
    diffraction geometry with qy and qz as diffraction plane is assumed. This
    is consistent with the coplanar geometry implemented in the HXRD-experiment
    class.

    This function works for 2D and 3D input data in the same way!

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of y, z (list with two components) or x, y, z (list with three
        components) momentum transfers
    intensity : array-like
        2D or 3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    tuple or list
        y/z-position or x/y/z-position at which the line scan should be
        extracted. this must be have two entries for 2D data (z-position) and a
        three for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir` in degree. data will be integrated
        from `-intrange .. +intrange`

    intdir :    {'2theta', 'radial'}, optional
        integration direction: '2theta': scattering angle (default), or
        'radial': omega-2theta direction.
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        Although applicable for any set of data, the extraction only makes
        sense when the data are aligned into the y/z-plane.

    Returns
    -------
    om, omint :     ndarray
        omega scan coordinates and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> omcut, omcut_int, mask = get_omega_scan([qy, qz], inten, [2.0, 5.0],
                                                250, intrange=0.1)
    """
    intdir = kwargs.get('intdir', '2theta')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make all data 3D
    if len(qpos) == 2:
        lqpos = [numpy.zeros_like(qpos[0]), qpos[0], qpos[1]]
        lcut = [0, cutpos[0], cutpos[1]]
    else:
        lqpos = qpos
        lcut = cutpos

    # make line cuts with selected integration direction
    om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
    ocut, dmy, dmy, ttcut = hxrd.Q2Ang(*lcut, trans=False, geometry='realTilt')
    if intdir == '2theta':
        ret = _get_cut(om, tt-ttcut, intensity, intrange/2, npoints)
    elif intdir == 'radial':
        ret = _get_cut(om-(tt-ttcut)/2, tt-ttcut, intensity, intrange/2,
                       npoints)

    return ret


def get_radial_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    """
    extracts a radial scan from reciprocal space map data with integration
    along either the omega or 2theta direction. The coplanar
    diffraction geometry with qy and qz as diffraction plane is assumed. This
    is consistent with the coplanar geometry implemented in the HXRD-experiment
    class.

    This function works for 2D and 3D input data in the same way!

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of y, z (list with two components) or x, y, z (list with three
        components) momentum transfers
    intensity : array-like
        2D or 3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    tuple or list
        y/z-position or x/y/z-position at which the line scan should be
        extracted. this must be have two entries for 2D data (z-position) and a
        three for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir` in degree. data will be integrated
        from `-intrange .. +intrange`

    intdir :    {'omega', '2theta'}, optional
        integration direction: 'omega': sample rocking angle (default),
        '2theta': scattering angle
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        Although applicable for any set of data, the extraction only makes
        sense when the data are aligned into the y/z-plane.

    Returns
    -------
    tt, omttint :     ndarray
        omega-2theta scan coordinates (2theta values) and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> ttcut, omtt_int, mask = get_radial_scan([qy, qz], inten, [2.0, 5.0],
                                                250, intrange=0.1)
    """
    intdir = kwargs.get('intdir', 'omega')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make all data 3D
    if len(qpos) == 2:
        lqpos = [numpy.zeros_like(qpos[0]), qpos[0], qpos[1]]
        lcut = [0, cutpos[0], cutpos[1]]
    else:
        lqpos = qpos
        lcut = cutpos

    # make line cuts with selected integration direction
    om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
    ocut, dmy, dmy, ttcut = hxrd.Q2Ang(*lcut, trans=False, geometry='realTilt')
    if intdir == 'omega':
        ret = _get_cut(tt, om-((tt-ttcut)/2+ocut), intensity, intrange/2,
                       npoints)
    elif intdir == '2theta':
        offcut = ttcut/2 - ocut
        ret = _get_cut(tt-2*(tt/2-om-offcut), 2*(tt/2-om-offcut), intensity,
                       intrange/2, npoints)

    return ret


def get_ttheta_scan(qpos, intensity, cutpos, npoints, intrange, **kwargs):
    """
    extracts a 2theta scan from reciprocal space map data with integration
    along either the omega or radial direction. The coplanar
    diffraction geometry with qy and qz as diffraction plane is assumed. This
    is consistent with the coplanar geometry implemented in the HXRD-experiment
    class.

    This function works for 2D and 3D input data in the same way!

    Parameters
    ----------
    qpos :      list of array-like objects
        arrays of y, z (list with two components) or x, y, z (list with three
        components) momentum transfers
    intensity : array-like
        2D or 3D array of reciprocal space intensity with shape equal to the
        qpos entries
    cutpos :    tuple or list
        y/z-position or x/y/z-position at which the line scan should be
        extracted. this must be have two entries for 2D data (z-position) and a
        three for 3D data
    npoints :   int
        number of points in the output data
    intrange :  float
        integration range in along `intdir` in degree. data will be integrated
        from `-intrange .. +intrange`

    intdir :    {'omega', 'radial'}, optional
        integration direction: 'omega': sample rocking angle (default),
        'radial': omega-2theta direction
    wl :       float or str, optional
        wavelength used to determine angular integration positions

    Note:
        Although applicable for any set of data, the extraction only makes
        sense when the data are aligned into the y/z-plane.

    Returns
    -------
    tt, ttint :     ndarray
        2theta scan coordinates and intensities
    used_mask :     ndarray
        mask of used data, shape is the same as the input intensity: True for
        points which contributed, False for all others

    Examples
    --------
    >>> ttcut, tt_int, mask = get_ttheta_scan([qy, qz], inten, [2.0, 5.0],
                                              250, intrange=0.1)
    """
    intdir = kwargs.get('intdir', 'omega')
    lam = kwargs.get('wl', config.WAVELENGTH)
    hxrd = HXRD([1, 0, 0], [0, 0, 1], wl=lam)

    # make all data 3D
    if len(qpos) == 2:
        lqpos = [numpy.zeros_like(qpos[0]), qpos[0], qpos[1]]
        lcut = [0, cutpos[0], cutpos[1]]
    else:
        lqpos = qpos
        lcut = cutpos

    # make line cuts with selected integration direction
    om, chi, phi, tt = hxrd.Q2Ang(*lqpos, trans=False, geometry='realTilt')
    ocut, dmy, dmy, ttcut = hxrd.Q2Ang(*lcut, trans=False, geometry='realTilt')
    if intdir == 'omega':
        ret = _get_cut(tt, om-ocut, intensity, intrange/2, npoints)
    elif intdir == 'radial':
        ret = _get_cut(tt-2*(om-ocut), 2*(om-ocut), intensity,
                       intrange/2, npoints)

    return ret
