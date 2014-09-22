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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from .. import config
from .. import experiment
from .. import gridder3d as xugridder


def getindex3d(x, y, z, xgrid, ygrid, zgrid):
    """
    gives the indices of the point x,y,z in the grid given by xgrid ygrid zgrid
    xgrid,ygrid,zgrid must be arrays containing equidistant points

    Parameters
    ----------
     x, y, z:     coordinates of the point of interest (float)
     xgrid, ygrid, zgrid:     grid coordinates in x, y, z direction (array)

    Returns
    -------
     ix, iy, iz:  index of the closest gridpoint (lower left) of the point
                  (x, y, z)
    """

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]

    ix = int((x - xgrid[0]) / dx)
    iy = int((y - ygrid[0]) / dy)
    iz = int((z - zgrid[0]) / dz)

    # check if index is in range of the given grid
    # to speed things up this is assumed to be the case
    # if (ix < 0 or ix > xgrid.size):
    #     print("Warning: point (%8.4f, %8.4f, %8.4f) out of range in x "
    #           "coordinate!"%(x,y,z))
    # if(iy < 0 or iy > ygrid.size):
    #     print("Warning: point (%8.4f, %8.4f, %8.4f) out of range in y "
    #           "coordinate!"%(x,y,z))
    # if (iz < 0 or iz > zgrid.size):
    #     print("Warning: point (%8.4f, %8.4f, %8.4f) out of range in z "
    #           "coordinate!"%(x,y,z))

    return ix, iy, iz


def get_qx_scan3d(gridder, qypos, qzpos, **kwargs):
    """
    extract qx line scan at position y,z from a
    gridded reciprocal space map by taking the closest line of the
    intensity matrix, or summing up a given area around this position

    Parameters
    ----------
     gridder:       3d xrayutilities.Gridder3D object containing the data
     qypos,qzpos:   position at which the line scan should be extracted

    **kwargs:       possible keyword arguments:
      qrange:       integration range perpendicular to scan direction
      qmin,qmax:    minimum and maximum value of extracted scan axis

    Returns
    -------
     qx,qxint: qx scan coordinates and intensities

    Example
    -------
    >>> qxcut,qxcut_int = get_qx_scan3d(gridder,0,0,qrange=0.03)
    """

    if not isinstance(gridder, xugridder.Gridder3D):
        raise TypeError("first argument must be of type XU.Gridder3D")

    if qypos < gridder.yaxis.min() or qypos > gridder.yaxis.max():
        raise ValueError("given qypos is not in the range of the given Y axis")

    if qzpos < gridder.zaxis.min() or qzpos > gridder.zaxis.max():
        raise ValueError("given qzpos is not in the range of the given Z axis")

    if 'qmin' in kwargs:
        qxmin = max(gridder.xaxis.min(), kwargs['qmin'])
    else:
        qxmin = gridder.xaxis.min()

    if 'qmax' in kwargs:
        qxmax = min(gridder.xaxis.max(), kwargs['qmax'])
    else:
        qxmax = gridder.xaxis.max()

    if 'qrange' in kwargs:
        qrange = kwargs['qrange']
    else:
        qrange = 0.

    # find line corresponding to qypos,qzpos
    ixmin, iymin, izmin = getindex3d(
        qxmin, qypos - qrange / 2., qzpos - qrange / 2.,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    ixmax, iymax, izmax = getindex3d(
        qxmax, qypos + qrange / 2., qzpos + qrange / 2.,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    if ('qmin' not in kwargs) and ('qmax' not in kwargs):
        ixmin = 0
        ixmax = gridder.xaxis.size

    if qrange > 0:
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_qx_scan3d: %d points used for integration"
                  % ((izmax - izmin + 1) * (iymax - iymin + 1)))
        return gridder.xaxis[ixmin:ixmax + 1], \
            (gridder.data[ixmin:ixmax + 1,
                          iymin:iymax + 1,
                          izmin:izmax + 1].sum(axis=2)).sum(axis=1) /\
            ((izmax - izmin + 1) * (iymax - iymin + 1))
    else:
        return gridder.xaxis[ixmin:ixmax + 1], \
            gridder.data[ixmin:ixmax + 1, iymin, izmin]


def get_qy_scan3d(gridder, qxpos, qzpos, **kwargs):
    """
    extract qy line scan at position x,z from a
    gridded reciprocal space map by taking the closest line of the
    intensity matrix, or summing up a given area around this position

    Parameters
    ----------
     gridder:       3d xrayutilities.Gridder3D object containing the data
     qxpos,qzpos:   position at which the line scan should be extracted

    **kwargs:       possible keyword arguments:
     qrange:        integration range perpendicular to scan direction
     qmin,qmax:     minimum and maximum value of extracted scan axis

    Returns
    -------
     qy,qyint: qy scan coordinates and intensities

    Example
    -------
    >>> qycut,qycut_int = get_qy_scan3d(gridder,0,0,qrange=0.03)
    """

    if not isinstance(gridder, xugridder.Gridder3D):
        raise TypeError("first argument must be of type XU.Gridder3D")

    if qxpos < gridder.xaxis.min() or qxpos > gridder.xaxis.max():
        raise ValueError("given qxpos is not in the range of the given X axis")

    if qzpos < gridder.zaxis.min() or qzpos > gridder.zaxis.max():
        raise ValueError("given qzpos is not in the range of the given Z axis")

    if 'qmin' in kwargs:
        qymin = max(gridder.yaxis.min(), kwargs['qmin'])
    else:
        qymin = gridder.yaxis.min()

    if 'qmax' in kwargs:
        qymax = min(gridder.yaxis.max(), kwargs['qmax'])
    else:
        qymax = gridder.yaxis.max()

    if 'qrange' in kwargs:
        qrange = kwargs['qrange']
    else:
        qrange = 0.

    # find line corresponding to qxpos,qzpos
    ixmin, iymin, izmin = getindex3d(
        qxpos - qrange / 2., qymin, qzpos - qrange / 2.,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    ixmax, iymax, izmax = getindex3d(
        qxpos + qrange / 2., qymax, qzpos + qrange / 2.,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    if ('qmin' not in kwargs) and ('qmax' not in kwargs):
        iymin = 0
        iymax = gridder.yaxis.size

    if qrange > 0:
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_qy_scan3d: %d points used for integration"
                  % ((izmax - izmin + 1) * (ixmax - ixmin + 1)))
        return gridder.yaxis[iymin:iymax + 1],\
            (gridder.data[ixmin:ixmax + 1,
                          iymin:iymax + 1,
                          izmin:izmax + 1].sum(axis=2)).sum(axis=0) /\
            ((izmax - izmin + 1) * (ixmax - ixmin + 1))
    else:
        return gridder.yaxis[iymin:iymax + 1], \
            gridder.data[ixmin, iymin:iymax + 1, izmin]


def get_qz_scan3d(gridder, qxpos, qypos, **kwargs):
    """
    extract qz line scan at position x,y from a
    gridded reciprocal space map by taking the closest line of the
    intensity matrix, or summing up a given area around this position

    Parameters
    ----------
     gridder:       3d xrayutilities.Gridder3D object containing the data
     qxpos,qypos:   position at which the line scan should be extracted

    **kwargs:       possible keyword arguments:
     qrange:        integration range perpendicular to scan direction
     qmin,qmax:     minimum and maximum value of extracted scan axis

    Returns
    -------
     qz,qzint: qz scan coordinates and intensities

    Example
    -------
    >>> qzcut,qzcut_int = get_qz_scan3d(gridder,0,0,qrange=0.03)
    """

    if not isinstance(gridder, xugridder.Gridder3D):
        raise TypeError("first argument must be of type XU.Gridder3D")

    if qxpos < gridder.xaxis.min() or qxpos > gridder.xaxis.max():
        raise ValueError("given qxpos is not in the range of the given X axis")

    if qypos < gridder.yaxis.min() or qypos > gridder.yaxis.max():
        raise ValueError("given qypos is not in the range of the given Y axis")

    if 'qmin' in kwargs:
        qzmin = max(gridder.zaxis.min(), kwargs['qmin'])
    else:
        qzmin = gridder.zaxis.min()

    if 'qmax' in kwargs:
        qzmax = min(gridder.zaxis.max(), kwargs['qmax'])
    else:
        qzmax = gridder.zaxis.max()

    if 'qrange' in kwargs:
        qrange = kwargs['qrange']
    else:
        qrange = 0.

    # find line corresponding to qxpos,qzpos
    ixmin, iymin, izmin = getindex3d(
        qxpos - qrange / 2., qypos - qrange / 2., qzmin,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    ixmax, iymax, izmax = getindex3d(
        qxpos + qrange / 2., qypos + qrange / 2., qzmax,
        gridder.xaxis, gridder.yaxis, gridder.zaxis)
    if ('qmin' not in kwargs) and ('qmax' not in kwargs):
        izmin = 0
        izmax = gridder.zaxis.size

    if qrange > 0:
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_qz_scan3d: %d points used for integration"
                  % ((ixmax - ixmin + 1) * (iymax - iymin + 1)))
        return gridder.zaxis[izmin:izmax + 1], \
            (gridder.data[ixmin:ixmax + 1,
                          iymin:iymax + 1,
                          izmin:izmax + 1].sum(axis=1)).sum(axis=0) /\
            ((ixmax - ixmin + 1) * (iymax - iymin + 1))
    else:
        return gridder.zaxis[izmin:izmax + 1], \
            gridder.data[ixmin, iymin, izmin:izmax + 1]
