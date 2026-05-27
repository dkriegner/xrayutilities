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
# Copyright (C) 2019-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import math

import numpy
from scipy import special


def _upper_incomplete_gamma_integer(m: int, a):
    """Evaluate the upper incomplete gamma integral for integer order.

    Parameters
    ----------
    m : int
        Positive integer order.
    a : array-like
        Dimensionless lower integration boundary.

    Returns
    -------
    array-like
        Integral ``int_a^inf x**(m - 1) exp(-x) dx``.
    """
    if int(m) != m or m < 1:
        raise ValueError("m must be an integer >= 1")

    return special.gammaincc(int(m), a) * special.gamma(int(m))


def _fft2s(f, x, y):
    """Calculate a centered, area-normalized 2D Fourier transform.

    ``numpy.fft.fft2`` assumes that the first sample is located at coordinate
    zero and returns an unscaled discrete sum. This helper applies phase ramps
    before and after the FFT so that ``f`` is interpreted as sampled on
    symmetric coordinate vectors ``x`` and ``y`` centered around zero. It also
    multiplies the result by ``dx * dy`` so the discrete transform approximates
    the continuous integral

    ``F(qx, qy) = integral integral f(x, y) exp(i * (qx*x + qy*y)) dx dy``.

    Parameters
    ----------
    f : array-like
        Values sampled on the ``(x, y)`` grid.
    x, y : array-like
        Uniform coordinate vectors. For the mosaic model these coordinates are
        real-space distances in angstrom.

    Returns
    -------
    numpy.ndarray
        Complex Fourier amplitudes ordered on the reciprocal grid implied by
        the input sampling. For angstrom input coordinates, the reciprocal
        coordinates are in inverse angstrom.
    """
    f = numpy.asarray(f, dtype=numpy.complex128)
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    nx = x.size
    ny = y.size
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    ix = numpy.arange(nx)
    iy = numpy.arange(ny)
    pre = (
        numpy.exp(1j * math.pi * ix * (nx - 1) / nx)[:, None]
        * numpy.exp(1j * math.pi * iy * (ny - 1) / ny)[None, :]
    )
    fq = numpy.fft.fft2(f * pre)
    post = (
        numpy.exp(
            1j * math.pi * (-((nx - 1) ** 2) / (2 * nx) + ix * (nx - 1) / nx)
        )[:, None]
        * numpy.exp(
            1j * math.pi * (-((ny - 1) ** 2) / (2 * ny) + iy * (ny - 1) / ny)
        )[None, :]
    )
    return dx * dy * fq * post


def mosaic_analytic(qx, qz, RL, RV, Delta, hx, hz, shape):
    """
    simulation of the coplanar reciprocal space map of a single mosaic layer
    using a simple analytic approximation

    Parameters
    ----------
    qx :    array-like
        vector of the qx values (offset from the Bragg peak)
    qz :    array-like
        vector of the qz values (offset from the Bragg peak)
    RL :    float
        lateral block radius in angstrom
    RV :    float
        vertical block radius in angstrom
    Delta : float
        root mean square misorientation of the grains in degree
    hx :    float
        lateral component of the diffraction vector
    hz :    float
        vertical component of the diffraction vector
    shape :  float
        shape factor (1..Gaussian)

    Returns
    -------
    array-like
    2D array with calculated intensities
    """
    QX, QZ = numpy.meshgrid(qx, qz)
    QX = QX.T
    QZ = QZ.T
    DD = numpy.radians(Delta)
    tmp = 6 + DD**2 * ((hz * RL) ** 2 + (hx * RV) ** 2)
    F = (
        (
            (DD * RL * RV) ** 2 * (QZ * hz + QX * hx) ** 2
            + 6 * ((RL * QX) ** 2 + (RV * QZ) ** 2)
        )
        / 4
        / tmp
    )
    return (
        math.pi
        * math.sqrt(6)
        * RL
        * RV
        / math.sqrt(tmp)
        * numpy.exp(-(F**shape))
    )


def mosaic_kinematic(
    qx, qz, RL, RV, m, Delta, taux, tauz, hx, hz, T=0, Lx=0, Lz=0
):
    """Simulate a coplanar reciprocal-space map with random microstrain.

    The model evaluates a real-space mosaic-block correlation function and
    Fourier-transforms it to reciprocal space. In addition to the Gaussian
    damping from mosaic misorientation, two independent Gaussian random-strain
    tensor components are included:

    ``exp(-taux**2 / 2 * (hx * X)**2) * exp(-tauz**2 / 2 * (hz * Z)**2)``.

    Parameters
    ----------
    qx, qz : array-like
        Uniformly sampled reciprocal-space coordinate vectors in inverse
        angstrom. The returned array has shape ``(len(qx), len(qz))``.
    RL, RV : float
        Mean lateral and vertical mosaic-block radii in angstrom. Both must be
        positive.
    m : int
        Order of the Gamma distribution of block radii. Use ``m=0`` to disable
        size averaging and use the finite ellipsoid autocorrelation directly.
        For size averaging, ``m`` must be an integer larger than 3.
    Delta : float
        RMS grain misorientation in degrees.
    taux, tauz : float
        RMS random strain tensor components ``eps_xx`` and ``eps_zz``. These
        are dimensionless strain values, for example ``1e-4``.
    hx, hz : float
        Lateral and vertical components of the diffraction vector in inverse
        angstrom.
    T : float, optional
        Layer thickness in angstrom. ``T=0`` represents the infinite-thickness
        limit.
    Lx, Lz : float, optional
        Coherence widths along ``x`` and ``z`` in angstrom. ``0`` disables the
        corresponding Gaussian coherence envelope.

    Returns
    -------
    numpy.ndarray
        Simulated intensity map in arbitrary units with axes ordered as ``qx``
        by ``qz``.
    """
    qx = numpy.asarray(qx, dtype=float)
    qz = numpy.asarray(qz, dtype=float)
    if qx.ndim != 1 or qz.ndim != 1:
        raise ValueError("qx and qz must be one-dimensional arrays")
    if qx.size < 2 or qz.size < 2:
        raise ValueError("qx and qz must contain at least two points")
    if RL <= 0 or RV <= 0:
        raise ValueError("RL and RV must be positive")
    if m > 0 and (int(m) != m or m <= 3):
        raise ValueError("m must be 0, or an integer > 3")

    dqx = qx[1] - qx[0]
    dqz = qz[1] - qz[0]
    if not numpy.allclose(numpy.diff(qx), dqx) or not numpy.allclose(
        numpy.diff(qz), dqz
    ):
        raise ValueError("qx and qz must be uniformly sampled")
    nx = qx.size
    nz = qz.size
    dx = 2 * math.pi / (nx * dqx)
    dz = 2 * math.pi / (nz * dqz)
    xmax = (nx - 1) / nx * math.pi / dqx
    zmax = (nz - 1) / nz * math.pi / dqz
    x = -xmax + dx * numpy.arange(nx)
    z = -zmax + dz * numpy.arange(nz)

    X, Z = numpy.meshgrid(x, z, indexing="ij")
    A = numpy.sqrt((X / RL) ** 2 + (Z / RV) ** 2)

    if m > 0:
        mm = int(m)
        a = mm * A / 2
        F = (
            _upper_incomplete_gamma_integer(mm, a)
            - 3 / 4 * A * mm * _upper_incomplete_gamma_integer(mm - 1, a)
            + (A * mm) ** 3 / 16 * _upper_incomplete_gamma_integer(mm - 3, a)
        ) / math.gamma(mm)
    else:
        F = (1 - 3 / 4 * A + A**3 / 16) * numpy.heaviside(2 - A, 0.5)

    if T > 0:
        pZ = T - numpy.abs(Z)
        F = F * numpy.heaviside(pZ, 0.5) * pZ

    DD = math.radians(Delta)
    F1 = (
        numpy.exp(-(DD**2) / 6 * (hz * X - hx * Z) ** 2)
        * numpy.exp(-(taux**2) / 2 * (hx * X) ** 2)
        * numpy.exp(-(tauz**2) / 2 * (hz * Z) ** 2)
    )
    F0x = numpy.exp(-((X / Lx) ** 2) * 4 * math.log(2)) if Lx > 0 else 1
    F0z = numpy.exp(-((Z / Lz) ** 2) * 4 * math.log(2)) if Lz > 0 else 1

    return numpy.abs(_fft2s(F * F1 * F0x * F0z, x, z))
