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
# Copyright (C) 2017-2020 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
Defines new matplotlib Sqrt scale which further allows for negative values by
using the sign of the original value as sign of the plotted value.
"""

import math

import matplotlib
import numpy
from matplotlib import scale as mscale
from matplotlib import ticker as mticker
from matplotlib import transforms as mtransforms


class SqrtAllowNegScale(mscale.ScaleBase):
    """
    Scales data using a sqrt-function, however, allowing also negative values.

    The scale function:
      sign(y) * sqrt(abs(y))

    The inverse scale function:
      sign(y) * y**2
    """
    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.
        """
        super().__init__(axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(SqrtTickLocator())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return vmin, vmax

    def get_transform(self):
        return self.SqrtTransform()

    class SqrtTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.
            """
            return numpy.sign(a) * numpy.sqrt(numpy.abs(a))

        def inverted(self):
            """
            return the inverse transform for this transform.
            """
            return SqrtAllowNegScale.InvertedSqrtTransform()

    class InvertedSqrtTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return numpy.sign(a) * a**2

        def inverted(self):
            return SqrtAllowNegScale.SqrtTransform()


class SqrtTickLocator(mticker.Locator):
    def __init__(self, nbins=7, symmetric=True):
        self._base = mticker._Edge_integer(1.0, 0)
        self.set_params(nbins, symmetric)

    def set_params(self, nbins, symmetric):
        """Set parameters within this locator."""
        self._nbins = nbins
        self._symmetric = symmetric

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        tmin = math.copysign(math.sqrt(abs(vmin)), vmin)
        tmax = math.copysign(math.sqrt(abs(vmax)), vmax)
        delta = (tmax - tmin) / self._nbins
        locs = numpy.arange(tmin, tmax, self._base.ge(delta))
        if self._symmetric and numpy.sign(tmin) != numpy.sign(tmax):
            locs -= locs[numpy.argmin(numpy.abs(locs))]
        locs = numpy.sign(locs) * locs**2
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest multiples of base that
        contain the data
        """
        vmin = self._base.le(math.copysign(math.sqrt(abs(dmin)), dmin))
        vmax = self._base.ge(math.sqrt(dmax))
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        return mtransforms.nonsingular(math.copysign(vmin**2, vmin), vmax**2)


# register new scale to matplotlib
mscale.register_scale(SqrtAllowNegScale)
