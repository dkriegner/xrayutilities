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

from .. import config


def center_of_mass(pos, data, background='none', full_output=False):
    """
    function to determine the center of mass of an array

    Parameter
    ---------
     pos:  position of the data points
     data: data values
     background: type of background, either 'none', 'constant' or 'linear'
     full_output: return background cleaned data and background-parameters

    Returns
    -------
     center of mass position (single float)
    """
    # subtract background
    slope = 0
    back = 0
    if background == 'linear':
        dx = float(pos[-1] - pos[0])
        if abs(dx) > 0:
            slope = (float(data[-1]) - float(data[0])) / dx
        back = (data[0] - slope * pos[0] +
                data[-1] - slope * pos[-1]) / 2.0
        ld = data - (slope * pos + back)
    elif background =='constant':
        back = numpy.median(data)
        ld = data - numpy.min(data)
    else:
        ld = data

    ipos = numpy.sum(pos * ld) / numpy.sum(ld)
    if full_output:
        return ipos, ld, back, slope
    else:
        return ipos


def fwhm_exp(pos, data):
    """
    function to determine the full width at half maximum value of experimental
    data. Please check the obtained value visually (noise influences the
    result)

    Parameter
    ---------
     pos:  position of the data points
     data: data values

    Returns
    -------
     fwhm value (single float)
    """

    m = data.max()
    p0 = numpy.argmax(data)
    datal = data[:p0+1]
    datar = data[p0:]

    # determine left side half value position
    try:
        pls = pos[:p0+1][datal < m / 2.][-1]
        pll = pos[:p0+1][datal > m / 2.][0]
        ds = data[pos == pls][0]
        dl = data[pos == pll][0]
        pl = pls + (pll - pls) * (m / 2. - ds) / (dl - ds)
    except IndexError:
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.math.fwhm_exp: warning: left side half value could"
                  " not be determined -> returns 2*hwhm")
        pl = 0

    # determine right side half value position
    try:
        prs = pos[p0:][datar < m / 2.][0]
        prl = pos[p0:][datar > m / 2.][-1]
        ds = data[pos == prs][0]
        dl = data[pos == prl][0]
        pr = prs + (prl - prs) * (m / 2. - ds) / (dl - ds)
    except IndexError:
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.math.fwhm_exp: warning: right side half value could"
                  " not be determined -> returns 2*hwhm")
        pr = 0

    return numpy.abs(pr - pl)
