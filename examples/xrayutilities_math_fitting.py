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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 3

x = numpy.linspace(10, 20, 250)
y = (xu.math.PseudoVoigt1d(x, 12, 0.2, 0.1, 10, numpy.random.rand()) +
     0.01 * (numpy.random.rand(len(x))-0.5))

# peaktype can be 'PseudoVoigt', 'Lorentz' or 'Gauss'
# furthermore a constant background (optional also linear with the option
# background='linear') can be added
xu.math.peak_fit(x, y, peaktype='PseudoVoigt', plot=True)

# for fitting multiple peaks simultansously see xu.math.multPeakFit
