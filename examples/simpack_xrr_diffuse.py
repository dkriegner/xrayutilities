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

import time

import xrayutilities as xu
from matplotlib.pylab import *

sub = xu.simpack.Layer(xu.materials.Si, inf, roughness=1, lat_correl=100)
lay1 = xu.simpack.Layer(xu.materials.Si, 200, roughness=1, lat_correl=200)
lay2 = xu.simpack.Layer(xu.materials.Ge, 70, roughness=3, lat_correl=50)

ls = xu.simpack.LayerStack('SL 5', sub+5*(lay2+lay1))

alphai = arange(0.17, 2, 0.001)

print("calculate method=1, H=1, vert=0")
start = time.time()
m = xu.simpack.DiffuseReflectivityModel(ls, sample_width=10, beam_width=1,
                                        energy='CuKa1', vert_correl=1000,
                                        vert_nu=0, H=1, method=1, vert_int=0)
d1 = m.simulate(alphai)
print("elapsed time: %.4f" % (time.time() - start))


print("calculate method=2, H=1, vert=0")
start = time.time()
m = xu.simpack.DiffuseReflectivityModel(ls, sample_width=10, beam_width=1,
                                        energy='CuKa1', vert_correl=1000,
                                        vert_nu=0, H=1, method=2, vert_int=0)
d2 = m.simulate(alphai)
print("elapsed time: %.4f" % (time.time() - start))

figure()
import warnings
warnings.warn('%g %g %g %g'%(alphai.min(), alphai.max(), d1.min(), d1.max()))
semilogy(alphai, d1, label='method=1')
semilogy(alphai, d2, label='method=2')

legend()
xlabel('incidence angle (deg)')
ylabel('intensity (arb. u.)')
tight_layout()
