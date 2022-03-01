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
# Copyright (C) 2019 Dominik Kriegner <dominik.kriegner@gmail.com>

from matplotlib.pylab import *

import xrayutilities as xu

# global parameters
wavelength = xu.wavelength('CuKa1')
offset = -0.035  # angular offset of the zero position of the data

# set up LayerStack for simulation: InAs(001)/(In,Mn)As(~25 nm)
InAs = xu.materials.InAs
InAs.lattice.a = 6.057
lInAs = xu.simpack.Layer(InAs, inf)
InMnAs = xu.materials.Crystal('InMnAs', xu.materials.SGLattice(
    216, 6.050, atoms=('In', 'Mn', 'As'), pos=('4a', '4a', '4c'),
    occ=(0.88, 0.12, 1)), cij=InAs.cij)
lInMnAs = xu.simpack.Layer(InMnAs, 254, relaxation=0)
pstack = xu.simpack.PseudomorphicStack001('list', lInAs, lInMnAs)

# set up simulation object
thetaMono = arcsin(wavelength/(2 * xu.materials.Ge.planeDistance(2, 2, 0)))
Cmono = cos(2 * thetaMono)
dyn = xu.simpack.DynamicalModel(pstack, I0=1.5e9, background=0,
                                resolution_width=2e-3, polarization='both',
                                Cmono=Cmono)
fitmdyn = xu.simpack.FitModel(dyn)
fitmdyn.set_param_hint('InMnAs_c', vary=True, min=6.02, max=6.06)
fitmdyn.set_param_hint('InAs_a', vary=True)
fitmdyn.set_param_hint('InMnAs_a', expr='InAs_a')
fitmdyn.set_param_hint('resolution_width', vary=True)
params = fitmdyn.make_params()

# plot experimental data
f = figure(figsize=(7, 5))
d = xu.io.RASFile('inas_layer_radial_002_004.ras.bz2', path='data')
scan = d.scans[-1]
tt = scan.data[scan.scan_axis] - offset
semilogy(tt, scan.data['int'], 'o-', ms=3, label='data')

# perform fit and plot the result
fitmdyn.lmodel.set_hkl((0, 0, 4))
ai = (d.scans[-1].data[d.scan.scan_axis] - offset)/2
fitr = fitmdyn.fit(d.scans[-1].data['int'], params, ai)
print(fitr.fit_report())

simdyn = fitmdyn.eval(fitr.params, x=tt/2)
plot(tt, simdyn, label='sim')

# label the plot
ylabel('Intensity (counts)')
xlabel('2Theta (deg)')
legend()
ylim(2e1, 1e9)
xlim(57, 65)
tight_layout()
show()
