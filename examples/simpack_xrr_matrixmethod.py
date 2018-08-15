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
# Copyright (C) 2018 Numan Laanait <nlaanait@gmail.com>
# Copyright (C) 2018 Dominik Kriegner <dominik.kriegner@gmail.com>

# Notes:
# 1. Code needs to be tested for multiple layers.
# 2. Code needs to be tested for roughness.

import os

import numpy as np
import xrayutilities as xu
from matplotlib.pylab import *

# Thin-film of (20 nm) CaTiO$_3$/SrTiO$_3$
CTO = xu.materials.CaTiO3
STO = xu.materials.SrTiO3

# Make CTO/STO thin-film.
STO_sub = xu.simpack.Layer(STO, inf)
CTO_layer = xu.simpack.Layer(CTO, 200, roughness=1e-5)
thin_film = xu.simpack.LayerStack('CTO/STO', STO_sub+CTO_layer)

# Initiate dynamical reflectivity model
# Simulating Reflectivity with S,P polarizations @ 1 keV
energy = 1000
model_S = xu.simpack.DynamicalReflectivityModel(thin_film, energy=energy,
                                                polarization='S')
model_P = xu.simpack.DynamicalReflectivityModel(thin_film, energy=energy,
                                                polarization='P')

alpha = np.linspace(0., 90., num=1000)
Reflec_S, Trans_S = model_S.simulate(alpha)
Reflec_P, Trans_P = model_P.simulate(alpha)

# Plots
fig, (ax_P, ax_S) = plt.subplots(1, 2, figsize=(12, 6))

# Reflectance
ax_P.plot(alpha, Reflec_P, label='P-polarization')
ax_P.plot(alpha, Reflec_S, label='S-polarization')
ax_P.legend(loc='upper right')
ax_P.set_xlabel(r'$\alpha$ (deg)')
ax_P.set_ylabel('Reflectivity')
ax_P.set_yscale('log')

# Transmittance
ax_S.plot(alpha, Trans_P, label='P-polarization')
ax_S.plot(alpha, Trans_S, label='S-polarization')
ax_S.legend(loc='upper right')
ax_S.set_xlabel(r'$\alpha$ (deg)')
ax_S.set_ylabel('Transmittance')
# ax_S.set_yscale('log')

suptitle(r'20 nm CaTiO$_3$/SrTiO$_3$, E=%d eV, $\theta_c$=%.2f$^\circ$'
         % (energy, CTO.critical_angle(en=energy)))
tight_layout(rect=(0, 0, 1, 0.94))

# Comparison between Dynamical and Kinematic

# For our simple thin-film, dynamical and kinematic must agree everywhere,
# except for $\alpha \ll \alpha_c$ (strong scattering near angle of total
# external reflection).  There are other conditions where dynamical is more
# accurate (multi-layers, near soft x-ray resonances, etc ...).

xrr_model = xu.simpack.SpecularReflectivityModel(thin_film, energy=energy)
R_kin = xrr_model.simulate(alpha)

# Reflectance
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
R_dyn, R_kin = get_R(alpha)
ax.plot(alpha, R_dyn, label='TransferMatrix')
ax.plot(alpha, R_kin, label='SpecularReflectivityModel', linestyle='dashed')
ax.legend(loc='upper right')
ax.set_xlabel(r'$\alpha$ (deg)')
ax.set_ylabel('Reflectance')
ax.set_yscale('log')
ax.set_title(r'20 nm CaTiO$_3$/SrTiO$_3$, E=%d eV, $\theta_c$=%.2f$^\circ$'
             % (energy, STO.critical_angle(en=energy)))
tight_layout()
