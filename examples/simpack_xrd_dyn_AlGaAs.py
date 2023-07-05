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
# Copyright (c) 2016-2023 Dominik Kriegner <dominik.kriegner@gmail.com>



# This is an example of the modelling and fitting of the diffraction data obtained on epitaxial heterostructure
# The design is as follow
#               _________________________
#
#                       GaAs 320 nm          using the same Ga cell as GaAs below (same growth rate)
#               _________________________
#
#                  Al_0.80GaAs 1.8 micron    using the same Ga & Al cells as Al_0.80GaAs below (same growth rate and composition)
#               _________________________
#
#                       GaAs 200 nm           
#               _________________________
#
#                  Al_0.80GaAs 2 micron
#               _________________________
#
#              GaAs substrate and buffer (001)
#
# Xray set-up: X'PERT Panalytical with
#       copper anode,focusing mirror, Ge(220)x4 monochromator, Ge(220)x2 triple axis detection
# Copyright (c) 2023 C2N - CNRS Author:	  Aristide Lemaître <aristide.lemaitre@c2n.upsaclay.fr>



import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt


def x_Al(aper,apar,mat):
    # determination of Al composition from c parameter, poisson coefficient and Veegard's law
    c11 = mat.cij[0,0] # approx of C11 and C12 from alloy with 
    c12 = mat.cij[0,1]
    eps_star = (apar - aper)/apar
    eps_AlAsGaAs = (AlAs_.a - GaAs_.a)/GaAs_.a
    x = -c11 * eps_star / eps_AlAsGaAs / (c11 * (1 + eps_star) + 2 * c12) 
    return x


# read experimental file

file ='data/simpack_xrd_AlGaAs.xrdml'
scan = xu.io.panalytical_xml.XRDMLFile(file)
om = scan.scans[0]['Omega']
Int = scan.scans[0].int


# Omega offset
offset = -0.047634587753920085
om_off = om + offset


# definition of materials

GaAs_ = xu.materials.GaAs
AlAs_ = xu.materials.AlAs
AlGaAs80 = xu.materials.AlGaAs(0.20)


# definition of heterostructure layers

sub = xu.simpack.Layer(GaAs_, float('inf'))
AlGaAs80_1 = xu.simpack.Layer(AlGaAs80, 20000, relaxation=0.)
GaAs_2 = xu.simpack.Layer(GaAs_, 2000, relaxation=0.)
AlGaAs80_3 = xu.simpack.Layer(AlGaAs80, 18000, relaxation=0.)
GaAs_4 = xu.simpack.Layer(GaAs_, 3200, relaxation=0.0)


# heterostructure content
heterostructure_ = AlGaAs80_1 \
                  +GaAs_2 \
                  +AlGaAs80_3\
                  +GaAs_4

# turned into a pseudomorphic structure grown along the 001 direction
heterostructure = xu.simpack.PseudomorphicStack001('Hetereostructure', sub, heterostructure_)

# X-Ray energy
wavelength = xu.wavelength('CuKa1')
en = xu.energy('CuKa1')

# Monochromator
thMono = np.arcsin(wavelength/(2 * xu.materials.Ge.planeDistance(2, 2, 0)))

# Dynamical diffraction model
md = xu.simpack.DynamicalModel(heterostructure, I0 = 4e5,
                               background = 1,
                               energy = en,
                               resolution_width = 0.0015,
                               Cmono = np.cos(2 * thMono),
                               polarization ='both'
                              )

fitmdyn = xu.simpack.FitModel(md)
fitmdyn.lmodel.set_hkl((0, 0, 4))

fitmdyn.verbose = True
fitmdyn.plot = True


# set fit parameters

# AlGaAs80_1, 'AlAs_0_80_GaAs_0_20__c' parameter name found using print(params) (see below)
fitmdyn.set_param_hint('AlAs_0_80_GaAs_0_20__c', vary = True, min = AlGaAs80.a * 0.998, max = AlGaAs80.a * 1.002)
fitmdyn.set_param_hint('AlAs_0_80_GaAs_0_20__thickness', vary = True, min = 19800, max = 20200)


# AlGaAs80_3 is grown using the same cells as AlGaAs_1 hence same composition and growth rate
fitmdyn.set_param_hint('AlAs_0_80_GaAs_0_20__1_c', vary = True, expr = 'AlAs_0_80_GaAs_0_20__c') # same composition as AlGaAs80_1
fitmdyn.set_param_hint('AlAs_0_80_GaAs_0_20__1_thickness', vary = True, expr = 'AlAs_0_80_GaAs_0_20__thickness * 18000 / 20000') # ratio 18000/20000 as thickness ratio between layer 1

#GaAs_2
fitmdyn.set_param_hint('GaAs_1_thickness', vary = True, min = 1990, max = 2010)

#GaAs_4
fitmdyn.set_param_hint('GaAs_2_thickness', vary = True, expr = 'GaAs_1_thickness*3200/2000') # thickness ratio between GaAs_4 and GaAs_2

fitmdyn.set_param_hint('background', vary= True, min = 0, max = 10)
fitmdyn.set_param_hint('I0', vary= True, min = 3e5, max = 5e5)
fitmdyn.set_param_hint('resolution_width', vary = True, min = 0.0012, max = 0.0015)

params = fitmdyn.make_params()


# choose your fitting method, default 'leastsq'
fit_kws = {'method':'differential_evolution'}

# option to avoid error during fit
lmfit_kws = {'nan_policy':'omit'}

# launch fitting procedure
fitr = fitmdyn.fit(Int, params, om_off, fit_kws = fit_kws, lmfit_kws = lmfit_kws)


# get Al composition for perpendicular lattice parameter
x_Al = x_Al(fitr.params['AlAs_0_80_GaAs_0_20__c'], GaAs_.a, AlGaAs80)

# print fit results
print(fitr.fit_report())
print('x_Al = ',x_Al )



# Plot results


plt.clf()

plt.semilogy(om_off, Int, 'r', label = 'Exp data')
simdyn = fitmdyn.eval(fitr.params, x = om)
plt.semilogy(om, simdyn, 'b', label = 'Simul')
plt.xlabel('$\omega(°)$')
plt.legend()
plt.text(33.15, 1000,
         'GaAs: {:.1f} nm \nGaAl$_{{{:.2f}}}$As: {:.0f} nm \nGaAs: {:.1f} nm \nGaAl$_{{{:.2f}}}$As: {:.0f} nm \nSubstrate'.format
             (fitr.params['GaAs_2_thickness'] / 10, x_Al, fitr.params['AlAs_0_80_GaAs_0_20__1_thickness'] / 10,
              fitr.params['GaAs_1_thickness'] / 10, x_Al, fitr.params['AlAs_0_80_GaAs_0_20__thickness']/10))
plt.title('HRXRD')
plt.show()