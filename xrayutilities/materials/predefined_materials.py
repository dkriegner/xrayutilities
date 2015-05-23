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
# Copyright (C) 2013,2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import numpy

from . import __path__
from .material import Material, CubicElasticTensor, CubicAlloy
from .material import HexagonalElasticTensor, WZTensorFromCub
from . import lattice
from . import elements
from .cif import CIFFile
from .. import config

# some predefined materials
# PLEASE use N/m^2 as unit for cij for newly entered material ( 1 dyn/cm^2 =
# 0.1 N/m^2 = 0.1 GPa)
# Use Kelvin as unit for the Debye temperature
Si = Material("Si", lattice.DiamondLattice(elements.Si, 5.43104),
              CubicElasticTensor(165.77e+9, 63.93e+9, 79.62e+9),
              thetaDebye=640)
Ge = Material("Ge", lattice.DiamondLattice(elements.Ge, 5.65785),
              CubicElasticTensor(128.5e+9, 48.3e+9, 66.8e+9), thetaDebye=374)
InAs = Material("InAs",
                lattice.ZincBlendeLattice(elements.In, elements.As, 6.0583),
                CubicElasticTensor(8.34e+10, 4.54e+10, 3.95e+10),
                thetaDebye=280)
InP = Material("InP",
               lattice.ZincBlendeLattice(elements.In, elements.P, 5.8687),
               CubicElasticTensor(10.11e+10, 5.61e+10, 4.56e+10),
               thetaDebye=425)
InSb = Material("InSb",
                lattice.ZincBlendeLattice(elements.In, elements.Sb, 6.47937),
                CubicElasticTensor(6.66e+10, 3.65e+10, 3.02e+10),
                thetaDebye=160)
GaP = Material("GaP",
               lattice.ZincBlendeLattice(elements.Ga, elements.P, 5.4505),
               CubicElasticTensor(14.05e+10, 6.20e+10, 7.03e+10),
               thetaDebye=445)
GaAs = Material("GaAs",
                lattice.ZincBlendeLattice(elements.Ga, elements.As, 5.65325),
                CubicElasticTensor(11.9e+10, 5.34e+10, 5.96e+10),
                thetaDebye=360)
AlAs = Material("AlAs",
                lattice.ZincBlendeLattice(elements.Al, elements.As, 5.6611),
                CubicElasticTensor(12.02e+10, 5.70e+10, 5.99e+10),
                thetaDebye=446)
GaSb = Material("GaSb",
                lattice.ZincBlendeLattice(elements.Ga, elements.Sb, 6.09593),
                CubicElasticTensor(8.83e+10, 4.02e+10, 4.32e+10),
                thetaDebye=266)
# from unpublished
GaAsWZ = Material("GaAs(WZ)",
                  lattice.WurtziteLattice(elements.Ga, elements.As,
                                          3.9845, 6.5701),
                  WZTensorFromCub(11.9e+10, 5.34e+10, 5.96e+10))
GaAs4H = Material("GaAs(4H)", lattice.WurtziteLattice(elements.Ga, elements.As,
                                                      3.9900, 13.0964))
# from Phys. Rev. B 88, 115315 (2013)
GaPWZ = Material("GaP(WZ)",
                 lattice.WurtziteLattice(elements.Ga, elements.P,
                                         3.8419, 6.3353, u=0.37385),
                 WZTensorFromCub(14.05e+10, 6.20e+10, 7.03e+10))
# from Nanotechnology 22 425704 (2011)
InPWZ = Material("InP(WZ)",
                 lattice.WurtziteLattice(elements.In, elements.P,
                                         4.1423, 6.8013),
                 WZTensorFromCub(10.11e+10, 5.61e+10, 4.56e+10))
# from Nano Lett., 2011, 11 (4), pp 1483-1489
InAsWZ = Material("InAs(WZ)",
                  lattice.WurtziteLattice(elements.In, elements.As,
                                          4.2742, 7.0250),
                  WZTensorFromCub(8.34e+10, 4.54e+10, 3.95e+10))
InAs4H = Material("InAs(4H)", lattice.WurtziteLattice(elements.In, elements.As,
                                                      4.2780, 14.0171))
InSbWZ = Material("InSb(WZ)",
                  lattice.WurtziteLattice(elements.In, elements.Sb,
                                          4.5712, 7.5221),
                  WZTensorFromCub(6.66e+10, 3.65e+10, 3.02e+10))
InSb4H = Material("InSb(4H)", lattice.WurtziteLattice(elements.In, elements.Sb,
                                                      4.5753, 15.0057))

# ? Unit of elastic constants for CdTe,PbTe,PbSe ?
PbTe = Material(
    "PbTe",
    lattice.RockSalt_Cubic_Lattice(elements.Pb, elements.Te, 6.464),
    CubicElasticTensor(93.6, 7.7, 13.4))
PbSe = Material(
    "PbSe",
    lattice.RockSalt_Cubic_Lattice(elements.Pb, elements.Se, 6.128),
    CubicElasticTensor(123.7, 19.3, 15.9))
CdTe = Material("CdTe",
                lattice.ZincBlendeLattice(elements.Cd, elements.Te, 6.482),
                CubicElasticTensor(53.5, 36.7, 19.9))
CdSe = Material(
    "CdSe",
    lattice.WurtziteLattice(elements.Cd, elements.Se, 4.300, 7.011),
    HexagonalElasticTensor(7.490e10, 4.609e10, 3.926e10, 8.451e10, 1.315e10))
CdSe_ZB = Material("CdSe ZB",
                   lattice.ZincBlendeLattice(elements.Cd, elements.Se, 6.052))
HgSe = Material("HgSe",
                lattice.ZincBlendeLattice(elements.Hg, elements.Se, 6.085),
                CubicElasticTensor(6.1e10, 4.4e10, 2.2e10))

NaCl = Material(
    "NaCl",
    lattice.RockSalt_Cubic_Lattice(elements.Na, elements.Cl, 5.6402))
MgO = Material("MgO",
               lattice.RockSalt_Cubic_Lattice(elements.Mg, elements.O, 4.212))
GaN = Material("GaN",
               lattice.WurtziteLattice(elements.Ga, elements.N, 3.189, 5.186),
               HexagonalElasticTensor(390.e9, 145.e9, 106.e9, 398.e9, 105.e9),
               thetaDebye=600)
BaF2 = Material("BaF2", lattice.CubicFm3mBaF2(elements.Ba, elements.F, 6.2001))
SrF2 = Material("SrF2", lattice.CubicFm3mBaF2(elements.Sr, elements.F, 5.8007))
MnTe = Material("MnTe",
                lattice.NiAsLattice(elements.Mn, elements.Te, 4.1429, 6.7031))
GeTe = Material(
    "GeTe",
    lattice.GeTeRhombohedral(elements.Ge, elements.Te, 5.996, 88.18, 0.237))
Al = Material("Al", lattice.FCCLattice(elements.Al, 4.04958))
Au = Material("Au", lattice.FCCLattice(elements.Au, 4.0782))
Fe = Material("Fe", lattice.BCCLattice(elements.Fe, 2.8665))
Rh = Material("Rh", lattice.FCCLattice(elements.Rh, 3.8034))
V = Material("V", lattice.BCCLattice(elements.V, 3.024))
Ta = Material("Ta", lattice.BCCLattice(elements.Ta, 3.306))
Ag2Se = Material(
    "Ag2Se",
    lattice.NaumanniteLattice(elements.Ag, elements.Se, 4.333, 7.062, 7.764))
VO2_Rutile = Material(
    "VO2",
    lattice.RutileLattice(elements.V, elements.O, 4.55, 2.88, 0.305))
VO2_Baddeleyite = Material(
    "VO2",
    lattice.BaddeleyiteLattice(elements.V, elements.O,
                               5.75, 5.42, 5.38, 122.6))
SiO2 = Material(
    "SiO2",
    lattice.QuartzLattice(elements.Si, elements.O, 4.916, 4.916, 5.4054))
In = Material("In",
              lattice.TetragonalIndiumLattice(elements.In, 3.2523, 4.9461))
Sb = Material("Sb", lattice.TrigonalR3mh(elements.Sb, 4.307, 11.273))
Sn = Material("Sn", lattice.TetragonalTinLattice(elements.Sn, 5.8197, 3.17488))
Ag = Material("Ag", lattice.FCCLattice(elements.Ag, 4.0853))
SnAlpha = Material("Sn-alpha", lattice.DiamondLattice(elements.Sn, 6.4912))
Cu = Material("Cu", lattice.FCCLattice(elements.Cu, 3.61496))
CuMnAs = Material("CuMnAs",
                  lattice.CuMnAsLattice(elements.Cu, elements.Mn, elements.As,
                                        3.82, 3.82, 6.30))
CaTiO3 = Material(
    "CaTiO3",
    lattice.PerovskiteTypeRhombohedral(elements.Ca, elements.Ti, elements.O,
                                       3.795, 90))
BiFeO3 = Material(
    "BiFeO3",
    lattice.PerovskiteTypeRhombohedral(elements.Bi, elements.Fe, elements.O,
                                       3.965, 89.3))
FeO = Material("FeO",
               lattice.RockSalt_Cubic_Lattice(elements.Fe, elements.O, 4.332))
CoO = Material("CoO",
               lattice.RockSalt_Cubic_Lattice(elements.Co, elements.O, 4.214))
Fe3O4 = Material(
    "Fe3O4",
    lattice.MagnetiteLattice(elements.Fe, elements.Fe, elements.O, 8.3958))
Co3O4 = Material(
    "Co3O4",
    lattice.MagnetiteLattice(elements.Co, elements.Co, elements.O, 8.0821))
FeRh = Material("FeRh", lattice.CsClLattice(elements.Fe, elements.Rh, 2.993))
Ir20Mn80 = Material(
    "Ir20Mn80",
    lattice.FCCSharedLattice(elements.Ir, elements.Mn, 0.2, 0.8, 3.780))

# materials defined from cif file
try:
    Al2O3 = Material.fromCIF(os.path.join(__path__[0], "data", "Al2O3.cif"))
except:
    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.materials: Warning: import of CIF "
              "file based material failed")


# Alloys with special properties
class SiGe(CubicAlloy):

    def __init__(self, x):
        CubicAlloy.__init__(self, Si, Ge, x)

    def lattice_const_AB(self, latA, latB, x):
        """
        method to calculate the lattice parameter of the SiGe alloy with
        composition Si_{1-x}Ge_x
        """
        return latA + (0.2 * x + 0.027 * x ** 2) * \
            latA / numpy.linalg.norm(latA)

    def _setxb(self, x):
        """
        method to set the composition of SiGe to Si_{1-x}Ge_x
        """
        if config.VERBOSITY >= config.DEBUG:
            print("XU.materials.SiGe._setxb: jump to base class")
        CubicAlloy._setxb(self, x)
        if config.VERBOSITY >= config.DEBUG:
            print("back from base class")
        # the lattice parameters need to be done in a different way
        a = self.lattice_const_AB(
            self.matA.lattice.a1[0], self.matB.lattice.a1[0], x)
        self.lattice = lattice.CubicLattice(a)
        self.rlattice = self.lattice.ReciprocalLattice()

    x = property(CubicAlloy._getxb, _setxb)
