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
# Copyright (C) 2013-2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path

import numpy

from . import elements as e
from . import __path__
from .. import config
from .cif import CIFFile
from .material import (Amorphous, Crystal, CubicAlloy, CubicElasticTensor,
                       HexagonalElasticTensor, WZTensorFromCub)
from .spacegrouplattice import SGLattice

# some predefined materials
# PLEASE use N/m^2 as unit for cij for newly entered material ( 1 dyn/cm^2 =
# 0.1 N/m^2 = 0.1 GPa)
# Use Kelvin as unit for the Debye temperature
# ref http://www.semiconductors.co.uk/propiviv5431.htm
C = Crystal("C", lattice=SGLattice('227:1', 3.5668, atoms=[e.C, ], pos=['8a', ]),
            cij=CubicElasticTensor(1.0764e12, 0.125e12, 0.577e12))
C_HOPG = Crystal('HOPG', lattice=SGLattice(194, 2.4612, 6.7079,
                                   atoms=[e.C, e.C], pos=['2b', '2c']))
Si = Crystal("Si", lattice=SGLattice('227:1', 5.43104, atoms=[e.Si, ], pos=['8a', ]),
             cij=CubicElasticTensor(165.77e+9, 63.93e+9, 79.62e+9), thetaDebye=640)
Ge = Crystal("Ge", lattice=SGLattice('227:1', 5.65785, atoms=[e.Ge, ], pos=['8a', ]),
             cij=CubicElasticTensor(128.5e+9, 48.3e+9, 66.8e+9), thetaDebye=374)
InAs = Crystal("InAs", lattice=SGLattice(216, 6.0583, atoms=[e.In, e.As],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(8.34e+10, 4.54e+10, 3.95e+10),
               thetaDebye=280)
InP = Crystal("InP", lattice=SGLattice(216, 5.8687, atoms=[e.In, e.P],
                               pos=['4a', '4c']),
              cij=CubicElasticTensor(10.11e+10, 5.61e+10, 4.56e+10),
              thetaDebye=425)
InSb = Crystal("InSb", lattice=SGLattice(216, 6.47937, atoms=[e.In, e.Sb],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(6.66e+10, 3.65e+10, 3.02e+10),
               thetaDebye=160)
GaP = Crystal("GaP", lattice=SGLattice(216, 5.4505, atoms=[e.Ga, e.P],
                               pos=['4a', '4c']),
              cij=CubicElasticTensor(14.05e+10, 6.20e+10, 7.03e+10),
              thetaDebye=445)
GaAs = Crystal("GaAs", lattice=SGLattice(216, 5.65325, atoms=[e.Ga, e.As],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(11.9e+10, 5.34e+10, 5.96e+10),
               thetaDebye=360)
AlAs = Crystal("AlAs", lattice=SGLattice(216, 5.6611, atoms=[e.Al, e.As],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(12.02e+10, 5.70e+10, 5.99e+10),
               thetaDebye=446)
GaSb = Crystal("GaSb", lattice=SGLattice(216, 6.09593, atoms=[e.Ga, e.Sb],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(8.83e+10, 4.02e+10, 4.32e+10),
               thetaDebye=266)
# from Cryst. Growth Des. 15, 4795-4803 (2015)
GaAsWZ = Crystal("GaAs(WZ)",
                 lattice=SGLattice(186, 3.9845, 6.5701, atoms=[e.Ga, e.As],
                           pos=[('2b', 0), ('2b', 3/8.)]),
                 cij=WZTensorFromCub(11.9e+10, 5.34e+10, 5.96e+10))
GaAs4H = Crystal("GaAs(4H)",
                 lattice=SGLattice(186, 3.9900, 13.0964,
                           atoms=[e.Ga, e.Ga, e.As, e.As],
                           pos=[('2a', 0), ('2b', 1/4.),
                                ('2a', 3/16.), ('2b', 7/16.)]))
# from Phys. Rev. B 88, 115315 (2013)
GaPWZ = Crystal("GaP(WZ)",
                lattice=SGLattice(186, 3.8419, 6.3353, atoms=[e.Ga, e.P],
                          pos=[('2b', 0), ('2b', 0.37385)]),
                cij=WZTensorFromCub(14.05e+10, 6.20e+10, 7.03e+10))
# from Nanotechnology 22 425704 (2011)
InPWZ = Crystal("InP(WZ)",
                lattice=SGLattice(186, 4.1423, 6.8013, atoms=[e.In, e.P],
                          pos=[('2b', 0), ('2b', 3/8.)]),
                cij=WZTensorFromCub(10.11e+10, 5.61e+10, 4.56e+10))
# from Nano Lett., 2011, 11 (4), pp 1483-1489
InAsWZ = Crystal("InAs(WZ)",
                 lattice=SGLattice(186, 4.2742, 7.0250, atoms=[e.In, e.As],
                           pos=[('2b', 0), ('2b', 3/8.)]),
                 cij=WZTensorFromCub(8.34e+10, 4.54e+10, 3.95e+10))
InAs4H = Crystal("InAs(4H)",
                 lattice=SGLattice(186, 4.2780, 14.0171,
                           atoms=[e.In, e.In, e.As, e.As],
                           pos=[('2a', 0), ('2b', 1/4.),
                                ('2a', 3/16.), ('2b', 7/16.)]))
InSbWZ = Crystal("InSb(WZ)",
                 lattice=SGLattice(186, 4.5712, 7.5221, atoms=[e.In, e.Sb],
                           pos=[('2b', 0), ('2b', 3/8.)]),
                 cij=WZTensorFromCub(6.66e+10, 3.65e+10, 3.02e+10))
InSb4H = Crystal("InSb(4H)",
                 lattice=SGLattice(186, 4.5753, 15.0057,
                           atoms=[e.In, e.In, e.Sb, e.Sb],
                           pos=[('2a', 0), ('2b', 1/4.),
                                ('2a', 3/16.), ('2b', 7/16.)]))

# ? Unit of elastic constants for CdTe, PbTe, PbSe ?
PbTe = Crystal("PbTe",
               lattice=SGLattice(225, 6.464, atoms=[e.Pb, e.Te], pos=['4a', '4b']),
               cij=CubicElasticTensor(93.6, 7.7, 13.4))
PbSe = Crystal("PbSe",
               lattice=SGLattice(225, 6.128, atoms=[e.Pb, e.Se], pos=['4a', '4b']),
               cij=CubicElasticTensor(123.7, 19.3, 15.9))
CdTe = Crystal("CdTe", lattice=SGLattice(216, 6.482, atoms=[e.Cd, e.Te],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(53.5, 36.7, 19.9))
CdSe = Crystal("CdSe", lattice=SGLattice(186, 4.5712, 7.5221, atoms=[e.In, e.Sb],
                                 pos=[('2b', 0), ('2b', 3/8.)]),
               cij=HexagonalElasticTensor(7.490e10, 4.609e10, 3.926e10,
                                      8.451e10, 1.315e10))
CdSe_ZB = Crystal("CdSe(ZB)", lattice=SGLattice(216, 6.052, atoms=[e.Cd, e.Se],
                                        pos=['4a', '4c']))
HgSe = Crystal("HgSe", lattice=SGLattice(216, 6.085, atoms=[e.Hg, e.Se],
                                 pos=['4a', '4c']),
               cij=CubicElasticTensor(6.1e10, 4.4e10, 2.2e10))
NaCl = Crystal("NaCl",
               lattice=SGLattice(225, 5.6402, atoms=[e.Na, e.Cl], pos=['4a', '4b']))
MgO = Crystal("MgO",
              lattice=SGLattice(225, 4.212, atoms=[e.Mg, e.O], pos=['4a', '4b']))
GaN = Crystal("GaN",
              lattice=SGLattice(186, 3.189, 5.186, atoms=[e.Ga, e.N],
                        pos=[('2b', 0), ('2b', 3/8.)]),
              cij=HexagonalElasticTensor(390.e9, 145.e9, 106.e9, 398.e9, 105.e9),
              thetaDebye=600)
BaF2 = Crystal("BaF2", lattice=SGLattice(225, 6.2001, atoms=[e.Ba, e.F],
                                 pos=['4a', '8c']))
SrF2 = Crystal("SrF2", lattice=SGLattice(225, 5.8007, atoms=[e.Sr, e.F],
                                 pos=['4a', '8c']))
CaF2 = Crystal("CaF2", lattice=SGLattice(225, 5.4631, atoms=[e.Ca, e.F],
                                 pos=['4a', '8c']))
MnTe = Crystal("MnTe", lattice=SGLattice(186, 4.1429, 6.7031, atoms=[e.Mn, e.Te],
                                 pos=[('2a', 0), ('2b', 0.25)]))
GeTe = Crystal("GeTe",
               lattice=SGLattice('160:R', 5.996, 88.18, atoms=[e.Ge, e.Ge, e.Te, e.Te],
                         pos=[('1a', -0.237), ('3b', (0.5-0.237, -0.237)),
                              ('1a', 0.237), ('3b', (0.5+0.237, +0.237))]))
SnTe = Crystal("SnTe",
               lattice=SGLattice(225, 6.3268, atoms=[e.Sn, e.Te], pos=['4a', '4b']))
Al = Crystal("Al", lattice=SGLattice(225, 4.04958, atoms=[e.Al, ], pos=['4a', ]))
Au = Crystal("Au", lattice=SGLattice(225, 4.0782, atoms=[e.Au, ], pos=['4a', ]))
Fe = Crystal("Fe", lattice=SGLattice(229, 2.8665, atoms=[e.Fe, ], pos=['2a', ]))
Co = Crystal("Co", lattice=SGLattice(194, 2.5071, 4.0695, atoms=[e.Co, ],
                             pos=['2c', ]))
Ru = Crystal("Ru", lattice=SGLattice(194, 2.7059, 4.2815, atoms=[e.Ru, ],
                             pos=['2c', ]))
Rh = Crystal("Rh", lattice=SGLattice(225, 3.8034, atoms=[e.Rh, ], pos=['4a', ]))
V = Crystal("V", lattice=SGLattice(229, 3.024, atoms=[e.V, ], pos=['2a', ]))
Ta = Crystal("Ta", lattice=SGLattice(229, 3.306, atoms=[e.Ta, ], pos=['2a', ]))
Pt = Crystal("Pt", lattice=SGLattice(225, 3.9242, atoms=[e.Pt, ], pos=['4a', ]))
Ag2Se = Crystal("Ag2Se", lattice=SGLattice(19, 4.333, 7.062, 7.764,
                atoms=[e.Ag, e.Ag, e.Se],
                pos=[('4a', (0.107, 0.369, 0.456)),
                     ('4a', (0.728, 0.029, 0.361)),
                     ('4a', (0.358, 0.235, 0.149))]))
VO2_Rutile = Crystal("VO2", lattice=SGLattice(136, 4.55, 2.88, atoms=[e.V, e.O],
                                      pos=['2a', ('4f', 0.305)]))
VO2_Baddeleyite = Crystal("VO2", lattice=SGLattice(14, 5.75, 5.42, 5.38, 122.6,
                          atoms=[e.V, e.O, e.O],
                          pos=[('4e', (0.242, 0.975, 0.025)),
                               ('4e', (0.1, 0.21, 0.20)),
                               ('4e', (0.39, 0.69, 0.29))]))
SiO2 = Crystal("SiO2", lattice=SGLattice(154, 4.916, 5.4054, atoms=[e.Si, e.O],
               pos=[('3a', 0.46970),
                    ('6c', (0.41350, 0.26690, 0.11910+2/3.))]))
In = Crystal("In", lattice=SGLattice(139, 3.2523, 4.9461, atoms=[e.In, ],
                             pos=['2a', ]))
Sb = Crystal("Sb", lattice=SGLattice('166:H', 4.307, 11.273, atoms=[e.Sb, ],
                             pos=[('6c', 0.23349), ]))
Sn = Crystal("Sn", lattice=SGLattice('141:1', 5.8197, 3.17488, atoms=[e.Sn, ],
                             pos=['4a', ]))
Ag = Crystal("Ag", lattice=SGLattice(225, 4.0853, atoms=[e.Ag, ], pos=['4a', ]))
SnAlpha = Crystal("Sn-alpha", lattice=SGLattice('227:1', 6.4912, atoms=[e.Sn, ],
                                        pos=['8a', ]))
Cu = Crystal("Cu", lattice=SGLattice(225, 3.61496, atoms=[e.Cu, ], pos=['4a', ]))
CaTiO3 = Crystal("CaTiO3", lattice=SGLattice(221, 3.795, atoms=[e.Ca, e.Ti, e.O],
                                     pos=['1a', '1b', '3c']))
SrTiO3 = Crystal("SrTiO3", lattice=SGLattice(221, 3.905, atoms=[e.Sr, e.Ti, e.O],
                                     pos=['1a', '1b', '3c']))
# BiFeO3 = Crystal("BiFeO3", lattice=SGLattice())
# BiFeO3 = Crystal(
#    "BiFeO3",
#    lattice.PerovskiteTypeRhombohedral(elements.Bi, elements.Fe, elements.O,
#                                       3.965, 89.3))
FeO = Crystal("FeO", lattice=SGLattice(225, 4.332, atoms=[e.Fe, e.O],
                               pos=['4a', '4b']))
CoO = Crystal("CoO", lattice=SGLattice(225, 4.214, atoms=[e.Co, e.O],
                               pos=['4a', '4b']))
Fe3O4 = Crystal("Fe3O4", lattice=SGLattice('227:2', 8.3958, atoms=[e.Fe, e.Fe, e.O],
                                   pos=['8a', '16d', ('32e', 0.255)]))
Co3O4 = Crystal("Co3O4", lattice=SGLattice('227:2', 8.0821, atoms=[e.Co, e.Co, e.O],
                                   pos=['8a', '16d', ('32e', 0.255)]))
FeRh = Crystal("FeRh", lattice=SGLattice(221, 2.993, atoms=[e.Fe, e.Rh],
                                 pos=['1a', '1b']))
Ir20Mn80 = Crystal("Ir20Mn80", lattice=SGLattice(225, 3.780, atoms=[e.Ir, e.Mn],
                                         pos=['4a', '4a'], occ=[0.2, 0.8]))
CoFe = Crystal("CoFe", lattice=SGLattice(221, 2.8508, atoms=[e.Co, e.Fe],
                                 pos=['1a', '1b']))
LaB6 = Crystal("LaB6", lattice=SGLattice(221, 4.15692, atoms=[e.La, e.B],
                                 pos=['1a', ('6f', 0.19750)]))
Al2O3 = Crystal("Al2O3", lattice=SGLattice('167:H', 4.7602, 12.9933,
                atoms=[e.Al, e.O], pos=[('12c', 0.35216), ('18e', 0.30624)]))

# materials defined from cif file
try:
    CuMnAs = Crystal.fromCIF(os.path.join(__path__[0], "data",
                                          "CuMnAs_tetragonal.cif"))
except:
    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.materials: Warning: import of CIF "
              "file based material failed")


# Alloys with special properties
class SiGe(CubicAlloy):

    def __init__(self, x):
        """
        Si_{1-x} Ge_x cubic compound
        """
        super(SiGe, self).__init__(Si, Ge, x)

    @staticmethod
    def lattice_const_AB(latA, latB, x):
        """
        method to calculate the lattice parameter of the SiGe alloy with
        composition Si_{1-x}Ge_x
        """
        return latA + (0.2 * x + 0.027 * x ** 2)


class AlGaAs(CubicAlloy):

    def __init__(self, x):
        """
        Al_{1-x} Ga_x As cubic compound
        """
        super(AlGaAs, self).__init__(AlAs, GaAs, x)
