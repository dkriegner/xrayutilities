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
"""
simulation subpackage of xrayutilities.

This package provides possibilities to simulate X-ray diffraction and
reflectivity curves of thin film samples. It could be extended for more
general use in future if there is demand for that.

In addition it provides a fitting routine for reflectivity data which is based
on lmfit.
"""

from .smaterials import SMaterial, MaterialList
from .smaterials import Powder, PowderList
from .smaterials import Layer, LayerStack
from .smaterials import CrystalStack, GradedLayerStack
from .smaterials import PseudomorphicStack001
from .smaterials import PseudomorphicStack111

from .models import Model, LayerModel
from .models import KinematicalModel
from .models import KinematicalMultiBeamModel
from .models import SpecularReflectivityModel
from .models import SimpleDynamicalCoplanarModel
from .models import DynamicalModel
from .models import DynamicalReflectivityModel

from .darwin_theory import GradedBuffer
from .darwin_theory import DarwinModel
from .darwin_theory import DarwinModelSiGe001
from .darwin_theory import DarwinModelGaInAs001
from .darwin_theory import DarwinModelAlGaAs001

from .fit import fit_xrr

from .helpers import coplanar_alphai
from .helpers import get_qz

from .powder import FP_profile
from .powder import PowderDiffraction

from .powdermodel import PowderModel
from .powdermodel import Rietveld_error_metrics
from .powdermodel import plot_powder
