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
# Copyright (C) 2016-2020 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
simulation subpackage of xrayutilities.

This package provides possibilities to simulate X-ray diffraction and
reflectivity curves of thin film samples. It could be extended for more
general use in future if there is demand for that.

In addition it provides a fitting routine for reflectivity data which is based
on lmfit.
"""

from .darwin_theory import (DarwinModel, DarwinModelAlGaAs001,
                            DarwinModelAlloy, DarwinModelGaInAs001,
                            DarwinModelSiGe001, GradedBuffer)
from .fit import FitModel
from .helpers import coplanar_alphai, get_qz
from .models import (DiffuseReflectivityModel, DynamicalModel,
                     DynamicalReflectivityModel, KinematicalModel,
                     KinematicalMultiBeamModel, LayerModel, Model,
                     SimpleDynamicalCoplanarModel, SpecularReflectivityModel,
                     effectiveDensitySlicing)
from .mosaicity import mosaic_analytic
from .powder import FP_profile, PowderDiffraction
from .powdermodel import PowderModel, Rietveld_error_metrics, plot_powder
from .smaterials import (CrystalStack, GradedLayerStack, Layer, LayerStack,
                         MaterialList, Powder, PowderList,
                         PseudomorphicStack001, PseudomorphicStack111,
                         SMaterial)
