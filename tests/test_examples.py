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

import os
import subprocess
import sys
import tempfile
import unittest


scriptdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                         'examples')
scriptfiles = [
    'simpack_powdermodel.py',
    'simpack_xrd_AlGaAs.py',
    'simpack_xrd_Darwin_AlGaAs.py',
    'simpack_xrd_InAs_fitting.py',
    'simpack_xrd_SiGe111.py',
    'simpack_xrd_SiGe_asymmmetric.py',
    'simpack_xrd_SiGe.py',
    'simpack_xrd_SiGe_superlattice.py',
    'simpack_xrr_diffuse.py',
    'simpack_xrr_matrixmethod.py',
    'simpack_xrr_SiO2_Ru_CoFe_IrMn_Al2O3.py',
    'xrayutilities_angular2hkl_conversion.py',
    # 'xrayutilities_ccd_parameter.py',  # data file not included
    'xrayutilities_components_of_the_structure_factor.py',
    'xrayutilities_define_material.py',
    'xrayutilities_energy_dependent_structure_factor.py',
    # 'xrayutilities_example_plot_3D_ESRF_ID01.py',  # data file not included
    'xrayutilities_experiment_angle_calculation.py',
    'xrayutilities_experiment_kappa.py',
    'xrayutilities_experiment_Powder_example_Iron.py',
    # 'xrayutilities_export_data2vtk.py',  # needs vtk + data
    'xrayutilities_fuzzygridding.py',
    'xrayutilities_hotpixelkill_variant.py',
    'xrayutilities_id01_functions.py',
    'xrayutilities_io_cif_parser_bi2te3.py',
    'xrayutilities_io_cif_parser.py',
    # 'xrayutilities_io_pdcif_plot.py',  # data file not included
    # 'xrayutilities_kmap_example_ESRF.py',  # data file not included
    # 'xrayutilities_linear_detector_parameters.py',  # data file not included
    'xrayutilities_materials_Alloy_contentcalc.py',
    'xrayutilities_math_fitting.py',
    'xrayutilities_orientation_matrix.py',
    'xrayutilities_peak_angles_beamtime.py',
    # 'xrayutilities_polefigure.py',  # basemap needed
    'xrayutilities_q2ang_general.py',
    'xrayutilities_read_panalytical.py',
    # 'xrayutilities_read_seifert.py',  # data file not included
    'xrayutilities_read_spec.py',
    'xrayutilities_reflection_strength.py',
    'xrayutilities_show_reciprocal_space_plane.py',
]


class TestExampleScriptsMeta(type):
    def __new__(mcs, name, bases, dict):
        def test_generator(scriptname):
            def test(self):
                with tempfile.TemporaryFile(mode='w') as fid:
                    env = os.environ.copy()
                    env['MPLBACKEND'] = 'agg'
                    cmd = [sys.executable, scriptname]
                    subprocess.run(cmd, env=env, cwd=scriptdir, stdout=fid,
                                   check=True)
            return test

        for sf in scriptfiles:
            test_name = 'test_%s' % os.path.splitext(sf)[0]
            test = test_generator(sf)
            dict[test_name] = test
        return type.__new__(mcs, name, bases, dict)


@unittest.skipIf(sys.version_info < (3, 5),
                 "needs subprocess.run -> python 3.5 or newer")
class TestExampleScripts(unittest.TestCase, metaclass=TestExampleScriptsMeta):
    pass


if __name__ == '__main__':
    unittest.main()
