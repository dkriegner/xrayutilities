# This file is part of xrayutilities, distributed under the GNU GPL v2 or later.
# Copyright (C) 2026 CAOShurong

import multiprocessing
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import xrayutilities as xu
from xrayutilities.simpack.powder import manager


class TestPowderModelMode(unittest.TestCase):
    def test_local_script_does_not_require_main_guard(self):
        source = textwrap.dedent(
            """
            import multiprocessing

            import numpy
            import xrayutilities as xu

            multiprocessing.set_start_method("spawn", force=True)
            powder = xu.simpack.Powder(xu.materials.Si, 1)
            model = xu.simpack.PowderModel(powder, tt_cutoff=40, mode="local")
            try:
                intensity = model.simulate(numpy.linspace(15, 40, 300))
                assert numpy.isfinite(intensity).all() and intensity.max() > 0
                assert not multiprocessing.active_children()
            finally:
                model.close()
            print("local mode completed")
            """
        )
        with tempfile.TemporaryDirectory() as path:
            script = Path(path) / "powder_script.py"
            script.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local mode completed", result.stdout)

    def test_enable_workers_after_a_local_calculation(self):
        powder = xu.simpack.Powder(xu.materials.Si, 1)
        angles = numpy.linspace(15, 40, 300)
        before = set(multiprocessing.active_children())
        with patch.object(xu.config, "NTHREADS", 2):
            model = xu.simpack.PowderModel(
                powder, wl=0.65, tt_cutoff=40, mode="local"
            )
            try:
                local = model.simulate(angles)
                self.assertEqual(
                    set(multiprocessing.active_children()), before
                )
                multi = model.simulate(angles, mode="multi")
                workers = set(multiprocessing.active_children()) - before
                self.assertEqual(len(workers), 2)
                numpy.testing.assert_allclose(
                    multi, local, rtol=1e-12, atol=1e-14
                )
                numpy.testing.assert_array_equal(model.simulate(angles), local)
            finally:
                model.close()
                model.close()
        self.assertEqual(set(multiprocessing.active_children()), before)

    def test_recover_from_an_interrupted_worker_start(self):
        powder = xu.simpack.Powder(xu.materials.Si, 1)
        angles = numpy.linspace(15, 40, 300)
        original_start = manager.start
        active = set(multiprocessing.active_children())
        started = []

        def fail_second_start(instance):
            if started:
                raise OSError("manager startup failed")
            original_start(instance)
            started.append(instance)

        with patch.object(xu.config, "NTHREADS", 2):
            model = xu.simpack.PowderModel(powder, tt_cutoff=40, mode="local")
            try:
                local = model.simulate(angles)
                with (
                    patch.object(manager, "start", fail_second_start),
                    self.assertRaisesRegex(OSError, "manager startup failed"),
                ):
                    model.simulate(angles, mode="multi")
                self.assertEqual(
                    set(multiprocessing.active_children()), active
                )
                numpy.testing.assert_array_equal(model.simulate(angles), local)
                numpy.testing.assert_allclose(
                    model.simulate(angles, mode="multi"),
                    local,
                    rtol=1e-12,
                    atol=1e-14,
                )
            finally:
                model.close()
        self.assertEqual(set(multiprocessing.active_children()), active)
