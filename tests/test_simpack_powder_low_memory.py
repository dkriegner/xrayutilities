# This file is part of xrayutilities, distributed under the GNU GPL v2 or later.
# Copyright (C) 2026 CAOShurong

import gc
import unittest
import weakref
from unittest.mock import patch

import numpy
import xrayutilities as xu
from xrayutilities.simpack.powder import FP_profile


class TestPowderLowMemory(unittest.TestCase):
    def make_model(self, **kwargs):
        material = xu.simpack.Powder(xu.materials.Si, 1)
        with patch.object(xu.config, "NTHREADS", 2):
            model = xu.simpack.PowderModel(material, tt_cutoff=60, **kwargs)
        self.addCleanup(model.close)
        return model

    def test_stock_profile_preserves_patterns_without_retaining_windows(self):
        cached = self.make_model()
        uncached = self.make_model(low_memory=True)

        for npoints, displacement in ((181, 0), (361, 0), (361, 4e-5)):
            angles = numpy.linspace(25, 60, npoints)
            for model in (cached, uncached):
                model.set_parameters(
                    {"displacement": {"specimen_displacement": displacement}}
                )
                model.set_lmfit_parameters(model.create_fitparameters())

            expected = cached.simulate(angles, mode="local")
            self.assertGreater(expected.max(), 0)
            for mode in ("local", "multi"):
                with self.subTest(
                    npoints=npoints, displacement=displacement, mode=mode
                ):
                    result = uncached.simulate(angles, mode=mode)
                    numpy.testing.assert_allclose(result, expected)

        cached_profiles = [
            line["conv"] for line in cached.pdiff[0].data.values()
        ]
        uncached_profiles = [
            line["conv"] for line in uncached.pdiff[0].data.values()
        ]
        self.assertTrue(
            any(hasattr(profile, "_rb1") for profile in cached_profiles)
        )
        self.assertFalse(
            any(hasattr(profile, "_rb1") for profile in uncached_profiles)
        )

    def test_low_memory_rejects_custom_profiles(self):
        class UserProfile(FP_profile):
            pass

        with self.assertRaisesRegex(ValueError, "default FP_profile"):
            self.make_model(low_memory=True, fpclass=UserProfile)

    def test_temporary_profiles_are_released_without_cyclic_gc(self):
        model = self.make_model(low_memory=True)
        original = FP_profile.set_window
        profiles = []

        def record_profile(profile, *args, **kwargs):
            profiles.append(weakref.ref(profile))
            return original(profile, *args, **kwargs)

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.object(FP_profile, "set_window", record_profile):
                result = model.simulate(
                    numpy.linspace(25, 60, 181), mode="local"
                )
            self.assertGreater(result.max(), 0)
            self.assertTrue(profiles)
            self.assertTrue(all(profile() is None for profile in profiles))
        finally:
            if was_enabled:
                gc.enable()
