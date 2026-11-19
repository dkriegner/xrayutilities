# This file is part of xrayutilities.
#
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 CAOShurong

import numpy
import pytest
import xrayutilities as xu


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot

    yield pyplot
    pyplot.close("all")


def test_inplane_plot_silicon_001(pyplot, capsys):
    from matplotlib.backend_bases import MouseEvent

    si = xu.materials.Si
    gid = xu.GID(si.Q(1, -1, 0), si.Q(0, 0, 1), en=8047.8)
    ax, points = xu.materials.show_reciprocal_space_plane(
        si, gid, ttmax=65, plane="inplane"
    )

    offsets = numpy.asarray(points.get_offsets())
    q220 = 2 * numpy.pi / si.lattice.a * numpy.sqrt(8)
    expected = [(-q220, 0), (0, -q220), (0, q220), (q220, 0)]
    sort_order = numpy.lexsort((offsets[:, 1], offsets[:, 0].round(8)))
    numpy.testing.assert_allclose(offsets[sort_order], expected, atol=1e-12)
    assert r"$Q_x$" in ax.get_xlabel()
    assert r"$Q_y$" in ax.get_ylabel()
    assert ax.get_ylim()[0] == -ax.get_ylim()[1]
    assert not ax.patches

    # This reflection lies along the experiment's positive qy axis.
    ax.figure.canvas.draw()
    xy = ax.transData.transform([0, q220])
    hover = MouseEvent("motion_notify_event", ax.figure.canvas, *xy)
    ax.figure.canvas.callbacks.process(hover.name, hover)
    assert ax.texts[-1].get_visible()
    assert "(2, -2, 0)" in ax.texts[-1].get_text()
    click = MouseEvent("button_press_event", ax.figure.canvas, *xy, button=1)
    ax.figure.canvas.callbacks.process(click.name, click)
    report = capsys.readouterr().out
    assert "(2, -2, 0)" in report
    assert str(gid.Q2Ang(si.Q(2, -2, 0))) in report

    hxrd = xu.HXRD(si.Q(1, -1, 0), si.Q(0, 0, 1), en=8047.8)
    overlay_ax, overlay = xu.materials.show_reciprocal_space_plane(
        si, hxrd, ttmax=65, plane="inplane", ax=ax, label="HXRD"
    )
    assert overlay_ax is ax
    numpy.testing.assert_array_equal(offsets, overlay.get_offsets())


def test_default_is_explicit_coplanar_plot(pyplot):
    si = xu.materials.Si
    exp = xu.HXRD(si.Q(1, 1, -2), si.Q(1, 1, 1), en=8047.8)
    _, default = xu.materials.show_reciprocal_space_plane(si, exp, ttmax=65)
    ax, explicit = xu.materials.show_reciprocal_space_plane(
        si, exp, ttmax=65, plane="coplanar"
    )

    numpy.testing.assert_array_equal(
        default.get_offsets(), explicit.get_offsets()
    )
    numpy.testing.assert_array_equal(default.get_sizes(), explicit.get_sizes())
    assert len(ax.patches) == 5
    assert ax.get_ylim()[0] > -ax.get_ylim()[1]


def test_plane_options(pyplot):
    si = xu.materials.Si
    exp = xu.HXRD(si.Q(1, -1, 0), si.Q(0, 0, 1), en=8047.8)
    with pytest.raises(ValueError, match="plane"):
        xu.materials.show_reciprocal_space_plane(si, exp, plane="invalid")
    with pytest.raises(ValueError, match="projection"):
        xu.materials.show_reciprocal_space_plane(
            si, exp, plane="inplane", projection="polar"
        )


def test_gid_click_reports_out_of_plane_peak(pyplot, capsys):
    from matplotlib.backend_bases import MouseEvent

    si = xu.materials.Si
    gid = xu.GID(si.Q(1, -1, 0), si.Q(0, 0, 1), en=8047.8)
    ax, _ = xu.materials.show_reciprocal_space_plane(
        si, gid, ttmax=65, plane="inplane", maxqout=0.5
    )

    # A wide slice includes the 111 spots, outside the GID angular model.
    q111 = si.Q(1, 1, 1)
    x, y = (q111 @ gid.scatplane, q111 @ gid.idir)
    ax.figure.canvas.draw()
    xy = ax.transData.transform([x, y])
    click = MouseEvent("button_press_event", ax.figure.canvas, *xy, button=1)
    ax.figure.canvas.callbacks.process(click.name, click)
    assert (
        "Reflection not reachable in GID geometry" in capsys.readouterr().out
    )
