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

import xrayutilities as xu

# very often its useful/needed to get oriented in reciprocal space. The
# following convenience function can visualize the reciprocal space position of
# Bragg peaks and limitations of coplanar diffraction geometry (Laue zoens). If
# you use an interactive matplotlib backend you get the Miller indices of the
# Bragg peak upon mouse movement over the peak and its Bragg angles are printed
# to the console upon clicking. The printed angles correspond to the output of
# Q2Ang of the experimental class. All this is shown below for a substrate/film
# system.

Si = xu.materials.Si
Ge = xu.materials.Ge

geom = xu.HXRD(Si.Q(1, 1, -2), Si.Q(1, 1, 1))

ax, h = xu.materials.show_reciprocal_space_plane(Si, geom, ttmax=160)
ax, h2 = xu.materials.show_reciprocal_space_plane(Ge, geom, ax=ax)

ax.figure.show()

# with default settings only Bragg peaks close to the coplanar plane are shown.
# This can be changed by optional settings 'projection' and 'maxqout'
