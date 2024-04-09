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
# Copyright (C) 2010-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities derives its own exceptions which are raised
upon wrong input when calling one of xrayutilities functions.
none of the pre-defined exceptions is made for that purpose.
"""

# other used Exception should mainly be the python built-in exceptions
#
# * TypeError
#   Raised when an operation or function is applied to an object of
#   inappropriate type
#
# * ValueError
#   Raised when a operation or function receives an argument that
#   has the right type but an inappropriate value
#
# * UserWarning
#   Base class for warnings generated by user code


class InputError(Exception):
    """
    Exception raised for errors in the input.
    Either wrong datatype not handled by TypeError or missing mandatory
    keyword argument (Note that the obligation to give keyword arguments
    might depend on the value of the arguments itself)
    """


class UsageError(Exception):
    """
    Exception raised when a wrong use of an object is detected.
    """
