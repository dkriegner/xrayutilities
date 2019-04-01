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
# Copyright (C) 2009-2010 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

from .cbf import CBFDirectory, CBFFile
from .desy_tty08 import gettty08_scan, tty08File
from .edf import EDFDirectory, EDFFile
from .fastscan import FastScan, FastScanCCD, FastScanSeries
from .helper import xu_h5open, xu_open
from .ill_numor import numor_scan, numorFile
from .imagereader import (ImageReader, PerkinElmer, Pilatus100K, RoperCCD,
                          TIFFRead, get_tiff)
from .panalytical_xml import XRDMLFile, getxrdml_map, getxrdml_scan
from .pdcif import pdCIF, pdESG
from .rigaku_ras import RASFile, RASScan, getras_scan
# parser for the alignment log file of the rotating anode
from .rotanode_alignment import RA_Alignment
from .seifert import SeifertMultiScan, SeifertScan, getSeifert_map
# for backward compatibility import also as old name
from .spec import SPECFile, SPECLog, SPECScan
from .spec import geth5_scan as geth5_map
from .spec import getspec_scan
from .spectra import SPECTRAFile, geth5_spectra_map
