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
# Copyright (c) 2016-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import glob
import os.path

from .. import config
from .helper import xu_h5open


class FileDirectory:
    """
    Parses a directory for files, which can be stored to a HDF5 file for
    further usage. The file parser is given to the constructor and must provide
    a Save2HDF5 method.
    """

    def __init__(self, datapath, ext, parser, **keyargs):
        """

        Parameters
        ----------
        datapath :  str
            directory of the files
        ext :       str
            extension of the files in the datapath
        parser :    class
            Parser class for the data files.
        keyargs :   dict
            further keyword arguments are passed to the constructor of the
            parser
        """

        self.datapath = os.path.normpath(datapath)
        self.extension = ext
        self.parser = parser

        # create list of files to read
        self.files = glob.glob(
            os.path.join(self.datapath, f"*.{self.extension}")
        )

        if not self.files:
            print(f"XU.io.FileDirectory: no file found in {self.datapath}")
            return

        if config.VERBOSITY >= config.INFO_ALL:
            print(
                "XU.io.FileDirectory: %d files found in %s"
                % (len(self.files), self.datapath)
            )

        self.init_keyargs = keyargs

    def Save2HDF5(self, h5f, group="", comp=True):
        """
        Saves the data stored in the found files in the specified directory in
        a HDF5 file as a HDF5 arrays in a subgroup.  By default the data is
        stored in a group given by the foldername - this can be changed by
        passing the name of a target group or a path to the target group via
        the "group" keyword argument.

        Parameters
        ----------
        h5f :   file-handle or str
            a HDF5 file object or name
        group : str, optional
            group where to store the data (defaults to pathname if group is
            empty string)
        comp :  bool, optional
            activate compression - true by default
        """
        with xu_h5open(h5f, "a") as h5:
            if isinstance(group, str):
                if group == "":
                    group = os.path.split(self.datapath)[1]
                g = h5.get(group)
                if not g:
                    g = h5.create_group(group)
            else:
                g = group

            for infile in self.files:
                # read EDFFile and save to hdf5
                filename = os.path.split(infile)[1]
                e = self.parser(
                    filename, path=self.datapath, **self.init_keyargs
                )
                e.Save2HDF5(h5, group=g, comp=comp)
