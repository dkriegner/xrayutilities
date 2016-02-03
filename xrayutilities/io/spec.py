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
# Copyright (C) 2009-2015 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
a class for observing a SPEC data file

Motivation:

SPEC files can become quite large. Therefore, subsequently reading
the entire file to extract a single scan is a quite cumbersome procedure.
This module is a proof of concept code to write a file observer starting
a reread of the file starting from a stored offset (last known scan position)
"""

import re
import os.path
import time
import warnings

import numpy

# relative imports from xrayutilities
from .helper import xu_open, xu_h5open
from .. import config
from ..exception import InputError

# define some uesfull regular expressions
SPEC_time_format = re.compile(r"\d\d:\d\d:\d\d")
SPEC_multi_blank = re.compile(r"\s+")
SPEC_multi_blank2 = re.compile(r"\s\s+")
# denotes a numeric value
SPEC_num_value = re.compile(r"[+-]*\d*\.*\d*e*[+-]*\d+")
SPEC_dataline = re.compile(r"^[+-]*\d.*")

SPEC_scan = re.compile(r"^#S")
SPEC_initmoponames = re.compile(r"#O\d+")
SPEC_initmopopos = re.compile(r"#P\d+")
SPEC_datetime = re.compile(r"^#D")
SPEC_exptime = re.compile(r"^#T")
SPEC_nofcols = re.compile(r"^#N")
SPEC_colnames = re.compile(r"^#L")
SPEC_MCAFormat = re.compile(r"^#@MCA")
SPEC_MCAChannels = re.compile(r"^#@CHANN")
SPEC_headerline = re.compile(r"^#")
SPEC_scanbroken = re.compile(r"#C[a-zA-Z0-9: .]*Scan aborted")
SPEC_scanresumed = re.compile(r"#C[a-zA-Z0-9: .]*Scan resumed")
SPEC_commentline = re.compile(r"#C")
SPEC_newheader = re.compile(r"^#E")
SPEC_errorbm20 = re.compile(r"^MI:")
scan_status_flags = ["OK", "NODATA", "ABORTED", "CORRUPTED"]


def makeNaturalName(name):
    ret = name.replace(" ", "_")
    ret = ret.replace("-", "_")
    ret = ret.replace(".", "_")
    return ret


class SPECScan(object):
    """
    Represents a single SPEC scan. This class is usually not called by the
    user directly but used via the SPECFile class.
    """

    def __init__(self, name, scannr, command, date, time, itime, colnames,
                 hoffset, doffset, fname, imopnames, imopvalues, scan_status):
        """
        Constructor for the SPECScan class.

        required arguments:
         name ............. name of the scan
         scannr ........... Number of the scan in the specfile
         command .......... command used to write the scan
         date ............. starting date of the scan
         time ............. starting time of the scan
         itime ............ integration time
         colnames ......... list of names of the data columns
         hoffset .......... file byte offset to the header of the scan
         doffset .......... file byte offset to the data section of the scan
         fname ............ file name of the SPEC file the scan belongs to
         imopnames ........ motor names for the initial motor positions array
         imopvalues ....... intial motor positions array
         scan_status ...... is one of the values
        """
        self.name = name  # name of the scan
        self.nr = scannr  # number of the scan
        self.command = command  # command used to record the data
        self.date = date  # date the command has been sent
        self.time = time  # time the command has been sent
        self.colnames = colnames  # list with column names
        self.hoffset = hoffset  # file offset where the header data starts
        self.doffset = doffset  # file offset where the data section starts
        self.fname = fname  # full file name of the file holding the data
        # flag to force resave to hdf5 file in Save2HDF5()
        self.ischanged = True
        self.header = []
        self.fid = None

        if scan_status in scan_status_flags:
            self.scan_status = scan_status
        else:
            self.scan_status = "CORRUPTED"
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.spec.SPECScan: unknown scan status flag - "
                      "set to CORRUPTED")

        # setup the initial motor positions dictionary - set the motor names
        # dictionary holding the initial motor positions
        self.init_motor_pos = {}
        if len(imopnames) == len(imopvalues):
            for i in range(len(imopnames)):
                self.init_motor_pos[
                    "INIT_MOPO_" +
                    makeNaturalName(imopnames[i])] = float(imopvalues[i])
        else:
            print("XU.io.spec.SPECScan: Warning: incorrect number of "
                  "initial motor positions")
            if config.VERBOSITY >= config.INFO_ALL:
                print(imopnames)
                print(imopvalues)
            # ASSUME ORDER DID NOT CHANGE!! (which might be wrong)
            # in fact this is sign for a broken spec file
            # number of initial motor positions should not change without new
            # file header!
            for i in range(min(len(imopnames), len(imopvalues))):
                self.init_motor_pos[
                    "INIT_MOPO_" +
                    makeNaturalName(imopnames[i])] = float(imopvalues[i])
            # read the rest of the positions into dummy INIT_MOPO__NONAME__%03d
            for i in range(len(imopnames), len(imopvalues)):
                self.init_motor_pos["INIT_MOPO___NONAME__%03d" % (i)] = \
                    float(imopvalues[i])

        # some additional attributes for the MCA data
        # False if scan contains no MCA data, True otherwise
        self.has_mca = False
        self.mca_column_format = 0  # number of columns used to save MCA data
        self.mca_channels = 0  # number of channels stored from the MCA
        self.mca_nof_lines = 0  # number of lines used to store MCA data
        self.mca_start_channel = 0  # first channel of the MCA that is stored
        self.mca_stop_channel = 0  # last channel of the MCA that is stored

        # a numpy record array holding the data - this is set using by the
        # ReadData method.
        self.data = None

        # check for duplicate values in column names
        for i in range(len(self.colnames)):
            name = self.colnames[i]
            cnt = self.colnames.count(name)
            if cnt > 1:
                # have multiple entries
                cnt = 1
                for j in range(self.colnames.index(name) + 1,
                               len(self.colnames)):
                    if self.colnames[j] == name:
                        self.colnames[j] = name + "_%i" % cnt
                    cnt += 1

    def SetMCAParams(self, mca_column_format, mca_channels,
                     mca_start, mca_stop):
        """
        Set the parameters used to save the MCA data to the file. This method
        calculates the number of lines used to store the MCA data from the
        number of columns and the

        required input aguments:
         mca_column_format ....... number of columns used to save the data
         mca_channels ............ number of MCA channels stored
         mca_start ............... first channel that is stored
         mca_stop ................ last channel that is stored
        """
        self.has_mca = True
        self.mca_column_format = mca_column_format
        self.mca_channels = mca_channels
        self.mca_start_channel = mca_start
        self.mca_stop_channel = mca_stop

        # calculate the number of lines per data point for the mca
        self.mca_nof_lines = int(mca_channels / mca_column_format)
        if mca_channels % mca_column_format != 0:
            # some additional values have to be read
            self.mca_nof_lines = self.mca_nof_lines + 1

        if config.VERBOSITY >= config.DEBUG:
            print("XU.io.SPECScan.SetMCAParams: number of channels: %d"
                  % self.mca_channels)
            print("XU.io.SPECScan.SetMCAParams: number of columns: %d"
                  % self.mca_column_format)
            print("XU.io.SPECScan.SetMCAParams: number of lines to read "
                  "for MCA: %d" % self.mca_nof_lines)

    def __str__(self):
        # build a proper string to print the scan information
        str_rep = "|%4i|" % (self.nr)
        str_rep = str_rep + "%50s|%10s|%10s|" % (self.command, self.time,
                                                 self.date)

        if self.has_mca:
            str_rep = str_rep + "MCA: %5i" % (self.mca_channels)

        str_rep = str_rep + "\n"
        return str_rep

    def ClearData(self):
        """
        Delete the data stored in a scan after it is no longer
        used.
        """

        self.__delattr__("data")
        self.data = None

    def ReadData(self):
        """
        Set the data attribute of the scan class.
        """

        if self.scan_status == "NODATA":
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.SPECScan.ReadData: %s has been aborted - "
                      "no data available!" % self.name)
            self.data = None
            return None

        if not self.has_mca:
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.SPECScan.ReadData: scan %d contains no MCA data"
                      % self.nr)

        with xu_open(self.fname) as self.fid:
            # read header lines
            self.fid.seek(self.hoffset, 0)
            self.header = []
            while self.fid.tell() < self.doffset:
                line = self.fid.readline().decode('ascii')
                self.header.append(line.strip())

            self.fid.seek(self.doffset, 0)

            # create dictionary to hold the data
            if self.has_mca:
                type_desc = {"names": self.colnames + ["MCA"],
                             "formats": len(self.colnames) * [numpy.float32] +
                             [(numpy.uint32, self.mca_channels)]}
            else:
                type_desc = {"names": self.colnames,
                             "formats": len(self.colnames) * [numpy.float32]}

            if config.VERBOSITY >= config.DEBUG:
                print("xu.io.SPECScan.ReadData: type descriptor: %s"
                      % (repr(type_desc)))

            record_list = []  # from this list the record array while be built

            mca_counter = 0
            scan_aborted_flag = False

            for line in self.fid:
                line = line.decode('ascii')
                line = line.strip()
                if not line:
                    continue

                # Bugfix for ESRF/BM20 data
                # the problem is that they store messages from automatic
                # absorbers in the SPEC file - need to handle this
                t = re.compile(r"^#C .* filter factor.*")
                if t.match(line):
                    continue

                # check if scan is broken
                if (SPEC_scanbroken.findall(line) != [] or
                        scan_aborted_flag):
                    # need to check next line(s) to know if scan is resumed
                    # read until end of comment block or end of file
                    if not scan_aborted_flag:
                        scan_aborted_flag = True
                        self.scan_status = "ABORTED"
                        if config.VERBOSITY >= config.INFO_ALL:
                            print("XU.io.SPECScan.ReadData: %s aborted"
                                  % self.name)
                        continue
                    elif SPEC_scanresumed.match(line):
                        self.scan_status = "OK"
                        scan_aborted_flag = False
                        if config.VERBOSITY >= config.INFO_ALL:
                            print("XU.io.SPECScan.ReadData: %s resumed"
                                  % self.name)
                        continue
                    elif SPEC_commentline.match(line):
                        continue
                    elif SPEC_errorbm20.match(line):
                        print(line)
                        continue
                    else:
                        break

                if SPEC_headerline.match(line) or \
                   SPEC_commentline.match(line):
                    break

                if mca_counter == 0:
                    # the line is a scalar data line
                    line_list = SPEC_num_value.findall(line)
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECScan.ReadData: %s" % line)
                        print("XU.io.SPECScan.ReadData: read scalar values %s"
                              % repr(line_list))
                    # convert strings to numbers
                    for i in range(len(line_list)):
                        line_list[i] = float(line_list[i])

                    # increment the MCA counter if MCA data is stored
                    if self.has_mca:
                        mca_counter = mca_counter + 1
                        # create a temporary list for the mca data
                        mca_tmp_list = []
                    else:
                        record_list.append(line_list)
                else:
                    # reading and MCA spectrum
                    tmp_list = SPEC_num_value.findall(line)
                    for x in tmp_list:
                        mca_tmp_list.append(float(x))

                    # increment MCA counter
                    mca_counter = mca_counter + 1
                    # if mca_counter exceeds the number of lines used to store
                    # MCA data: append everything to the record list
                    if mca_counter > self.mca_nof_lines:
                        record_list.append(line_list + [mca_tmp_list])
                        mca_counter = 0

            # convert the lists in the data dictionary to numpy arrays
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.SPECScan.ReadData: %s: %d %d %d"
                      % (self.name, len(record_list),
                         len(record_list[0]), len(type_desc["names"])))
            if len(record_list[0]) == len(type_desc["names"]):
                try:
                    self.data = numpy.rec.fromrecords(record_list,
                                                      dtype=type_desc)
                except ValueError:
                    self.scan_status = 'NODATA'
                    print("XU.io.SPECScan.ReadData: %s exception while "
                          "parsing data" % self.name)
            else:
                self.scan_status = 'NODATA'

    def plot(self, *args, **keyargs):
        """
        Plot scan data to a matplotlib figure. If newfig=True a new
        figure instance will be created. If logy=True (default is False)
        the y-axis will be plotted with a logarithmic scale.

        Parameters
        ----------
         *args:  arguments for the plot: first argument is the name of x-value
                 column the following pairs of arguments are the y-value names
                 and plot styles allowed are 3,5,7,... number of arguments

         **keyargs:
          newfig:  if True a new figure instance will be created otherwise an
                   existing one will be used
          logy:    if True a semilogy plot will be done
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.SPECScan: Warning: plot "
                      "functionality not available")
            return

        if "newfig" in keyargs:
            newfig = keyargs["newfig"]
        else:
            newfig = True

        if "logy" in keyargs:
            logy = keyargs["logy"]
        else:
            logy = False

        try:
            xname = args[0]
            xdata = self.data[xname]
        except:
            raise InputError("name of the x-axis is invalid!")

        alist = args[1:]
        leglist = []

        if len(alist) % 2 != 0:
            raise InputError("wrong number of yname/style arguments!")

        if newfig:
            plt.figure()
            plt.subplots_adjust(left=0.08, right=0.95)

        for i in range(0, len(alist), 2):
            yname = alist[i]
            ystyle = alist[i + 1]
            try:
                ydata = self.data[yname]
            except:
                raise InputError("no column with name %s exists!" % yname)
                continue
            if logy:
                plt.semilogy(xdata, ydata, ystyle)
            else:
                plt.plot(xdata, ydata, ystyle)

            leglist.append(yname)

        plt.xlabel("%s" % xname)
        plt.legend(leglist)
        plt.title("scan %i %s\n%s %s"
                  % (self.nr, self.command, self.date, self.time))
        # need to adjust axis limits properly
        lim = plt.axis()
        plt.axis([xdata.min(), xdata.max(), lim[2], lim[3]])

    def Save2HDF5(self, h5f, group="/", title="", optattrs={}, comp=True):
        """
        Save a SPEC scan to an HDF5 file. The method creates a group with the
        name of the scan and stores the data there as a table object with name
        "data". By default the scan group is created under the root group of
        the HDF5 file.  The title of the scan group is ususally the scan
        command. Metadata of the scan are stored as attributes to the scan
        group. Additional custom attributes to the scan group can be passed as
        a dictionary via the optattrs keyword argument.

        input arguments:
         h5f ........ a HDF5 file object or its filename

        optional keyword arguments:
         group ...... name or group object of the HDF5 group where to store the
                      data
         title ...... a string with the title for the data, defaults to the
                      name of scan if empty
         optattrs ... a dictionary with optional attributes to store for the
                      data
         comp ....... activate compression - true by default
        """

        with xu_h5open(h5f, 'a') as h5:
            # check if data object has been already written
            if self.data is None:
                raise InputError("XU.io.SPECScan.Save2HDF5: No data has been"
                                 "read so far - call ReadData method of the "
                                 "scan")
                return None

            # parse keyword arguments:
            if isinstance(group, str):
                rootgroup = h5.get(group)
            else:
                rootgroup = group

            if title != "":
                group_title = title
            else:
                group_title = self.name
            group_title = group_title.replace(".", "_")

            # create the dataset and fill it
            copy_count = 0
            if self.ischanged and group_title in rootgroup:
                del rootgroup[group_title]
            raw_grp_title = group_title
            # if the group already exists the name must be changed and
            # another will be made to create the group.
            while group_title in rootgroup:
                group_title = raw_grp_title + "_%i" % (copy_count)
                copy_count = copy_count + 1
            g = rootgroup.create_group(group_title)

            kwds = {'fletcher32': True}
            if comp:
                kwds['compression'] = 'gzip'

            dset = g.create_dataset("data", data=self.data, **kwds)

            # write attribute data for the scan
            g.attrs['ScanNumber'] = numpy.uint(self.nr)
            g.attrs['Command'] = self.command
            g.attrs['Date'] = self.date
            g.attrs['Time'] = self.time

            # write the initial motor positions as attributes
            for k in self.init_motor_pos.keys():
                g.attrs[k] = numpy.float(self.init_motor_pos[k])

            # if scan contains MCA data write also MCA parameters
            g.attrs['mca_start_channel'] = numpy.uint(self.mca_start_channel)
            g.attrs['mca_stop_channel'] = numpy.uint(self.mca_stop_channel)
            g.attrs['mca_nof_channels'] = numpy.uint(self.mca_channels)

            for k in optattrs.keys():
                g.attrs[k] = optattrs[k]

            h5.flush()


class SPECFile(object):

    """
    This class represents a single SPEC file. The class provides
    methodes for updateing an already opened file which makes it particular
    interesting for interactive use.
    """

    def __init__(self, filename, path=""):
        """
        SPECFile init routine

        Parameters
        ----------
         filename:  filename of the spec file
         path:      path to the specfile (optional)
        """
        self.full_filename = os.path.join(path, filename)
        self.filename = os.path.basename(self.full_filename)

        # list holding scan objects
        self.scan_list = []
        self.fid = None
        self.last_offset = 0

        # initially parse the file
        self.init_motor_names = []  # this list will hold the names of the
        # motors saved in initial motor positions

        self.Parse()

    def __getitem__(self, index):
        """
        function to return the n-th scan in the spec-file.  be aware that
        numbering starts at 0! If scans are missing the relation between the
        given number and the "number" of the returned scan might be not
        trivial.

        See also
        --------
         scanI attributes of the SPECFile object
        """
        return self.scan_list[index]

    def __getattr__(self, name):
        """
        return scanX objects where X stands for the scan number in the SPECFile
        which for this purpose is assumed to be unique. (otherwise the first
        instance of scan number X is returned)
        """
        if name.startswith("scan"):
            index = name[4:]

            try:
                scannr = int(index)
            except:
                raise AttributeError("scannumber needs to be convertable to "
                                     "integer")

            # try to find the scan in the list of scans
            s = None
            for scan in self.scan_list:
                if scan.nr == scannr:
                    s = scan
                    break

            if s is not None:
                return s
            else:
                raise AttributeError("requested scan-number not found")
        else:
            raise AttributeError("FastScan has no attribute '%s'" % name)

    def __len__(self):
        return self.scan_list.__len__()

    def __str__(self):
        ostr = ""
        for i in range(len(self.scan_list)):
            ostr = ostr + "%5i" % (i)
            ostr = ostr + self.scan_list[i].__str__()

        return ostr

    def Save2HDF5(self, h5f, comp=True):
        """
        Save the entire file in an HDF5 file. For that purpose a group is set
        up in the root group of the file with the name of the file without
        extension and leading path.  If the method is called after an previous
        update only the scans not written to the file meanwhile are saved.

        required arguments:
         h5f .................... a HDF5 file object or its filename

        optional keyword arguments:
         comp .................. activate compression - true by default
        """
        with xu_h5open(h5f, 'a') as h5:
            groupname = os.path.splitext(os.path.splitext(self.filename)[0])[0]
            try:
                g = h5.create_group(groupname)
            except ValueError:
                g = h5.get(groupname)

            g.attrs['TITLE'] = "Data of SPEC - File %s" % (self.filename)
            for s in self.scan_list:
                if (((s.name not in g) or s.ischanged) and
                        s.scan_status != "NODATA"):
                    s.ReadData()
                    if s.data is not None:
                        s.Save2HDF5(h5, group=g, comp=comp)
                        s.ClearData()
                        s.ischanged = False

    def Update(self):
        """
        reread the file and add newly added files. The parsing starts at the
        data offset of the last scan gathered during the last parsing run.
        """

        # reparse the SPEC file
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.io.SPECFile.Update: reparsing file for new scans ...")
        # mark last found scan as not saved to force reread
        idx = len(self.scan_list)
        if idx > 0:
            lastscan = self.scan_list[idx - 1]
            lastscan.ischanged = True
        self.Parse()

    def Parse(self):
        """
        Parses the file from the starting at last_offset and adding found scans
        to the scan list.
        """
        with xu_open(self.full_filename) as self.fid:
            # move to the last read position in the file
            self.fid.seek(self.last_offset, 0)
            scan_started = False
            scan_has_mca = False
            # list with the motors from whome the initial
            # position is stored.
            init_motor_values = []

            if config.VERBOSITY >= config.DEBUG:
                print('XU.io.SPECFile: start parsing')

            for line in self.fid:
                linelength = len(line)
                line = line.decode('ascii')
                if config.VERBOSITY >= config.DEBUG:
                    print('parsing line: %s' % line)

                # remove trailing and leading blanks from the read line
                line = line.strip()

                # fill the list with the initial motor names
                if SPEC_newheader.match(line):
                    self.init_motor_names = []

                elif SPEC_initmoponames.match(line):
                    line = SPEC_initmoponames.sub("", line)
                    line = line.strip()
                    self.init_motor_names = self.init_motor_names + \
                        SPEC_multi_blank2.split(line)

                # if the line marks the beginning of a new scan
                elif SPEC_scan.match(line) and not scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found scan")
                    line_list = SPEC_multi_blank.split(line)
                    scannr = int(line_list[1])
                    scancmd = "".join(" " + x + " " for x in line_list[2:])
                    scan_started = True
                    scan_has_mca = False
                    scan_header_offset = self.last_offset
                    scan_status = "OK"
                    # define some necessary variables which could be missing in
                    # the scan header
                    itime = numpy.nan
                    time = ''
                    if config.VERBOSITY >= config.INFO_ALL:
                        print("XU.io.SPECFile.Parse: processing scan nr. %d "
                              "..." % scannr)

                # if the line contains the date and time information
                elif SPEC_datetime.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found date and time")
                    # fetch the time from the line data
                    time = SPEC_time_format.findall(line)[0]
                    line = SPEC_time_format.sub("", line)
                    line = SPEC_datetime.sub("", line)
                    date = SPEC_multi_blank.sub(" ", line)

                # if the line contains the integration time
                elif SPEC_exptime.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found exposure time")
                    itime = float(SPEC_num_value.findall(line)[0])
                # read the initial motor positions
                elif SPEC_initmopopos.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found initial motor "
                              "positions")
                    line = SPEC_initmopopos.sub("", line)
                    line = line.strip()
                    line_list = SPEC_multi_blank.split(line)
                    # sometimes initial motor position are simply empty and
                    # this should not lead to an error
                    try:
                        for value in line_list:
                            init_motor_values.append(float(value))
                    except:
                        pass

                # if the line contains the number of colunmns
                elif SPEC_nofcols.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found number of columns")
                    line = SPEC_nofcols.sub("", line)
                    line = line.strip()
                    nofcols = int(line)

                # if the line contains the column names
                elif SPEC_colnames.match(line) and scan_started:
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found column names")
                    line = SPEC_colnames.sub("", line)
                    line = line.strip()
                    col_names = SPEC_multi_blank.split(line)

                    # this is a fix in the case that blanks are allowed in
                    # motor and detector names (only a single balanks is
                    # supported meanwhile)
                    if len(col_names) > nofcols:
                        col_names = SPEC_multi_blank2.split(line)

                elif SPEC_MCAFormat.match(line) and scan_started:
                    mca_col_number = int(SPEC_num_value.findall(
                                         line)[0])
                    scan_has_mca = True

                elif SPEC_MCAChannels.match(line) and scan_started:
                    line_list = SPEC_num_value.findall(line)
                    mca_channels = int(line_list[0])
                    mca_start = int(line_list[1])
                    mca_stop = int(line_list[2])

                elif (SPEC_scanbroken.findall(line) != [] and
                      scan_started):
                    # this is the case when a scan is broken and no data has
                    # been written, but nevertheless a comment is in the file
                    # that tells us that the scan was aborted
                    try:
                        s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                     date, time, itime, col_names,
                                     scan_header_offset, scan_data_offset,
                                     self.full_filename, self.init_motor_names,
                                     init_motor_values, "NODATA")
                    except:
                        scan_data_offset = self.last_offset
                        s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                     date, time, itime, col_names,
                                     scan_header_offset, scan_data_offset,
                                     self.full_filename, self.init_motor_names,
                                     init_motor_values, "NODATA")

                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                elif SPEC_dataline.match(line) and scan_started:
                    # this is now the real end of the header block. at this
                    # point we know that there is enough information about the
                    # scan

                    # save the data offset
                    scan_data_offset = self.last_offset

                    # create an SPECFile scan object and add it to the scan
                    # list the name of the group consists of the prefix scan
                    # and the number of the scan in the file - this shoule make
                    # it easier to find scans in the HDF5 file.
                    s = SPECScan("scan_%i" % (scannr), scannr, scancmd, date,
                                 time, itime, col_names, scan_header_offset,
                                 scan_data_offset, self.full_filename,
                                 self.init_motor_names, init_motor_values,
                                 scan_status)
                    if scan_has_mca:
                        s.SetMCAParams(mca_col_number, mca_channels, mca_start,
                                       mca_stop)

                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                elif SPEC_scan.match(line) and scan_started:
                    # this should only be the case when there are two
                    # consecutive file headers in the data file without any
                    # data or abort notice of the first scan; first store
                    # current scan as aborted then start new scan parsing

                    try:
                        s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                     date, time, itime, col_names,
                                     scan_header_offset, scan_data_offset,
                                     self.full_filename, self.init_motor_names,
                                     init_motor_values, "ABORTED")
                    except:
                        scan_data_offset = self.last_offset
                        s = SPECScan("scan_%i" % (scannr), scannr, scancmd,
                                     date, time, itime, col_names,
                                     scan_header_offset, scan_data_offset,
                                     self.full_filename, self.init_motor_names,
                                     init_motor_values, "ABORTED")

                    self.scan_list.append(s)

                    # reset control flags
                    scan_started = False
                    scan_has_mca = False
                    # reset initial motor positions flag
                    init_motor_values = []

                    # start parsing of new scan
                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.io.SPECFile.Parse: found scan "
                              "(after aborted scan)")
                    line_list = SPEC_multi_blank.split(line)
                    scannr = int(line_list[1])
                    scancmd = "".join(" " + x + " " for x in line_list[2:])
                    scan_started = True
                    scan_has_mca = False
                    scan_header_offset = self.last_offset
                    scan_status = "OK"
                    if config.VERBOSITY >= config.INFO_LOW:
                        print("XU.io.SPECFile.Parse: processing scan nr. %i"
                              "..." % scannr)

                # store the position of the file pointer
                self.last_offset += linelength

            # if reading of the file is finished store the data offset of the
            # last scan as the last offset for the next parsing run of the file
            self.last_offset = self.scan_list[-1].doffset


class SPECCmdLine(object):

    def __init__(self, n, prompt, cmdl, out=""):
        self.linenumber = n
        self.prompt = prompt
        self.command = cmdl
        self.out = out

    def __str__(self):
        ostr = "%i.%s> %s" % (self.linenumber, self.prompt, self.command)
        return ostr


class SPECLog(object):
    """
    class to parse a SPEC log file to find the command history
    """
    def __init__(self, filename, prompt, path=""):
        """
        init routine for a class to read a SPEC log file

        Parameters
        ----------
         filename:  SPEC log file name
         prompt:    SPEC command prompt (e.g. 'PSIC' or 'SPEC')
         path:      (optional) directory where the SPEC log can be found
        """
        self.filename = filename
        self.full_filename = os.path.join(path, self.filename)

        self.prompt = prompt
        self.prompt_re = re.compile(r"%s>" % self.prompt)

        self.cmdl_list = []
        self.line_counter = 0
        self.Parse()

    def Parse(self):
        with xu_open(self.full_filename, 'r') as fid:
            for line in fid:
                line = line.decode('ascii')
                self.line_counter += 1

                line = line.strip()
                if self.prompt_re.findall(line):
                    [line, cmd] = self.prompt_re.split(line)
                    self.cmdl_list.append(SPECCmdLine(int(float(line)),
                                                      self.prompt, cmd))

    def __getitem__(self, index):
        """
        function to return the n-th cmd in the spec-log.
        """
        return self.cmdl_list[index]

    def __str__(self):
        ostr = "%s with %d lines\n" % (self.filename, self.line_counter)

        for cmd in self.cmdl_list:
            ostr = ostr + cmd.__str__() + "\n"

        return ostr


def geth5_scan(h5f, scans, *args, **kwargs):
    """
    function to obtain the angular cooridinates as well as intensity values
    saved in an HDF5 file, which was created from a spec file by the Save2HDF5
    method. Especially useful for reciprocal space map measurements.

    further more it is possible to obtain even more positions from
    the data file if more than two string arguments with its names are given

    Parameters
    ----------
     h5f:     file object of a HDF5 file opened using h5py or its filename
     scans:   number of the scans of the reciprocal space map (int,tuple or
              list)

     *args:   names of the motors (optional) (strings)
     to read reciprocal space maps measured in coplanar diffraction give:
     omname:  e.g. name of the omega motor (or its equivalent)
     ttname:  e.g. name of the two theta motor (or its equivalent)

     **kwargs (optional):
       samplename:  string with the hdf5-group containing the scan data
                    if ommited the first child node of h5f.root will be used
       rettype:     how to return motor positions. by default a list of arrays
                    is returned. when rettype == 'numpy' a record array will
                    be returned.

    Returns
    -------
     MAP

     or

     [ang1,ang2,...],MAP:
                angular positions of the center channel of the position
                sensitive detector (numpy.ndarray 1D) together with all the
                data values as stored in the data file (includes the
                intensities e.g. MAP['MCA']).

    Example
    -------
    >>> [om, tt], MAP = xu.io.geth5_scan(h5file, 36, 'omega', 'gamma')
    """

    with xu_h5open(h5f) as h5:
        if "samplename" in kwargs:
            h5g = h5.get(kwargs["samplename"])
        else:
            h5g = h5.get(list(h5.keys())[0])

        if numpy.iterable(scans):
            scanlist = scans
        else:
            scanlist = list([scans])

        angles = dict.fromkeys(args)
        for key in angles.keys():
            if not isinstance(key, str):
                raise InputError("*arg values need to be strings with "
                                 "motornames")
            angles[key] = numpy.zeros(0)
        buf = numpy.zeros(0)
        MAP = numpy.zeros(0)

        for nr in scanlist:
            h5scan = h5g.get("scan_%d" % nr)
            sdata = numpy.asarray(h5scan.get('data'))
            if MAP.dtype == numpy.float64:
                MAP.dtype = sdata.dtype
            # append scan data to MAP, where all data are stored
            MAP = numpy.append(MAP, sdata)
            # check type of scan
            notscanmotors = []
            for i in range(len(args)):
                motname = args[i]
                try:
                    buf = sdata[motname]
                    scanshape = buf.shape
                    angles[motname] = numpy.concatenate((angles[motname], buf))
                except:
                    notscanmotors.append(i)
            if len(notscanmotors) == len(args):
                scanshape = len(sdata)
            for i in notscanmotors:
                motname = args[i]
                buf = numpy.ones(scanshape) * \
                    h5scan.attrs["INIT_MOPO_%s" % makeNaturalName(motname)]
                angles[motname] = numpy.concatenate((angles[motname], buf))

    # create return values in correct order
    def create_retval():
        retval = []
        for motname in args:
            retval.append(angles[motname])
        return retval

    if "rettype" in kwargs:
        if kwargs['rettype'] == 'numpy':
            retval = numpy.core.records.fromarrays([angles[m] for m in args],
                                                   names=args)
        else:
            retval = create_retval()
    else:
        retval = create_retval()

    if len(args) == 0:
        return MAP
    else:
        return retval, MAP


def getspec_scan(specf, scans, *args, **kwargs):
    """
    function to obtain the angular cooridinates as well as intensity values
    saved in a SPECFile. Especially useful to combine the data from multiple
    scans.

    further more it is possible to obtain even more positions from
    the data file if more than two string arguments with its names are given

    Parameters
    ----------
     specf:   SPECFile object
     scans:   number of the scans of the reciprocal space map (int,tuple or
              list)

     *args:   names of the motors and counters (strings)
     keyword arguments:
      rettype:     how to return motor positions. by default a list of arrays
                   is returned. when rettype == 'numpy' a record array will
                   be returned.

    Returns
    -------
     [ang1,ang2,...]:
                coordinates and counters from the SPEC file

    Example
    -------
    >>> [om, tt, cnt2] = xu.io.getspec_scan(s, 36, 'omega', 'gamma',
                                            'Counter2')
    """
    if len(args) == 0:
        return

    if numpy.iterable(scans):
        scanlist = scans
    else:
        scanlist = list([scans])

    angles = dict.fromkeys(args)
    for key in angles.keys():
        if not isinstance(key, str):
            raise InputError("*arg values need to be strings with "
                             "motornames")
        angles[key] = numpy.zeros(0)
        buf = numpy.zeros(0)

    for nr in scanlist:
        sscan = specf.__getattr__("scan%d" % nr)
        sscan.ReadData()
        sdata = sscan.data
        # check type of scan
        notscanmotors = []
        for i in range(len(args)):
            motname = args[i]
            try:
                buf = sdata[motname]
                scanshape = buf.shape
                angles[motname] = numpy.concatenate((angles[motname], buf))
            except:
                notscanmotors.append(i)
        if len(notscanmotors) == len(args):
            scanshape = len(sdata)
        for i in notscanmotors:
            motname = args[i]
            buf = (numpy.ones(scanshape) *
                   sscan.init_motor_pos["INIT_MOPO_%s"
                                        % makeNaturalName(motname)])
            angles[motname] = numpy.concatenate((angles[motname], buf))

    # create return values in correct order
    def create_retval():
        retval = []
        for motname in args:
            retval.append(angles[motname])
        return retval

    if "rettype" in kwargs:
        if kwargs['rettype'] == 'numpy':
            retval = numpy.core.records.fromarrays([angles[m] for m in args],
                                                   names=args)
        else:
            retval = create_retval()
    else:
        retval = create_retval()

    return retval
