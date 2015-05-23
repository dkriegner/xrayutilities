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
# Copyright (C) 2009-2012 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to handle spectra data
"""

import numpy
import numpy.lib.recfunctions
import re
import tables
import os
from numpy import rec
import glob

from .. import config

re_wspaces = re.compile(r"\s+")
re_colname = re.compile(r"^Col")

re_comment_section = re.compile(r"^%c")
re_parameter_section = re.compile(r"^%p")
re_data_section = re.compile(r"^%d")
re_end_section = re.compile(r"^!")
re_unit = re.compile(r"\[.+\]")
re_obracket = re.compile(r"\[")
re_cbracket = re.compile(r"\]")
re_underscore = re.compile(r"_")
re_column = re.compile(r"^Col")
re_col_name = re.compile(r"\d+\s+.+\s*\[")
re_col_index = re.compile(r"\d+\s+")
re_col_type = re.compile(r"\[.+\]")
re_num = re.compile(r"[0-9]")

re_mca_int_tmp = re.compile(r"%.*i")

dtype_map = {"FLOAT": "f4"}

_absorber_factors = None


class SPECTRAFileComments(dict):

    """
    Class that describes the comments in the header of a SPECTRA file.
    The different comments are accessible via the comment keys.
    """

    def __init__(self):
        pass

    def __getattr__(self, name):
        if name in self:
            return self[key]


class SPECTRAFileParameters(dict):

    def __init__(self):
        pass

    def __getattr__(self, name):
        if name in self:
            return self[name]

    def __str__(self):
        ostr = ""
        n = len(self.keys())
        lmax_key = 0
        lmax_item = 0
        strlist = []

        # find the length of the longest key
        for k in self.keys():
            if len(k) > lmax_key:
                lmax_key = len(k)

            i = self[k]
            if not isinstance(i, str):
                # if the item is not a string it must be converted
                i = "%f" % i

            if len(i) > lmax_item:
                lmax_item = len(i)

        # define the format string for a single key-value pair
        kvfmt = "|%%-%is = %%-%is" % (lmax_key, lmax_item)

        nc = 3
        nres = len(self.keys()) % nc
        nrow = (len(self.keys()) - nres) / nc

        cnt = 0
        ostr += (3 * (lmax_key + lmax_item + 4) + 1) * "-" + "\n"
        ostr += "|Parameters:" + (3 * (lmax_key + lmax_item)) * " " + "|\n"
        ostr += (3 * (lmax_key + lmax_item + 4) + 1) * "-" + "\n"
        for key in self.keys():
            value = self[key]
            if not isinstance(value, str):
                value = "%f" % value

            ostr += kvfmt % (key, value)
            cnt += 1
            if cnt == 3:
                ostr += "|\n"
                cnt = 0

        if cnt != 0:
            ostr += "|\n"
        ostr += (3 * (lmax_key + lmax_item + 4) + 1) * "-" + "\n"

        return ostr


class SPECTRAFileDataColumn(object):

    def __init__(self, index, name, unit, type):
        self.index = int(index)
        self.name = name
        self.unit = unit
        self.type = type

    def __str__(self):
        ostr = "%i %s %s %s" % (self.index, self.name, self.unit, self.type)
        return ostr


class SPECTRAFileData(object):

    def __init__(self):
        self.collist = []
        self.data = None

    def append(self, col):
        self.collist.append(col)

    def __getitem__(self, key):
        try:
            return self.data[key]
        except:
            print("XU.io.specta.SPECTRAFileData: data contains no column "
                  "named: %s!" % key)

    def __str__(self):
        ostr = ""

        # determine the maximum lenght of every column string
        lmax = 0
        for c in self.collist:
            if len(c.__str__()) > lmax:
                lmax = len(c.__str__())

        lmax += 3

        # want to print in three columns
        nc = 3
        nres = len(self.collist) % nc
        nrows = (len(self.collist) - nres) / nc

        fmtstr = "| %%-%is| %%-%is| %%-%is|\n" % (lmax, lmax, lmax)

        ostr += (3 * lmax + 7) * "-" + "\n"
        ostr += "|Column names:" + (3 * lmax - 8) * " " + "|\n"
        ostr += (3 * lmax + 7) * "-" + "\n"
        # full output rows
        for i in range(nrows):
            c1 = self.collist[i * nc + 0]
            c2 = self.collist[i * nc + 1]
            c3 = self.collist[i * nc + 2]
            ostr += fmtstr % (c1.__str__(), c2.__str__(), c3.__str__())

        # residual output row
        c = ['', '', '']
        for j in range(nres):
            c[j] = self.collist[-nres + j]

        ostr += fmtstr % (c[0].__str__(), c[1].__str__(), c[2].__str__())

        ostr += (3 * lmax + 7) * "-" + "\n"
        return ostr


class SPECTRAFile(object):

    """
    Represents a SPECTRA data file. The file is read during the
    Constructor call. This class should work for data stored at
    beamlines P08 and BW2 at HASYLAB.

    Required constructor arguments:
    ------------------------------
     filename ............. a string with the name of the SPECTRA file

    Optional keyword arguments:
    --------------------------
     mcatmp ............... template for the MCA files
     mcastart,mcastop ..... start and stop index for the MCA files, if not
                            given, the class tries to determine the start and
                            stop index automatically.
    """

    def __init__(self, filename, mcatmp=None, mcastart=None, mcastop=None):
        self.filename = filename
        self.comments = SPECTRAFileComments()
        self.params = SPECTRAFileParameters()
        self.data = SPECTRAFileData()
        self.mca = None
        self.mca_channels = None

        self.Read()  # reads the .fio data file

        if mcatmp is not None:
            self.mca_file_template = mcatmp

            if mcastart is not None and mcastop is not None:
                self.mca_start_index = mcastart
                self.mca_stop_index = mcastop
            else:
                # try to determine the number of MCA spectra automatically
                # spat = re_mca_int_tmp.sub("*",self.mca_file_template)
                spat = self.mca_file_template.replace("%i", "*")
                l = glob.glob(spat)
                self.mca_start_index = 1
                self.mca_stop_index = 0
                if len(l) != 0:
                    self.mca_stop_index = self.data.data.size  # len(l)

            if self.mca_stop_index != 0:
                self.ReadMCA()

    def Save2HDF5(self, h5file, name, group="/", description="SPECTRA scan",
                  mcaname="MCA"):
        """
        Saves the scan to an HDF5 file. The scan is saved to a
        seperate group of name "name". h5file is either a string
        for the file name or a HDF5 file object.
        If the mca attribute is not None mca data will be stored to an
        chunked array of with name mcaname.

        required input arguments:
         h5file .............. string or HDF5 file object
         name ................ name of the group where to store the data

        optional keyword arguments:
         group ............... root group where to store the data
         description ......... string with a description of the scan

        Return value:
        The method returns None in the case of everything went fine, True
        otherwise.
        """
        if isinstance(h5file, str):
            try:
                h5 = tables.openFile(h5file, mode="a")
            except:
                print("XU.io.spectra.Save2HDF5: cannot open file %s for "
                      "writing!" % h5file)
                return True

        else:
            h5 = h5file

        # create the group where to store the data
        try:
            g = h5.createGroup(group, name, title=description,
                               createparents=True)
        except:
            print("XU.io.spectra.Save2HDF5: cannot create group %s for "
                  "writing data!" % name)
            if isinstance(h5file, str):
                h5.close()
            return True

        # start with saving scan comments
        for k in self.comments.keys():
            try:
                h5.setNodeAttr(g, k, self.comments[k])
            except:
                print("XU.io.spectra.Save2HDF5: cannot save file comment "
                      "%s = %s to group %s!" % (k, self.comments[k], name))

        # save scan parameters
        for k in self.params.keys():
            try:
                h5.setNodeAttr(g, k, self.params[k])
            except:
                print("XU.io.spectra.Save2HDF5: cannot save file parametes "
                      "%s to group %s!" % (k, name))

        # ----------finally we need to save the data -------------------

        # first save the data stored in the FIO file
        tab_desc_dict = {}
        if self.data.data is not None:
            for t in self.data.data.dtype.descr:
                cname = t[0]
                if len(t[1:]) == 1:
                    ctype = numpy.dtype((t[1]))
                else:
                    ctype = numpy.dtype((t[1], t[2]))

                tab_desc_dict[cname] = tables.Col.from_dtype(ctype)

            # create the table object
            try:
                tab = h5.createTable(g, "data", tab_desc_dict, "scan data")
            except:
                print("XU.io.spectra.Save2HDF5: cannot create table for "
                      "storing scan data!")
                return True

            # now write the data to the tables
            for rec in self.data.data:
                for cname in rec.dtype.names:
                    tab.row[cname] = rec[cname]
                tab.row.append()

            tab.flush()

        # if there is MCA data - store this
        if self.mca is not None:
            a = tables.Float32Atom()
            f = tables.Filters(complib="zlib", complevel=9, fletcher32=True)
            try:
                c = h5.createCArray(g, mcaname, a, self.mca.shape)
            except:
                print("XU.io.spectra.Save2HDF5: cannot create carray %s for "
                      "MCA data!" % mcaname)
                return True

            c[...] = self.mca[...]

            # set MCA specific attributes
            h5.setNodeAttr(c, "channels", self.mca_channels)
            h5.setNodeAttr(c, "nchannels", self.mca_channels.shape[0])

        h5.flush()

        if isinstance(h5file, str):
            h5.close()

        return None

    def ReadMCA(self):
        dlist = []
        for i in range(self.mca_start_index, self.mca_stop_index + 1):
            fname = self.mca_file_template % i
            data = numpy.loadtxt(fname)

            if i == self.mca_start_index:
                if len(data.shape) == 2:
                    self.mca_channels = data[:, 0]
                else:
                    self.mca_channels = numpy.arange(0, data.shape[0])

            if len(data.shape) == 2:
                dlist.append(data[:, 1].tolist())
            else:
                dlist.append(data.tolist())

        self.mca = numpy.array(dlist, dtype=float)

    def __str__(self):
        ostr = self.params.__str__()
        ostr += self.data.__str__()

        return ostr

    def Read(self):
        """
        Read the data from the file.
        """
        try:
            fid = open(self.filename, "r")
        except:
            print("XU.io.SPECTRAFile.Read: cannot open data file %s for "
                  "reading!" % (self.filename))
            return None

        col_names = ""
        col_units = []
        col_types = ""
        rec_list = []

        while True:
            lbuffer = fid.readline()
            if lbuffer == "":
                break
            lbuffer = lbuffer.strip()

            # read the next line if the line starts with a "!"
            if re_end_section.match(lbuffer):
                continue

            # select the which section to read
            if re_comment_section.match(lbuffer):
                read_mode = 1
                continue

            if re_parameter_section.match(lbuffer):
                read_mode = 2
                continue

            if re_data_section.match(lbuffer):
                read_mode = 3
                continue

            # here we decide how to proceed with the data
            if read_mode == 1:
                # read the file comments
                try:
                    (key, value) = lbuffer.split("=")
                except:
                    # avoid annoying output
                    if config.VERBOSITY >= config.INFO_ALL:
                        print("XU.io.SPECTRAFile.Read: cannot interpret the "
                              "comment string: %s" % (lbuffer))
                    continue

                key = key.strip()
                # remove whitespaces to be conform with natural naming
                key = key.replace(' ', '')
                # remove possible number at first position
                if re_num.findall(key[0]) != []:
                    key = "_" + key
                value = value.strip()
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECTRAFile.Read: comment: k,v: %s, %s"
                          % (key, value))

                try:
                    value = float(value)
                except:
                    pass

                # need to handle the case, that a key may appear several times
                # in the list
                kcnt = 0
                while True:
                    try:
                        self.comments[key] = value
                        # if adding the key/value pair to the dictionary
                        # was successful - leave the loop
                        break
                    except:
                        key += "_%i" % (kcnt + 2)

                    kcnt += 1

            elif read_mode == 2:
                # read scan parameters
                try:
                    (key, value) = lbuffer.split("=")
                except:
                    print("XU.io.SPECTRAFile.Read: cannot interpret the "
                          "parameter string: %s" % (lbuffer))

                key = key.strip()
                # remove whitespaces to be conform with natural naming
                key = key.replace(' ', '')
                # remove possible number at first position
                if re_num.findall(key[0]) != []:
                    key = "_" + key
                value = value.strip()
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECTRAFile.Read: parameter: k,v: %s, %s"
                          % (key, value))

                try:
                    value = float(value)
                except:
                    # if the conversion of the parameter to float
                    # fails it will be saved as a string
                    pass

                # need to handle the case, that a key may appear several times
                # in the list
                kcnt = 0
                while True:
                    try:
                        self.params[key] = value
                        # if adding the key/value pair to the dictionary
                        # was successful - leave the loop
                        break
                    except:
                        key += "_%i" % (kcnt + 2)

                    kcnt += 1

            elif read_mode == 3:
                if re_column.match(lbuffer):
                    try:
                        unit = re_unit.findall(lbuffer)[0]
                    except IndexError:
                        unit = "NONE"

                    try:
                        lval = re_obracket.split(lbuffer)[0]
                        rval = re_cbracket.split(lbuffer)[-1]
                        dtype = rval.strip()
                        l = re_wspaces.split(lval)
                        index = int(l[1])
                        name = "".join(l[2:])
                    except IndexError:
                        l = re_wspaces.split(lbuffer)
                        index = int(l[1])
                        dtype = l[-1]
                        name = "".join(l[2:-1])

                    # store column definition
                    self.data.append(
                        SPECTRAFileDataColumn(index, name, unit, dtype))

                    if name in col_names.split(","):
                        name += "%s_1" % name

                    col_names += "%s," % name
                    col_types += "%s," % (dtype_map[dtype])

                else:
                    # read data
                    dlist = re_wspaces.split(lbuffer)
                    for i in range(len(dlist)):
                        dlist[i] = float(dlist[i])

                    rec_list.append(dlist)

        col_names = col_names[:-1]
        col_types = col_types[:-1]
        if config.VERBOSITY >= config.DEBUG:
            print("XU.io.SPECTRAFile.Read: data columns: name,type: %s, %s"
                  % (col_names, col_types))
        if len(rec_list) != 0:
            self.data.data = rec.fromrecords(rec_list, formats=col_types,
                                             names=col_names)
        else:
            self.data.data = None


class Spectra(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.h5_file = None
        self.h5_group = None
        self.abs_factors = None

    def set_abs_factors(self, ff):
        """
        Set the global absorber factors in the module.
        """
        if isinstance(ff, list):
            self.abs_factors = numpy.array(ff, dtype=numpy.double)
        elif isinstance(ff, numpy.ndarray):
            self.abs_factors = ff

    def recarray2hdf5(self, h5g, rec, name, desc):
        """
        Save a record array in an HDF5 file. A pytables table
        object is used to store the data.

        required input arguments:
         h5g ................. HDF5 group object or path
         rec ................ record array
         name ............... name of the table in the file
         desc ............... description of the table in the file

        return value:
         tab ................. a HDF5 table object
        """

        # to build the table data types and names must be extracted
        descr = rec.dtype.descr
        tab_desc = {}
        cname_list = []

        for d in descr:
            tab_desc[d[0]] = tables.Col.from_dtype(numpy.dtype(d[1]))
            cname_list.append(d[0])

        # create the table object
        try:
            tab = self.h5_file.createTable(h5g, name, tab_desc, desc)
        except:
            print("XU.io.spectra.Spectra: Error creating table object %s!")
            return None

        # fill in data values
        for i in range(rec.shape[0]):
            for k in cname_list:
                tab.row[k] = rec[k][i]

            tab.row.append()

        tab.flush()
        self.h5_file.flush()

    def spectra2hdf5(self, dir, fname, mcatemp, name="", desc="SPECTRA data"):
        """
        Convert SPECTRA scan data to a HDF5 format.

        required input arguments:
         dir ............... directory where the scan is stored
         fname ............. name of the SPECTRA data file
         mcatemp ........... template for the MCA file names

        optional keyword arguments:
         name .............. optional name under which to save the data
                             if empty the basename of the filename will be used
         desc .............. optional description of the scan
        """

        (basename, ext) = os.path.splitext(fname)
        mcadir = os.path.join(dir, name)

        # evaluate keyword arguments
        if name == "":
            sg_name = basename
        else:
            sg_name = name

        sg_desc = desc

        # check wether an MCA directory exists or not
        if os.path.exists(mcadir):
            has_mca = True
        else:
            has_mca = False

        fullfname = os.path.join(dir, fname)
        if not os.path.exists(fullfname):
            print("XU.io.spectra.Spectra.spectra2hdf5: data file does not "
                  "exist!")
            return None

        # read data file
        (data, hdr) = read_data(fullfname)

        # create a new group to save the scan data in
        # this group is created below the default group determined by
        # self.h5_group
        try:
            sg = self.h5_file.createGroup(self.h5_group, sg_name, sg_desc)
        except:
            print("XU.io.spectra.Spectra.spectra2hdf5: cannot create scan "
                  "group!")
            return None

        self.recarray2hdf5(sg, data, "data", "SPECTRA tabular data")

        # write attribute data
        for k in hdr.keys():
            self.h5_file.setNodeAttr(sg, "MOPOS_" + k, hdr[k])

        if has_mca:
            mca = read_mca_dir(mcadir, mcatemp)
            a = tables.Float64Atom()
            filter = tables.Filters(complib="zlib", complevel=4,
                                    fletcher32=True)
            c = self.h5_file.createCArray(sg, "MCA", a, mca.shape, "MCA data",
                                          filters=filter)
            c[...] = mca[...]

        self.h5_file.flush()

        return sg

    def abs_corr(self, data, f, **keyargs):
        """
        Perform absorber correction. Data can be either a 1 dimensional data
        (point detector) or a 2D MCA array. In the case of an array the data
        array should be of shape (N,NChannels) where N is the number of points
        in the scan an NChannels the number of channels of the MCA. The
        absorber values are passed to the function as a 1D array of N elements.

        By default the absorber values are taken form a global variable stored
        in the module called _absorver_factors. Despite this, costume values
        can be passed via optional keyword arguments.

        required input arguments:
         mca ............... matrix with the MCA data
         f ................. filter values along the scan

        optional keyword arguments:
         ff ................ custome filter factors

        return value:
         Array with the same shape as mca with the corrected MCA data.
        """

        mcan = numpy.zeros(data.shape, dtype=numpy.double)

        if "ff" in keyargs:
            ff = keyargs["ff"]
        else:
            ff = _absorber_factors

        if len(data.shape) == 2:
            # MCA and matrix data
            data = data * ff[f][:, numpy.newaxis]
        elif len(data.shape) == 1:
            data = data * ff[f]

        return data


def get_spectra_files(dirname):
    """
    Return a list of spectra files within a directory.

    required input arguments:
     dirname .............. name of the directory to search

    return values:
     list with filenames
    """

    fnlist = os.listdir(dirname)
    onlist = []

    for fname in fnlist:
        (name, ext) = os.path.splitext(fname)
        if ext == ".fio":
            onlist.append(fname)

    onlist.sort()
    return onlist


def read_mca_dir(dirname, filetemp, sort=True):
    """
    Read all MCA files within a directory
    """

    flist = get_spectra_files(dirname)

    # create a list with the numbers of the files
    nlist = []
    for fname in flist:
        (name, ext) = os.path.splitext(fname)
        name = name.replace(filetemp, "")
        nlist.append(int(name))

    if sort:
        nlist.sort()

    dlist = []

    for i in nlist:
        fname = os.path.join(dirname, filetemp + "%i.fio")
        fname = fname % (i)
        d = read_mca(fname)
        dlist.append(d.tolist())

    return numpy.array(dlist)


def read_mca(fname):
    """
    Read a single SPECTRA MCA file.

    required input arguments:
     fname ............... name of the file to read

    return value:
     data ................ a numpy array witht the MCA data
    """

    try:
        fid = open(fname)
    except:
        print("XU.io.spectra.read_mca: cannot open file %s!" % fname)
        return None

    dlist = []
    hdr_flag = True

    while True:
        lbuffer = fid.readline()

        if lbuffer == "":
            break
        lbuffer = lbuffer.strip()
        if lbuffer == "%d":
            hdr_flag = False
            lbuffer = fid.readline()
            continue

        if not hdr_flag:
            dlist.append(float(lbuffer))

    return numpy.array(dlist, dtype=numpy.double)


def read_mcas(ftemp, cntstart, cntstop):
    """
    Read MCA data from a SPECTRA MCA directory. The filename is passed as a
    generic
    """

    fnums = range(cntstart, cntstop + 1)
    mcalist = []

    for i in fnums:
        fname = ftemp % i
        print("XU.io.spectra.read_mcas: processing file %s ..." % fname)
        mcalist.append(read_mca(fname))

    return numpy.array(mcalist, dtype=numpy.double)


def read_data(fname):
    """
    Read a spectra data file (a file with now MCA data).

    required input arguments:
     fname .................... name of the file to read

    return values: (data,hdr)
     data .......... numpy record array where the keys are the column names
     hdr ........... a dictionary with header information
    """

    try:
        fid = open(fname, "r")
    except:
        print("XU.io.spectra.read_data: cannot open file %s!" % fname)
        return None

    hdr_dict = {}
    hdr_flag = False
    data_flag = False
    col_cnt = 0  # column counter
    col_names = []  # list with column names
    data = []

    fname = os.path.basename(fname)
    fname, ext = os.path.splitext(fname)
    print(fname)

    while True:
        lbuffer = fid.readline()
        if lbuffer == "":
            break

        lbuffer = lbuffer.strip()
        # check for common break conditions
        # if the line is a comment skip it
        if lbuffer[0] == "!":
            continue

        # remove leading and trailing whitespace symbols
        lbuffer = lbuffer.strip()

        if lbuffer == "%p":
            hdr_flag = True
            continue

        if lbuffer == "%d":
            hdr_flag = False
            data_flag = True
            continue

        if hdr_flag:
            # read header data (initial motor positions)
            key, value = lbuffer.split("=")
            key = key.strip()
            value = value.strip()
            hdr_dict[key] = float(value)

        if data_flag:
            # have to read the column names first
            if re_colname.match(lbuffer):
                l = re_wspaces.split(lbuffer)
                col_names.append(l[2].replace(fname.upper() + "_", ""))
            else:
                # read data values
                dlist = re_wspaces.split(lbuffer)
                # convert strings to float values
                for i in range(len(dlist)):
                    dlist[i] = float(dlist[i])

                data.append(dlist)

    # create a record array to hold data
    data = numpy.rec.fromrecords(data, names=col_names)

    return (data, hdr_dict)


def geth5_spectra_map(h5file, scans, *args, **kwargs):
    """
    function to obtain the omega and twotheta as well as intensity values
    for a reciprocal space map saved in an HDF5 file, which was created
    from a spectra file by the Save2HDF5 method.

    further more it is possible to obtain even more positions from
    the data file if more than two string arguments with its names are given

    Parameters
    ----------
     h5f:     file object of a HDF5 file opened using pytables
     scans:   number of the scans of the reciprocal space map (int,tuple or
              list)

    *args:   arbitrary number of motor names (strings)
     omname:  name of the omega motor (or its equivalent)
     ttname:  name of the two theta motor (or its equivalent)

    **kwargs (optional):
     mca:        name of the mca data (if available) otherwise None
                 (default: "MCA")
     samplename: string with the hdf5-group containing the scan data
                 if omitted the first child node of h5f.root will be used
                 to determine the sample name

    Returns
    -------
     [ang1,ang2,...],MAP:
                angular positions of the center channel of the position
                sensitive detector (numpy.ndarray 1D) together with all the
                data values as stored in the data file (includes the
                intensities e.g. MAP['MCA']).
    """
    if isinstance(h5file, str):
        try:
            h5 = tables.openFile(h5file, mode="r")
        except:
            print("XU.io.spectra.geth5_spectra_map: cannot open file %s "
                  "for reading!" % h5file)
            return True

    else:
        h5 = h5file

    if "mca" in kwargs:
        mca = kwargs["mca"]
    else:
        mca = "MCA"

    if "samplename" in kwargs:
        basename = kwargs["samplename"]
    else:
        nodename = h5.listNodes(h5.root)[0]._v_name
        basenlist = re_underscore.split(nodename)
        basename = "_".join(basenlist[:-1])
        if config.VERBOSITY >= config.DEBUG:
            print("XU.io.spectra.geth5_spectra_map: using \'%s\' as basename"
                  % (basename))

    if isinstance(scans, (list, tuple)):
        scanlist = scans
    else:
        scanlist = list([scans])

    angles = dict.fromkeys(args)
    for key in angles.keys():
        angles[key] = numpy.zeros(0)
    buf = numpy.zeros(0)
    MAP = numpy.zeros(0)

    for nr in scanlist:
        h5scan = h5.getNode(h5.root, basename + "_%05d" % nr)
        sdata = h5scan.data.read()
        if mca:
            mcanode = h5.getNode(h5.root, basename + "_%05d/%s" % (nr, mca))
            mcadata = mcanode.read()

        # append scan data to MAP, where all data are stored
        sdtmp = numpy.lib.recfunctions.append_fields(
            sdata, [mca, ], [mcadata, ],
            dtypes=[(numpy.double, mcadata.shape[1])], usemask=False,
            asrecarray=True)
        if MAP.dtype == numpy.float64:
            MAP.dtype = sdtmp.dtype
        MAP = numpy.append(MAP, sdtmp)

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
        for i in notscanmotors:
            motname = args[i]
            buf = numpy.ones(scanshape) * \
                h5.getNodeAttr(h5scan, "%s" % motname)
            angles[motname] = numpy.concatenate((angles[motname], buf))

    retval = []
    for motname in args:
        # create return values in correct order
        retval.append(angles[motname])

    if isinstance(h5file, str):
        h5.close()

    return retval, MAP
