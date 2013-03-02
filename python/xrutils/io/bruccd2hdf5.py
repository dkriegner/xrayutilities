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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
this module provides routines to insert the data of a Bruker Apex CCD
detector into a HDF5 format

GENERAL NOTES ON THE BRUKER CCD FILE FORMAT
-------------------------------------------

Several version of the Bruker CCDF file format exist. This module supports
actually two versions. The version of the CCD file is determined by two values
stored in the files header:

* FORMAT    determining the data format
* VERSION   determining the version of a certain format

Format values
-------------
* FORMAT=86     the file uses a frame compression format as described in
                appendix A3 of the APEX CCD manual
* FORMAT=100    the file uses the new image compression format as described
                in appendix A2 of the Bruker APEX CCD manual

The format determines mainly how the data is stored in the file.

Version values
--------------
The version value describes the version of the file header format.
There, the following values can appear:

- VERSION=11    fixed-format/free-line header, only valid with Format 100
- VERSION=101   new free-format header

This module supports actually two Bruker CCD formats:

- fixed-format/free line (Format=100, Version=11)
- fixed-format frame image (Format=86, Version=101)

The aim of this module
======================
Unlike many other modules this one should not be treated as a 'data importer' in a
common sense. To work with the CCD data two steps have to be done.

1. convert the data files to a HDF5 file
2. do some data correction and manipulate the data

For a better understanding one can take the term 'conversion' in its very strict
sense. In the first step the data is converted from Brukers own file format
to a HDF5 file by means of representing the storage structures of a HDF5 file
by using HDF5 objects. The aim of this step is to include as much information as
possible from the original file format (for instance the entire header).
In the second step the module provides helper functions to extract data from these
HDF5 structures. The motivation for this that it is much easier and faster to work
on a bulk of HDF5 objects in one file than to work on a huge bulk of single
files on a physical file system.

Data abstraction
----------------
Bruker uses an overflow/underflow concept for file compressions. Every data file
consists in fact of something between 3 and 5 objects:

- a header
- the raw 1b image data
- a single 1byte overflow table
- a 1Byte and a 2Byte overflow table
- an underflow table

Bruker is using a overflow underflow table for doing file compression.
Since HDF5 provides the possibilities to do inline compression of an array I will
do the overflow underflow correction within the code and store the data in an EArray
including compression with zlib.
The detector images are assumed to be stored somewhere in a group of a HDF5 file.
In addition into every of such groups a seperate table is generated to hold the header data
of the data files. The header table is called "BRUKER_CCD_V11_HEADERS"

Filename convention
===================
Since CCD data is usually stored during a scan of something the module provides
the feature of reading a whole set of data files.
In this case some naming convention for the filenames is used:
pattern+_filenumber_framenumber.sfrm
In such a case the user has to provide the following information to automatically
read the data:

filepattern :  the general pattern for the filename.
scanrange :    range of filenumbers to read.
framerange :   range of frames to read.

The filepattern is given in the form of a format string like: "patter__%5i_%3%i.sfrm"
where the first integer is the scan number and the second one the frame number.

User functions
==============
The module provides the following user functions (everything else is low level and
therfore hardly usefull for the common user):

 ccd2hdf5 :   reads a single CCD file and stores it to HDF5
 ccdf2hdf5 :  reads a single HDF5 file with all frames
 ccds2hdf5 :  reads an entire collection of CCD files
"""

import tables
import numpy  # replaced Numeric; changes not tested
import struct
import re
import os.path

from . import bruker_header   #manages to read the big Bruker header

#some global module variables which should be set by the user before using the
#module

file_number_prec = 7  #number of digits for the filenumber
frame_number_prec = 3 #number of digits for the framenumber
file_suffix = ".sfrm" #suffix for the CCD files
file_path = "."      #default path where to look for the files

regexp_file_suffix = re.compile(r"$\.sfrm")

#global flags in the module
be_verbose = 0   # this flag determines if the module should print status information
                  # during the run of its functions

def ccds2hdf5(h5file,pattern,sn_list,fn_list,**optargs):
    """
    Stores an entire collection of Bruker CCD files to a HDF5 file.
    
    Parameters
    ----------
     h5file :    HDF5 file object
     pattern :   file pattern as a format string (two numbers are included describing the scan and the frame number)
     sn_list :   list of scan_numbers to read
     fn_list :   list of frame numbers to read

    optional arguments:
     path :      data path where to read the data
     group :     HDF5 group where to store the data
    """

    #check for optional arguments:
    if "path" in optargs:
        path = optargs["path"]
    else:
        path = file_path

    if "group" in optargs:
        h5group = optargs["group"]
    else:
        h5group = h5file.root

    #build a list with the filenames
    filename_list = []
    for i in sn_list:
        for j in fn_list:
            filename_list.append(pattern %(i,j))

    #start to import the data:
    for f in filename_list:
        try:
            print("Importing %s ...." %(f))
            ccd2hdf5(h5file,f,path=path,group=h5group)
        except:
            raise IOError("error importing file: %s" %f)




def ccd2hdf5(h5file,filename,**optargs):
    """
    ccd2hdf5(filename,h5file,h5group)
    Converts the content of a Bruker CCD file to a HDF5 object and stores it in
    a HDF5 file.
    
    Input arguments:
     h5file :       HDF5 file object
     filename :  name of the CCD data file
    
    optional input arguments:
     path :      path to the
     group :     the HDF5 group where to store the data
    """

    #set the default filter for compression fo the picture data
    filter = tables.Filters(complevel=1,complib="zlib")
    filter.fletcher32 = 1

    #setting the path to the file
    if "path" in optargs:
        path = optargs["path"]
    else:
        path = file_path

    if "group" in optargs:
        h5group = optargs["group"]
    else:
        h5group = h5file.root

    #try to open the file
    try:
        fid = open(os.path.join(path,filename),mode="r")
    except:
        print("error opening CCD file %s" %(os.path.join(path,filename)))

    #have to determine the format used in the file and the version of the header
    fid.read(8) #dummy read
    format = int(fid.read(72)) #read file format
    fid.read(8) #dummy read
    version = int(fid.read(72)) #read file header version
    fid.seek(0) #reset file pointer to the starting position

    if version != 11:
        print("unsupported header version - not implemented now")
        return

    #check if in the selected HDF5 group allready exists an header table
    try:
        htab = h5group.BRUKER_CCD_V11_HEADERS
    except:
        print("build a new header table for this group")
        #initiate a HDF5 table for the header information:
        htab_name ="BRUKER_CCD_V11_HEADERS"
        htab_title = "Table holding the header information of Bruker ANEX II V11 headers"
        htab = h5file.createTable(h5group,htab_name,bruker_header.V11_header_table,\
                              "header for file "+htab_name)

    h5_earray_name = filename[:-5]
    h5_earray_title = "CCD data from file "+filename
    #add the frame number to the header table
    bruker_header.read_header(fid,htab,h5_earray_name)
    #need the index of the table entry where the header is stored
    for i in range(len(htab.cols.FNAME)):
        if htab.cols.FNAME[i] == h5_earray_name:
            hdr_tab_index = i
            break

    nrows = htab.cols.NROWS[hdr_tab_index]
    ncols  = htab.cols.NCOLS[hdr_tab_index]
    nof_underflow = htab.cols.NOVERFL[hdr_tab_index][0,0]
    nof_1b_overflow = htab.cols.NOVERFL[hdr_tab_index][0,1]
    nof_2b_overflow = htab.cols.NOVERFL[hdr_tab_index][0,2]
    nof_pixel_bytes = htab.cols.NPIXELB[hdr_tab_index][0,0]
    nof_underflow_bytes = htab.cols.NPIXELB[hdr_tab_index][0,1]
    #calculate the file pointer offsets to the data positions
    data_offset = 512*htab.cols.HDRBLKS[hdr_tab_index]      #calculate the offset to the main data block
    uv_data_offset = data_offset+nrows*ncols*nof_pixel_bytes #calculate the offset to the underflow table
    if nof_underflow>=0:
        ov_data_offset = data_offset+uv_data_offset+nof_underflow*nof_underflow_bytes
    else:
        ov_data_offset = data_offset+nrows*ncols*nof_pixel_bytes


    #after reading the header start to read the file
    b1_ov_flag = 0  #flag indicating the useage of 1Byte overflow table
    b2_ov_flag = 0  #flag indicating the useage of 2Byte overflow table
    uv_flag = 0    #flag indicating the useage fo the underflow table
    if nof_1b_overflow > 0: b1_ov_flag = 1
    if nof_2b_overflow > 0: b2_ov_flag = 1
    if nof_underflow > 0: uv_flag = 1

    #after setting all necessary flags one can start with setting up the array for storing the data
    atype = tables.IntAtom(shape=(0,nrows,ncols))
    h5_earray = h5file.createEArray(h5group,h5_earray_name,atype,h5_earray_title,filters=filter)

    #--------------------------------------start now to read the data from the file--------------------------------------------
    #first we have to set the format string for reading the main data block
    if nof_pixel_bytes==1:
        data_fmt_str = nrows*ncols*"B"
    elif nof_pixel_bytes==2:
        data_fmt_str = nrows*ncols*"H"
    elif nof_pixel_bytes ==4:
        data_fmt_str = nrows*ncols*"I"

    #read the data string from the file
    fid.seek(data_offset)
    data_string = fid.read(nrows*ncols*nof_pixel_bytes)
    data_sequ = struct.unpack(data_fmt_str,data_string)
    data_list = []
    #the data list is now a pack of strings - have to convert it to int
    for i in range(len(data_sequ)):
        data_list.append(int(data_sequ[i]))

    #in the next step the overflow correction will be done
    ov_counter_1b = 0
    ov_counter_2b = 0
    ov_table_1b = []
    ov_table_2b = []

    #read the 1byte overflow table
    fid.seek(ov_data_offset)
    for i in range(nof_1b_overflow):
        data_string = fid.read(2)
        ov_table_1b.append(int(struct.unpack('H',data_string)[0]))


    #calculate the offset position of the 2byte overflow table
    if (2*nof_1b_overflow)%16 ==0:
        #in this case the bytes for the 1byte table are a multiple of 16
        fid.seek(ov_data_offset+2*nof_1b_overflow)
    else:
        #if the bytes for the 1byte overflow are not an integer multiple of 16
        fid.seek(ov_data_offset+2*nof_1b_overflow+16-(2*nof_1b_overflow)%16)

    #read now the 2byte overflow table
    for i in range(nof_2b_overflow):
        data_string = fid.read(4)
        ov_table_2b.append(int(struct.unpack('I',data_string)[0]))

    #now start with the correction of the main data block by using the overflow tables:
    for i in range(len(data_list)):
        if data_list[i]==255:
            #1-byte overflow
            data_list[i]=ov_table_1b[ov_counter_1b]
            ov_counter_1b = ov_counter_1b + 1
            if data_list[i]==65535:
                data_list[i] = ov_table_2b[ov_counter_2b]
                ov_counter_2b = ov_counter_2b + 1

    #finally we have to store the data in the HDF5 EArray:
    tmparray = numpy.array(data_list).astype(numpy.int)
    tmparray = numpy.reshape(tmparray,(nrows,ncols))
    h5_earray.append([tmparray])

def load_data(hdr,fid):
    """
    Load the raw data from the file (without under- and overflow correction).
    
    Required input arguments:
     hdr ............... the header class of the CCD file
     fid ............... file object to the CCD file

    return values:
     data .............. an array of size (n,n) with the raw 1byte data
    """

    #load the datasize and shape from the header
    nof_rows = hdr.nrows
    nof_cols = hdr.ncols
    nof_pixbytes = hdr.npixelb[0]

    #get the old fid position and determine the startin point for the raw data
    old_fid_pos = fid.tell()
    end_of_hdr = hdr.hdrblks*512
    fid.seek(0)
    fid.seek(end_of_hdr)

    #read the data string
    try:
        data_string = fid.read(nof_rows*nof_cols*nof_pixbytes)
    except:
        print("error reading raw data from the file")
        return 0

    fmt_str = (nof_rows*nof_cols)*'B'
    raw_data = struct.unpack(fmt_str,data_string)
    data_array = numpy.array(raw_data,numpy.int)

    data_array = numpy.reshape(data_array,(nof_rows,nof_cols))

    #set the file object to the old offset position in  the file
    fid.seek(old_fid_pos)

    return data_array


def load_underflow(hdr,fid):
    """
    Load the underflow data from the file.
    """

    nof_uderflows = 0

    try:
        nof_underflows = hdr.noverfl[2] #if this works the file is version 11
        nof_underflows = hdr.noverfl[0] #now we have to read the real underflow value
    except:
        print("file is not verions 11 so no underflow data is stored")
        return 0

    if nof_underflows<0:
        if be_verbose: print("no baseline subtraction done - no underflow data")
        return 0

    #if we reached this point we can load the underflow data
    nof_underflow_pix = hdr.npixelb[1]

    #set the file pointer to the right position
    old_fid_pos = fid.tell()
    fid_pos = 512*hdr.hdrblks+hdr.nrows*hdr.ncols*hdr.npixelb[0]
    fid.seek(fid_pos)

    #read the data in string format
    data_string = fid.read(nof_underflows*nof_underflow_pix)

    #build the format string
    if nof_underflow_pix==1:
        fmt_str = nof_underflows*'B'
    elif nof_underflow_pix==2:
        fmt_str = nof_underflows*'H'

    #convert the underflow data to a numeric array
    underflow_array = numpy.array(struct.unpack(fmt_str,data_string),numpy.int)

    #set the file pointer to its original position
    fid.seek(old_fid_pos)

    return underflow_array

def load_overflow(hdr,fid):
    """
    Load the overflow data from the file.
    """

    v11_flag = 0
    b1_overflow_tab = []
    b2_overflow_tab = []

    if len(hdr.noverfl)>1:
        v11_flag = 1

    #read the number of overflows depending on the data format version
    if v11_flag:
        nof_1b_overflows = hdr.noverfl[1]
        nof_2b_overflows = hdr.noverfl[2]
        if be_verbose: print("v11: number of 1 byte overflows: %i" %(nof_1b_overflows))
        if be_verbose: print("v11: number of 2 byte overflows: %i" %(nof_2b_overflows))
    else:
        nof_overflows = hdr.noverfl[0]
        if be_verbose: print("number of overflows: %i" %(nof_overflows))

    #set the file object to the starting position of the overflow data
    fid_pos_old = fid.tell()

    if v11_flag:
        if be_verbose: print("calculate overflow position for v11 data")
        fid_pos = 512*hdr.hdrblks+hdr.nrows*hdr.ncols*hdr.npixelb[0]

    if hdr.noverfl[0]>=0:
        fid_pos = fid_pos + hdr.npixelb[1]*hdr.noverfl[0]
    else:
        fid_pos = 512*hdr.hdrblks+hdr.nrows*hdr.ncols*hdr.npixelb[0]

    fid.seek(fid_pos)
    #read the 1 byte overflow table
    if be_verbose: print("reading 1Byte overflow")
    for i in range(nof_1b_overflows):
        data_string = fid.read(2)
        b1_overflow_tab.append(struct.unpack('H',data_string)[0])

    #since the overflow data is stored as a muliple of 16:
    #have to take padding into account
    tot_size = nof_1b_overflows*2



    #read the two byte overflow table
    if be_verbose: print("reading 2Byte overflow")
    for i in range(nof_2b_overflows):
        data_string = fid.read(4)
        b2_overflow_tab.append(struct.unpack('L',data_string)[0])



    if v11_flag:
        return [b1_overflow_tab,b2_overflow_tab]
    else:
        return [overflow_tab]


def load(filename,h5file,h5group):
    """
    Load the data from a CCD data file given by 'filename'. The function takes
    the following required input arguments:
     
     filename ............ filename of the CCD data file

    return values:
     data ................ an array of size (n,n) of type int with the CCD data
     hdf ................. the header class of the file
    """

    try:
        fid = open(filename,mode="r")
    except:
        print 'error opening file %s \n' %(filename)
        return 0

    hdr = bruker_header.Header(fid)

    #print some information of the file:
    if be_verbose:
        if hdr.format==86:
            print("file format 86: frame compression format (old)")
        elif hdr.format==100:
            print("file format 100: image compression format (new)")

        if hdr.version == 11:
            print("header version (11): fixed format/free line")
        elif hdr.version==101:
            print("header version (101): free format header")


    if be_verbose: print("load raw image data")
    raw_data = load_data(hdr,fid)

    if be_verbose: print("load underflow data")
    underflow_data = load_underflow(hdr,fid)

    if be_verbose: print("load overflow data")
    [b1_overflow_tab,b2_overflow_tab] = load_overflow(hdr,fid)

    #loop now over the raw data and replace all values with 255 by the
    #values in the 2byte overflow table
    if be_verbose: print("do image decompression")
    ov_1b_counter = 0
    ov_2b_counter = 0
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            if raw_data[i,j]==255:
                raw_data[i,j]=b1_overflow_tab[ov_1b_counter]
                ov_1b_counter = ov_1b_counter + 1
            if raw_data[i,j]==65535:
                raw_data[i,j] = b2_overflow_tab[ov_2b_counter]
                ov_2b_counter = ov_2b_counter + 1

    return [raw_data,hdr]
