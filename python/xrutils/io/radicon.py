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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@aol.at>

"""
python module for converting radicon data to
HDF5
"""

import re
import tables
import struct
import numpy
import os.path

from .. import config

rdc_start = re.compile(r"^START")
rdc_end   = re.compile(r"^END")

rdc_mopo = re.compile(r"^[A-Z]+=.*")
rdc_param = re.compile(r"^.*:.+")

rdc_colname = re.compile(r"^-+")
rdc_data_line = re.compile(r"(\s*[0-9\.]\s*)+")

rem_blank = re.compile(r"\s+")  #remove all multiple blanks in a line
blank_extract = re.compile(r"\S+") #extract all columns seperated by single blanks

def rad2hdf5(h5,rdcfile,**keyargs):
    """
    rad2hdf5(h5,rdcfile,**keyargs):
    Converts a RDC file to an HDF5 file.
    
    Required input arguments:
    h5 .................. HDF5 object where to store the data
    rdcfile ............. name of the RDC file

    optional (named) input arguments:
    h5path .............. Path in the HDF5 file where to store the data
    rdcpath ............. path where the RDC file is located (default
                          is the current working directory)
                
    """

    if keyargs.has_key("rdcpath"):
        rdcpath = keyargs["rdcpath"]
    else:
        rdcpath = "."

    rdcfilename = os.path.join(rdcpath,rdcfile)

    if keyargs.has_key("h5path"):
        h5path = keyargs["h5path"]
    else:
        h5path = h5.root

    try:
        rdcfid = open(rdcfilename,mode="r")
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.io.rad2hdf5: successfully opened RDC file %s for reading" %rdcfilename)
    except:
        raise IOError("error opening RDC file %s !" %rdcfilename)
        
    line_buffer = " "
    while True:

        #read a line from the file
        line_buffer = rdcfid.readline()
        
        if line_buffer=="":
            if config.VERBOSITY >= config.DEBUG:
                print("XU.io.rad2hdf5: reached end of RDC file")
            break
        
        line_buffer = line_buffer.strip()

        if rdc_start.match(line_buffer):
            #reaching the start of a new scan - reinit all variables
            motor_list = [];       #list with the names of the motors in the motor pos
                                   #table
            motor_pos_list = [];   #list of the initial motor positions
            param_name_list = [];  #parameter names
            param_value_list = []; #parameter values

            col_name_list = [];    #list with column names
            tab_dict = {};         #dictionary for the table            
            

        if rdc_param.match(line_buffer):
            data_buffer = re.compile(r":\s+").split(line_buffer)
            data_buffer[0] = data_buffer[0].replace("/","_")
            param_name_list.append(data_buffer[0])
            param_value_list.append(data_buffer[1])
            if data_buffer[0]=="Scan":
                param_name_list.append("scantype")
                line_buffer = rdcfid.readline()
                line_buffer = line_buffer.strip()
                param_value_list.append(line_buffer)

        if rdc_mopo.match(line_buffer):            
            data_buffer = re.compile(r"=\s+").split(line_buffer)
            motor_list.append(data_buffer[0])
            motor_pos_list.append(data_buffer[1])

        if rdc_colname.match(line_buffer):
            line_buffer = rdcfid.readline()
            line_buffer = line_buffer.strip()
            col_name_list = re.compile(r"\s+").split(line_buffer)
            #perform an extra read cycle
            line_buffer = rdcfid.readline()

            #after the column names have been read - build the table and
            #add the header attributes
            tab_name = param_value_list[0]+'_'+param_value_list[1]
            tab_title = "Scan %s of type %s on sample %s" %(param_value_list[0],\
                                                            param_value_list[1],\
                                                            param_value_list[2])
            #build the table dictionary
            for name in col_name_list:
                tab_dict[name] = tables.FloatCol()

            #create the new table object
            table = h5.createTable(h5path,tab_name,tab_dict,tab_title)

            #add the attributes (parameters and initial motor positions)
            for i in range(len(param_name_list)):
                param_name = param_name_list[i]
                param_value = param_value_list[i]
                param_name = param_name.replace(" ","_")
                param_name = param_name.replace(".","")
                param_name = param_name.replace("-","_")
                param_name = param_name.replace("(","")
                param_name = param_name.replace(")","")
                table.attrs.__setattr__(param_name,param_value)

            for i in range(len(motor_list)):
                table.attrs.__setattr__(motor_list[i],motor_pos_list[i])

            #set finally the scan status to aborted (will be corrected if the
            #scan has finished properly
            table.attrs.scan_status = "ABORTED"

        if rdc_data_line.match(line_buffer):
            data_buffer = re.compile("\s+").split(line_buffer)

            #store the data in the table
            for i in range(len(data_buffer)):
                table.row[col_name_list[i]] = float(data_buffer[i])

            table.row.append()

        if rdc_end.match(line_buffer):
            table.attrs.scan_status = "SUCCEEDED"
            table.flush()
            if config.VERBOSITY >= config.INFO_ALL: 
                print("XU.io.rad2hdf5: scan finished")




    #flush the last table (for sure)
    table.flush()
    rdcfid.close()



def hst2hdf5(h5,hstfile,nofchannels,**keyargs):
    """
    hst2hdf5(h5,hstfile,**keyargs):
    Converts a HST file to an HDF5 file.

    Required input arguments:
      h5 .................. HDF5 object where to store the data
      hstfile ............. name of the HST file
      nofchannels ......... number of channels

    optional (named) input arguments:
      h5path .............. Path in the HDF5 file where to store the data
      hstpath ............. path where the HST file is located (default
                            is the current working directory)
    """
    if keyargs.has_key("hstpath"):
        hstpath = keyargs["hstpath"]
    else:
        hstpath = "."

    hstfilename = os.path.join(hstpath,hstfile)

    if keyargs.has_key("h5path"):
        h5path = keyargs["h5path"]
    else:
        h5path = h5.root

    try:
        hstfid = open(hstfilename,mode="r")

    except:
        raise IOError("XU.io.hst2hdf5: error opening HST file %s !" %hstfilename)

    filters = tables.Filters(complevel=5,complib="zlib",shuffle=True,fletcher32=True)

    #jump the first header entry - it is nof of interest
    hstfid.seek(12,0)

    nofhists = 0

    #some format strings used to read the file
    fmt_hist = 'ii128c128c8HiId'+nofchannels*"i"
    fmt_hist_size = struct.calcsize(fmt_hist)
    
    #read the top header and determine the number of histograms
    #and the size of the histograms
    data_buffer= struct.unpack("i",hstfid.read(struct.calcsize("i")))
    nofhists = data_buffer[0]

    if config.VERBOSITY >= config.INFO_ALL: 
        print("XU.io.hst2hdf5: number of histograms found: %d" %nofhists)

    #now the table and the EArray
    table_dict = {}
    table_dict["index"] = tables.IntCol()
    table_dict["channels"]  = tables.IntCol()
    table_dict["type"] = tables.IntCol()
    table_dict["name"] = tables.StringCol(itemsize=128)
    table_dict["ExpTime"] = tables.FloatCol()
    table = h5.createTable(h5path,"MCA_info",table_dict,"MCA info table")

    atype = tables.IntAtom()
    array = h5.createEArray(h5path,"MCAarray",atype,(0,nofchannels),
            "MCA data of file %s" %(hstfilename),filters=filters)

    #setup the buffer array for storing a single spectrum
    data = numpy.zeros((nofchannels),numpy.int)

    #loop over all histograms
    for i in range(nofhists):        
        #read the header structure
        data_buffer = struct.unpack(fmt_hist,hstfid.read(fmt_hist_size))
        table.row["index"] = i
        table.row["type"] = data_buffer[1]
        table.row["name"]  = (("".join(data_buffer[2:(2+128)])).replace(" ","")).strip()
        table.row["channels"] = data_buffer[0]
        table.row["ExpTime"] = data_buffer[268]

        table.row.append()
            
        #copy the data to the storage array
        for j in range(nofchannels):
            data[j] = data_buffer[269+j]


        #append the array to the EArray
        array.append([data])
        
    table.flush()
        
    hstfid.close()

def selecthst(et_limit,mca_info,mca_array):
    """
    selecthst(et_limit,mca_info,mca_array):
    Select historgrams form the complete set of recorded MCA data
    and stores it into a new numpy array. The selection is done due to a 
    exposure time limit. Spectra below this limit are ignored.

    required input arguments:
    et_limit .............. exposure time limit
    mca_info .............. pytables table with the exposure data
    mca_array ............. array with all the MCA spectra

    return value:
    a numpy array with the selected mca spectra of shape (hstnr,channels).
    """

    #read the exposure time 
    et = mca_info.cols.ExpTime[:]
    sel = numpy.zeros(et.shape,dtype=numpy.int)

    for i in range(et.shape[0]):
        if et[i]>et_limit: sel[i] = 1

    if config.VERBOSITY >= config.DEBUG:
        print("XU.io.selecthst: found %i valid arrays" %sel.sum())

    #load the data
    mca = mca_array.read()

    #create data array
    data = numpy.zeros((sel.sum(),mca.shape[1]),dtype=numpy.float)
    cnt = 0
    for i in range(sel.shape[0]):
        if sel[i]:
            data[cnt,:] = mca[i,:]
            cnt += 1

    return data

