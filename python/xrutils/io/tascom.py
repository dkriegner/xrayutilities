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
Abstraction of the Tascom data format to HDF5
=============================================
Tascom stores its data in an ASCII format. There are two major cases when TASCOM
is writting data: line data with the suffix *.dat and data from an MCA with suffix
*.det. In both cases the data file consits of a header and a data block. 
Depending on what has been measured this module provides the user some controll over
how the measurements are imported (this is especially important for PSD data where 
TASCOM stores a single file for every single measurement with the MCA, this causes
usually the production of a huge amount of files). 
Both file types are stored in a different way:
 -> dat files are stored as tables where the header information is appended as 
    HDF5 attributes
 -> det files are stored as arrays where the header information is appended as 
    HDF5 attributes
Tascom is useing a template approach to generate a filename with
temp+number.dat/det
The pure filename without path and suffix is used as name for the HDF5 storage 
object.

To make the module more user-friendly not only import functions are provided 
but also functions to import a bulk of files into a HDF5 group. This is implemented 
in the following functions:
 dat2hdf5 ............ imports a single dat file
 det2hdf5 ............ imports a single det file
 dats2hdf5 ........... imports several dat files 
 dets2hdf5 ........... imports several det files

design note 20.11.2006:
The concept of commands as described above has been successfully implemented
for various dataformats. However, in many cases it has turned out not to be
that handy since everytime a file should be visualized it has to be transfered
first to a HDF5 file. 
"""

import tables
import re
import os
from scipy.io import read_array

filedate = re.compile(r"^#fdt")
file_extract = re.compile(r"\S+")
rem_blank = re.compile(r"\s+")
hparam_name  = re.compile(r"^#fih")
hparam_value = re.compile(r"^#fhp")
motorpos = re.compile(r"^#fmp")
mopo_number = re.compile(r"\d+[:]\s")
mopo_equal = re.compile(r"\s[=]\s")
#mopo_name = re.compile(r"[A-Za-z0-9]+\s")
mopo_name = re.compile(r"[A-Za-z]+\w*\s")
mopo_value = re.compile(r"[-]*\d+[.]\d*\s*")
colnames = re.compile(r"^#fip")

#some regular expressions to determine if a variable is an integer type 
#or a float type
int_type = re.compile(r"[-]*[0-9^\.]+")
float_type = re.compile(r"[-]*\d+[\.]\d*")
data_line = re.compile(r"\s*\d*.*\n")

class tascom_data_table(tables.IsDescription):
    pass        


def read_tascom_dat(fileobject,**keyargs):

    if isinstance(fileobject,file):
        fid = fileobject
        filename = fid.name
    elif isinstance(fileobject,string):
        try:
            fid = open(fileobject)
            filename = fileobject
        except:
            print("XU.io.tascom.read_tascom_dat: error opening file %s" %(filename))
            return None
    else:
        print("XU.io.tascom.read_tascom_dat: fileobject must be either be a file descriptor or a filename")
        return None

    #evaluate optinal keyword arguments
    if "dartmp" in keyargs:
        dar_template = keyargs["dartmp"]
    else:
        dar_tmplate = None

    #initialize column dictionary
    coldict = {}

    #have to read now until column headers
    line_buffer = " "
    while line_buffer != "":
        line_buffer = fid.readline()
        line_buffer = line_buffer.strip()
        line_buffer = rem_blank.sub(" ",line_buffer)

        #found the row with the column names
        if colnames.match(line_buffer):
            line_list = line_buffer.split(" ")
            name_list = line_list.remove("#fip") #remove the leading tascom token
            
            #break the reading loop
            break
        
    #now read the data columns
    data = read_array(filename)

    #finally we have to sort the data to the column dict
    for i in range(len(namelist)):
        coldict[namelist[i]] = data[:,i]

    #we have now the dat file data but we have to handle also MCA data automatically
    if "SEAR" in coldict and dar_template != None:
        #read MCA data from the files
        filenos = coldict["SEAR"]
        dar_tmp_len = len(dar_template)

        #loop over all file numbers
        for i in range(len(filenos)):
            fileno = filenos[i]
            fileno = int(fileno)
            filenostr = "%i" %(fileno)

            if dar_tmp_len+len(filenostr) > 8:
                diff = abs(dar_tmp_len-len(filenostr))
                filetemp = dar_template[:-diff]+filenostr+".dar"
            elif dar_tmp_len+len(filenostr) < 8:
                diff = abs(dar_tmp_len-len(filenostr))
                filetemp = dar_template+diff*"0"+filenostr+".dar"
            else:
                filetemp = dar_template + filenostr + ".dar"

            #finally we have to read the data
            dardata = read_array(filetemp)

            if "MCA" not in coldict:
                coldict["MCA"] = Numeric.array(filenos.shape[0],dardata.shape[1])

            coldict["MCA"][i,:] = dardata[:]
            

    #in the end we define a dataset object that will be returned to the 
            
            

def read_tascom_dar():
    pass

def read_dat(fid,h5file,h5group,tabname,comment):
    """
    read_dat(fid,h5file,h5group):
    Read a HDF5 data file and returns the result as a table.
    input arguments:
      fid .......... python file object
      h5file ....... HDF5 file where the data should be stored
      h5group ...... HDF5 group where the data should be stored
      tabname ...... name of the table
      comment ...... comment for the table
      
    return value:
      h5tab ........ HDF5 table construction for the data
    """
    
    #reset file pointer
    fpos_old = fid.tell()
    fid.seek(0)
    
    mopo_names = []
    mopo_values = []
    file_param_dict = {}
    mopo_dict = {}
    names = []
    values = []
    
    line_counter = 0
    
    while True:
        data_string = fid.readline()
        
        #step through all parameters:
        if filedate.match(data_string):
            #extract filename and date of measurement
            data_string = ''.join(filedate.split(data_string))
            data_string = data_string.strip()
            data_string = rem_blank.sub(' ',data_string)
            data_list = file_extract.findall(data_string)
            file_string = data_list[0]
            date_string = data_list[1]+"/"+data_list[2]
            time_string = data_list[3]
            
        if hparam_name.match(data_string):
            data_string = ''.join(hparam_name.split(data_string))
            data_string = data_string.strip()
            data_string = rem_blank.sub(' ',data_string)
            param_names = file_extract.findall(data_string)     
            
        if hparam_value.match(data_string):
            data_string = ''.join(hparam_value.split(data_string))
            data_string = data_string.strip()
            data_string = rem_blank.sub(' ',data_string)
            param_values = file_extract.findall(data_string)
            
            for i in range(len(param_values)):
                param_values[i] = float(param_values[i])

            #build the dictionary with the file parameters
            for i in range(len(param_values)):
                file_param_dict[param_names[i]] = param_values[i]
                
        if motorpos.match(data_string):
            data_string = ''.join(motorpos.split(data_string))
            data_string = data_string.strip()
            data_string = mopo_number.sub('',data_string)
            data_string = mopo_equal.sub('',data_string)
            data_string = rem_blank.sub(' ',data_string)        
            names = mopo_name.findall(data_string)
            values = mopo_value.findall(data_string)    
            for i in names:
                mopo_names.append(i)
                
            for i in values:
                mopo_values.append(float(i))                                                    
        
        if colnames.match(data_string):
            data_string = ''.join(colnames.split(data_string))
            data_string = data_string.strip()
            data_string = rem_blank.sub(' ',data_string)
            column_names = file_extract.findall(data_string)    
        
            #stop read loop 
            break
            
    tab_dict = {}

    #build the dictionary with the motorpositions
    for i in range(len(mopo_values)):       
        mopo_dict[mopo_names[i]]=mopo_values[i]
        
    #read the data
    while True:
        data_string = fid.readline()    
        
        if not data_line.match(data_string):
            print("XU.io.tascom.read_dat: finished with reading data")
            break
            
        data_string = data_string.strip()       
        data_string = rem_blank.sub(" ",data_string)
        data_list = file_extract.findall(data_string)                   
        
        #determine the data types of the read data values       
        if line_counter == 0:
            for i in range(len(data_list)):
                #check of which type the columne is: 
                #ATTENTION: THE ORDER OF ASKING IF FLOAT OR INT IS IMPORTANT - SINCE
                #INTEGER IS A SUBSET OF FLOAT IN REGEXP
                if float_type.match(data_list[i]): 
                    tab_dict[column_names[i]]=tables.Col("Float")
                elif int_type.match(data_list[i]): 
                    tab_dict[column_names[i]]=tables.Col("Int")
                else:
                    print("XU.io.tascom.read_dat: unsupported data type")
                
            #create the table object in the HDF5 file
            tab = h5file.createTable(h5group,tabname,tab_dict,comment)          

        #append data to the table:
        for i in range(len(data_list)):
            if float_type.match(data_list[i]):
                tab.row[column_names[i]]=float(data_list[i])
            elif int_type.match(data_list[i]):
                tab.row[column_names[i]]=float(data_list[i])
        tab.row.append() #append the row to the table
            
            
        line_counter = line_counter + 1
                        
    tab.flush() #flush the table to the HDF5 file
    
    #set the file header information as attributes
    #general information about the data file
    tab.attrs.filename = file_string
    tab.attrs.date = date_string
    tab.attrs.time = time_string
    #write the general file parameters
    tab.attrs.file_paramters = file_param_dict
    tab.attrs.motor_positions = mopo_dict
    

    
    #append file parameters and motor positions as dictionaries
    
    return None
            
                
            
            
def dat2hdf5(filename,h5file,h5group,**optargs):
    """
    tas2hdf5(filename):
    Converts a Tascom ASCII datafile into an HDF5 file structure.
    Input arguments:
       filename .................. full name of the file
       h5file .................... HDF5 file where the data should be stored
       h5group ................... the HDF5 group where to put the scan    
    optional input arguments:
       path ...................... the path to the file if it is not in the
                                   current directory   
       comment ................... add a custom comment to the data set
    """
    
    if "path" in optargs:
        path = optargs["path"]
    else: 
        path = "."
        
    if "comment" in optargs:
        comment = optargs["comment"]
    else:
        comment = os.path.join(path,filename)
        
    tabname = os.path.splitext(filename)[0]
        
    
    try:
        fid = open(os.path.join(path,filename),mode="r")
    except:
        print("XU.io.tascom.dat2hdf5: error opening file %s" %(filename))
        
    h5tab = read_dat(fid,h5file,h5group,tabname,comment)

def dats2hdf5(filepattern,h5file,h5group,scanrange,**optargs):
    """
    dats2hdf5(filepattern,h5file,h5group,scanrange)
    Import a complete range of dat files to HDF5. The required input arguments are:
       filepattern ................. pattern for the filename
       h5file ...................... HDF5 file object
       h5group ..................... group where the data should be stored
    scanrange ................... range of integers determining which scan numbers 
                          should be stored.
    optional input arguments:
       path .................. path to the filenames
    """
    
    #generate list of filenames using the Tascom filename convention
    # 4 characters template + 4 digits filenumber
    filename_list = []
    for i in scanrange:
        filename_list.append(filepattern + "%04i" %(i))

    for name in filename_list:
        if "path" in optargs:
            dat2hdf5(name,h5file,h5group,path=optargs["path"])
        else:
            dat2hdf5(name,h5file,h5group)
    
    
def det2hdf5(filename,h5file,**keyargs):
    """
    det2hdf5(filename,h5file,**keyargs):
    Read a single TASCOM detector file and convert it to an HDF5 file.

    required input arguments:
    filename ............. the TASCOM detector filename
    h5file ............... an hdf5 file object

    optional keyword arguments:
    group ................ HDF5 group object where to store the file
    path ................. path where to read the file from
    """


    #handle keyword arguments
    if "group" in keyargs:
        h5group = keyargs["group"]
    else:
        h5group = h5file.root

    if "path" in keyargs:
        path = keyargs["path"]
        fullname = os.path.join(path,filename)
    else:
        fullname = filename
        
    #open the file for reading
    try:
        data = rio.read_array(filename,"r")
    except:
        print("XU.io.tascom.det2hdf5: error reading file %s !" %filename)
        return None

    
    
    return 0

def dets2hdf5():
    """
    dets2hdf5():
    Import a set of Tascom MCA files an store them into a HDF5 group.
    """
    pass

    
        
