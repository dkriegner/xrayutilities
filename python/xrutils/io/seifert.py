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
# Copyright (C) 2009-2010 Dominik Kriegner <dominik.kriegner@aol.at>

"""
a set of  routines to convert Seifert ASCII files to HDF5
in fact there exist two posibilities how the data is stored (depending on the 
use detector):
 1.) as a simple line scan (using the point detector)
 2.) as a map using the PSD

In the first case the data ist stored 
"""

import re
import tables
import numpy

from .. import config

#define some regular expressions
nscans_re = re.compile(r"^&NumScans=\d+")
scan_data_re    = re.compile(r"^\#Values\d+")
scan_partab_re  = re.compile(r"^\#ScanTableParameter")
novalues_re = re.compile(r"^&NoValues=\d+")
scanaxis_re = re.compile(r"^&ScanAxis=.*")

#some constant regular expressions
re_measparams = re.compile(r"#MeasParameter")
re_rsmparams  = re.compile(r"#RsmParameter")
re_data = re.compile(r"#Values")
re_keyvalue = re.compile(r"&\S+=\S")
re_invalidkeyword = re.compile(r"^\d+\S")
re_multiblank = re.compile(r"\s+")
re_position = re.compile(r"&Pos=[+-]*\d*\.\d*")
re_start = re.compile(r"&Start=[+-]*\d*\.\d*")
re_end   = re.compile(r"&End=[+-]*\d*\.\d*")
re_step  = re.compile(r"&Step=[+-]*\d*\.\d*")
re_time  = re.compile(r"&Time=\d*.\d*")
re_stepscan = re.compile(r"^&Start")
re_dataline = re.compile(r"^[+-]*\d*\.\d*")
re_absorber = re.compile(r"^&Axis=A6")

def repair_key(key):
    """
    repair_key(key):
    Repair a key string in the sense that the string is changed in a way that 
    it can be used as a valid Python identifier. For that purpose all blanks 
    within the string will be replaced by _ and leading numbers get an
    preceeding _.
    """

    if re_invalidkeyword.match(key):
        key = "_"+key

    #now replace all blanks
    key = key.replace(" ","_")

    return key

class SeifertHeader(object):
    def __init__(self):
        pass

    def __str__(self):
        ostr = ""
        for k in self.__dict__.keys():
            value = self.__getattribute__(k)
            if isinstance(value,float):
                ostr += k + " = %f\n" %value
            else:
                ostr += k + " = %s\n" %value

        return ostr

    def save_h5_attribs(self,obj):
        for a in self.__dict__.keys():
            value = self.__getattribute__(a)
            obj._v_attrs.__setattr__(a,value)



class SeifertMultiScan(object):
    def __init__(self,filename,m_scan,m2):
        """
        Parse data from a multiscan Seifert file.

        required input arguments:
        filename ................... name of the NJA file
        m_scan ..................... name of the scan axis
        m2 ......................... name of the second moving motor
        """

        self.Filename =filename
        try:
            self.fid = open(filename,"r")
        except:
            self.fid = None

        self.nscans = 0   #total number of scans
        self.npscan = 0   #number of points per scan
        self.ctime = 0    #counting time
        self.re_m2 = re.compile(r"^&Axis=%s\s+&Task=Drive" %m2)
        self.re_sm = re.compile(r"^&ScanAxis=%s" %m_scan)
        self.scan_motor_name = m_scan
        self.sec_motor_name = m2

        self.m2_pos = []
        self.sm_pos = []
        self.int = []
        self.n_sm_pos = 0
        self.n_m2_pos = 0

        if self.fid:
            self.parse()

    def parse(self):
        self.data = []
        m2_tmppos = 0
        self.int = []
        self.sm_pos = None 
        self.m2_pos = []
        s = 0
        e = 0
        d = 0

        while True:
            lb = self.fid.readline()
            if not lb: break
            lb = lb.strip()
            
            #the first thing needed is the number of scans in the fiel
            if nscans_re.match(lb):
                t = lb.split("=")[1]
                self.nscans = int(t)
                self.n_m2_pos = int(t)

            if self.re_m2.match(lb):
                t = re_position.findall(lb)[0]
                t = t.split("=")[1]
                m2_tmppos = float(t)

            if novalues_re.match(lb):
                t = lb.split("=")[1]
                self.n_sm_pos = int(t)
                self.m2_pos.append(m2_tmppos)
                
            if re_stepscan.match(lb):
                t = re_start.findall(lb)[0]
                t = t.split("=")[1]
                s = float(t)
                t = re_end.findall(lb)[0]
                t = t.split("=")[1]
                e = float(t)
                t = re_step.findall(lb)[0]
                t = t.split("=")[1]
                d = float(t)

            if re_dataline.match(lb):
                t = re_multiblank.split(lb)[1]
                self.int.append(float(t))

        #after reading all the data 
        self.m2_pos = numpy.array(self.m2_pos,dtype=numpy.double)
        self.sm_pos = numpy.arange(s,e+0.5*d,d,dtype=numpy.double)
        self.int = numpy.array(self.int,dtype=numpy.double)
        self.int = self.int.reshape((self.n_m2_pos,self.n_sm_pos))


    def dump2hdf5(self,h5,*args,**keyargs):
        """
        dump2hdf5(h5,*args,**keyargs):
        Saves the content of a multi-scan file to a HDF5 file. By default the
        data is stored in the root group of the file. To save data somewhere
        else the keyword argument "group" must be used. 
        
        required arguments:
        h5 ................. a HDF5 file object

        optional positional arguments:
        name for the intensity matrix
        name for the scan motor
        name for the second motor
        more then three parameters are ignored.

        optional keyword arguments:
        group ............... path to the HDF5 group where to store the data
        """

        try:
            iname = args[0]
        except:
            iname = "INT"

        try:
            smname = args[1]
        except:
            smname = self.scan_motor_name

        try:
            m2name = args[2]
        except:
            m2name = self.sec_motor_name

        if "group" in keyargs:
            g = keyargs["group"]
        else:
            g = h5.root


        a = tables.Float32Atom()
        f = tables.Filters(complevel=9,complib="zlib",fletcher32=True)

        c = h5.createCArray(g,iname,a,self.int.shape,filters=f)
        c[...] = self.int[...]
        h5.flush()

        c = h5.createCArray(g,smname,a,self.sm_pos.shape,filters=f)
        c[...] = self.sm_pos[...]
        h5.flush()

        c = h5.createCArray(g,m2name,a,self.m2_pos.shape,filters=f)
        c[...] = self.m2_pos[...]
        h5.flush()



    def dump2mlab(self,fname,*args):
        """
        dump2malb(fname,*args):
        Store the data in a matlab file. 
        """
        pass



class SeifertScan(object):
    def __init__(self,filename):
        """
        Constructor for a SeifertScan object.

        required input arguments:
        filename ................... a string with the name of the file to read

        """
        self.Filename = filename
        try:
            self.fid = open(filename,"r")
        except:
            self.fid = None

        self.hdr = SeifertHeader()
        self.data = []


        if self.fid:
            self.parse()

    def parse(self):
        if config.VERBOSITY >= config.INFO_ALL: 
            print("XU.io.SeifertScan.parse: starting the parser")
        self.data = []
        while True:
            lb = self.fid.readline()
            if not lb: break
            #remove leading and trailing whitespace and newline characeters
            lb = lb.strip()

            #every line is broken into its content
            llist = re_multiblank.split(lb)
            tmplist = []
            for e in llist:
                #if the entry is a key value pair
                if re_keyvalue.match(e):
                    (key,value) = e.split("=")
                    #remove leading & from the key
                    key = key[1:]
                    #have to manage malformed key names that cannot be used as 
                    #Python identifiers (leading numbers or blanks inside the
                    #name)
                    key = repair_key(key)

                    #try to convert the values to float numbers 
                    #leave them as strings if this is not possible
                    try:
                        value = float(value)
                    except:
                        pass

                    self.hdr.__setattr__(key,value)
                else:
                    try:
                        tmplist.append(float(e)) 
                    except:
                        pass
                    
            if tmplist!=[]:
                self.data.append(tmplist)

        #in the end we convert the data list to a numeric array
        self.data = numpy.array(self.data,dtype=numpy.float)

    
    def dump2h5(self,h5,*args,**keyargs):
        """
        dump2h5(h5,**keyargs):
        Save the data stored in the Seifert ASCII file to a HDF5 file. 


        required input arguments:
        h5 ............. HDF5 file object

        optional arguments:
        names to use to store the motors. The first must be the name 
        for the intensity array. The number of names must be equal to the second 
        element of the shape of the data object.

        optional keyword arguments:
        group .......... HDF5 group object where to store the data.
        """
         
        #handle optional arguments:
        motor_names = []
        if len(args)!=0:
            for name in args:
                motor_names.append(name)
        else:
            for i in range(self.data.shape[1]-1):
                motor_names.append("motor_%i" %i)
            motor_names.append("Int")

        #evaluate optional keyword arguments:
        if "group" in keyargs:
            g = keyargs["group"]
        else:
            g = h5.root

        a = tables.FloatAtom()
        s = [self.data.shape[0]]
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.SeifertScan.dump2h5: shape of data %d" %s)

        for i in range(self.data.shape[1]):
            title = "SEIFERT data from %s" %self.Filename
            c = h5.createCArray(g,motor_names[i],a,s,title)
            c[...] = self.data[:,i][...]

        #dump the header data
        self.hdr.save_h5_attribs(g)

        h5.flush() 

    def dump2mlab(self,fname,*args):
        """
        dump2mlab(fname):
        Save the data from a Seifert scan to a matlab file.

        required input arugments:
        fname .................. name of the matlab file

        optional position arguments:
        names to use to store the motors. The first must be the name 
        for the intensity array. The number of names must be equal to the second 
        element of the shape of the data object.

        """
        pass


