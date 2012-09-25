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
a threaded class for observing a SPEC data file
Motivation:
 SPEC files can become quite large. Therefore, subsequently reading
the entire file to extract a single scan is a quite cumbersome procedure.
This module is a proof of concept code to write a file observer starting
a reread of the file starting from a stored offset (last known scan position)
"""

import re
import numpy
import os
import time
import tables

# relative imports from xrutils
from .. import config
from ..exception import InputError

try:
    from matplotlib import pylab
except RuntimeError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.io.spec: warning; spec class plotting functionality not available")

#define some uesfull regular expressions
SPEC_time_format = re.compile(r"\d\d:\d\d:\d\d")
SPEC_multi_blank = re.compile(r"\s+")
SPEC_multi_blank2 = re.compile(r"\s\s+")
SPEC_num_value = re.compile(r"[+-]*\d*\.*\d*e*[+-]*\d+") #denotes a numeric value
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
scan_status_flags = ["OK","ABORTED","CORRUPTED"]

class SPECMCA(object):
    """
    SPECMCA - represents an MCA object in a SPEC file.
    This class is an abstract class not itended for being used directly.
    Instead use one of the derived classes SPECMCAFile or SPECMCAInline.

    """
    def __init__(self,nchan,roistart,roistop):
        self.n_channels = nchan
        self.roi_start  = roistart
        self.roi_stop   = roistop

class SPECMCAFile(SPECMCA):
    def __init__(self):
        SPECMCA.__init__(self)

    def ReadData(self):
        pass

class SPECMCAInline(SPECMCA):
    def __init__(self):
        SPEMCA.__init__(self)

    def ReadData(self):
        pass

class SPECScan(object):
    """
    class SPECScan:
    Represents a single SPEC scan.
    """
    def __init__(self,name,scannr,command,date,time,itime,colnames,hoffset,
                 doffset,fid,imopnames,imopvalues,scan_status):
        """
        Constructor for the SPECScan class.

        required arguments:
        name ............. name of the scan
        scannr ........... Number of the scan in the specfile
        command .......... command used to write the scan
        date ............. starting date of the scan
        time ............. starting time of the scan
        itime ............ integration time
        hoffset .......... file byte offset to the header of the scan
        doffset .......... file byte offset to the data section of the scan
        fid .............. file ID of the SPEC file the scan belongs to
        imopnames ........ motor names for the initial motor positions array
        imopvalues ....... intial motor positions array
        scan_status ...... is one of the values

        """
        self.name = name            #name of the scan
        self.nr = scannr            #number of the scan
        self.command = command      #command used to record the data
        self.date = date            #date the command has been sent
        self.time = time            #time the command has been sent
        self.colnames = colnames    #list with column names
        self.hoffset = hoffset      #file offset where the header data starts
        self.doffset = doffset      #file offset where the data section starts
        self.fid = fid              #descriptor of the file holding the data
        self.ischanged = True       #flag to force resave to hdf5 file in Save2HDF5()

        if scan_status in scan_status_flags:
            self.scan_status = scan_status
        else:
            self.scan_status = "CORRUPTED"
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.spec.SPECScan: unknown scan status flag - set to CORRUPTED")

        #setup the initial motor positions dictionary - set the motor names
        self.init_motor_pos = {}    #dictionary holding the initial motor positions
        if len(imopnames) == len(imopvalues):
            for i in range(len(imopnames)):
                self.init_motor_pos["INIT_MOPO_"+imopnames[i]] = float(imopvalues[i])
        else:
            print("XU.io.spec.SPECScan: incorrect number of initial motor positions")
            if config.VERBOSITY >= config.INFO_ALL:
                print(imopnames)
                print(imopvalues)

        #some additional attributes for the MCA data
        self.has_mca = False       #False if scan contains no MCA data, True otherwise
        self.mca_column_format = 0 #number of columns used to save MCA data
        self.mca_channels = 0      #number of channels stored from the MCA
        self.mca_nof_lines = 0     #number of lines used to store MCA data
        self.mca_start_channel = 0 #first channel of the MCA that is stored
        self.mca_stop_channel = 0  #last channel of the MCA that is stored

        #a numpy record array holding the data - this is set using by the ReadData method.
        self.data = None

        #check for duplicate values in column names
        for i in range(len(self.colnames)):
            name = self.colnames[i]
            cnt = self.colnames.count(name)
            if cnt>1:
                #have multiple entries
                cnt = 1
                for j in range(self.colnames.index(name)+1,len(self.colnames)):
                    if self.colnames[j]==name: self.colnames[j] = name+"_%i" %cnt
                    cnt += 1

    def SetMCAParams(self,mca_column_format,mca_channels,mca_start,mca_stop):
        """
        SetMCAParams(mca_column_format,mca_channels):
        Set the parameters used to save the MCA data to the file. This method
        calculates the number of lines used to store the MCA data from the
        number of columns and the

        required input aguments:
        mca_column_format ....................... number of columns used to save the data
        mca_channels ............................ number of MCA channels stored
        mca_start ............................... first channel that is stored
        mca_stop ................................ last channel that is stored

        """
        self.has_mca = True
        self.mca_column_format = mca_column_format
        self.mca_channels = mca_channels
        self.mca_start_channel = mca_start
        self.mca_stop_channel = mca_stop

        #calculate the number of lines per data point for the mca
        self.mca_nof_lines = int(mca_channels/mca_column_format)
        if mca_channels%mca_column_format != 0:
            #some additional values have to be read
            self.mca_nof_lines = self.mca_nof_lines + 1

        if config.VERBOSITY >= config.DEBUG:
            print("XU.io.SPECScan.SetMCAParams: number of channels: %d" %self.mca_channels)
            print("XU.io.SPECScan.SetMCAParams: number of columns: %d" %self.mca_column_format)
            print("XU.io.SPECScan.SetMCAParams: number of lines to read for MCA: %d" %self.mca_nof_lines)

    def __str__(self):
        #build a proper string to print the scan information
        str_rep = "|%4i|" %(self.nr)
        str_rep = str_rep + "%50s|%10s|%10s|" %(self.command,self.time,self.date)

        if self.has_mca:
            str_rep = str_rep + "MCA: %5i" %(self.mca_channels)

        str_rep = str_rep + "\n"
        return str_rep

    def ClearData(self):
        """
        ClearData():
        Delete the data stored in a scan after it is no longer
        used.
        """

        self.__delattr__("data")
        self.data = None

    def ReadData(self):
        """
        GetData():
        Set the data attribute of the scan class.
        """

        if self.scan_status == "ABORTED":
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.SPECScan.ReadData: %s has been aborted - no data available!" %self.name)
            self.data = None
            return None

        if not self.has_mca:
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.SPECScan.ReadData: scan %d contains no MCA data" %self.nr)

        #save the actual position of the file pointer
        oldfid = self.fid.tell()
        self.fid.seek(self.doffset,0)

        #create dictionary to hold the data
        if self.has_mca:
            type_desc = {"names":self.colnames+["MCA"],"formats":len(self.colnames)*[numpy.float32]+\
                         [(numpy.uint32,self.mca_channels)]}
        else:
            type_desc = {"names":self.colnames,"formats":len(self.colnames)*[numpy.float32]}

        if config.VERBOSITY >= config.DEBUG:
            print("xu.io.SPECScan.ReadData: type descriptor: %s" %(repr(type_desc)))

        record_list = [] #from this list the record array while be built

        mca_counter = 0
        scan_aborted_flag = False

        while True:
            line_buffer = self.fid.readline()
            line_buffer = line_buffer.strip()

            #Bugfix for ESRF/BM20 data
            #the problem is that they store messages from automatic absorbers
            #in the SPEC file - need to handle this
            t = re.compile(r"^#C .* filter factor.*")
            if t.match(line_buffer): continue
            #these lines should do the job

            if line_buffer=="": break #EOF
            #check if scan is broken
            if SPEC_scanbroken.findall(line_buffer) != [] or scan_aborted_flag:
                # need to check next line(s) to know if scan is resumed
                # read until end of comment block or end of file
                if not scan_aborted_flag:
                    scan_aborted_flag = True
                    self.scan_status = "ABORTED"
                    if config.VERBOSITY >= config.INFO_ALL:
                        print("XU.io.SPECScan.ReadData: %s aborted" %self.name)
                    continue
                elif SPEC_scanresumed.match(line_buffer):
                    self.scan_status = "OK"
                    scan_aborted_flag = False
                    if config.VERBOSITY >= config.INFO_ALL:
                        print("XU.io.SPECScan.ReadData: %s resumed" %self.name)
                    continue
                elif SPEC_commentline.match(line_buffer):
                    continue
                else:
                    break

            if SPEC_headerline.match(line_buffer) or SPEC_commentline.match(line_buffer): break

            if mca_counter == 0:
                #the line is a scalar data line
                line_list = SPEC_num_value.findall(line_buffer)
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECScan.ReadData: %s" %line_buffer)
                    print("XU.io.SPECScan.ReadData: read scalar values %s" %repr(line_list))
                #convert strings to numbers
                for i in range(len(line_list)):
                    line_list[i] = float(line_list[i])

                #increment the MCA counter if MCA data is stored
                if self.has_mca:
                    mca_counter = mca_counter + 1
                    #create a temporary list for the mca data
                    mca_tmp_list = []
                else:
                    record_list.append(line_list)
            else:
                #reading and MCA spectrum
                tmp_list = SPEC_num_value.findall(line_buffer)
                for x in tmp_list:
                    mca_tmp_list.append(float(x))

                #increment MCA counter
                mca_counter = mca_counter + 1
                #if mca_counter exceeds the number of lines used to store MCA
                #data append everything to the record list
                if mca_counter > self.mca_nof_lines:
                    record_list.append(line_list + [mca_tmp_list])
                    mca_counter = 0

        #convert the lists in the data dictionary to numpy arrays
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.io.SPECScan.ReadData: %s: %d %d %d" %(self.name,len(record_list),len(record_list[0]),len(type_desc["names"])) )
        self.data = numpy.rec.fromrecords(record_list,dtype=type_desc)

        #reset the file pointer position
        self.fid.seek(oldfid,0)

    def plot(self,*args,**keyargs):
        """
        plot(*args,newfig=True,logy=False):
        Plot scan data to a matplotlib figure. If newfig=True a new
        figure instance will be created. If logy=True (default is False)
        the y-axis will be plotted with a logarithmic scale.
        """

        try: pylab.__version__
        except NameError:
            print("XU.io.SPECScan.plot: ERROR: plot functionality not available")
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

        if len(alist)%2 != 0:
            raise InputError("wrong number of yname/style arguments!")

        if newfig:
            pylab.figure()
            pylab.subplots_adjust(left=0.08,right=0.95)

        for i in range(0,len(alist),2):
            yname = alist[i]
            ystyle = alist[i+1]
            try:
                ydata = self.data[yname]
            except:
                raise InputError("no column with name %s exists!" %yname)
                continue
            if logy:
                pylab.semilogy(xdata,ydata,ystyle)
            else:
                pylab.plot(xdata,ydata,ystyle)

            leglist.append(yname)

        pylab.xlabel("%s" %xname)
        pylab.legend(leglist)
        pylab.title("scan %i %s\n%s %s" %(self.nr,self.command,self.date,self.time))
        #need to adjust axis limits properly
        lim = pylab.axis()
        pylab.axis([xdata.min(),xdata.max(),lim[2],lim[3]])




    def Save2HDF5(self,h5f,**keyargs):
        """
        Save2HDF5(h5f,**keyargs):
        Save a SPEC scan to an HDF5 file. The method creates a group with the name of the
        scan and stores the data there as a table object with name "data". By default the
        scan group is created under the root group of the HDF5 file.
        The title of the scan group is ususally the scan command. Metadata of the scan
        are stored as attributes to the scan group. Additional custom attributes to the scan
        group can be passed as a dictionary via the optattrs keyword argument.

        input arguments:
        h5f ..................... a HDF5 file object or its filename

        optional keyword arguments:
        group ...................... name or group object of the HDF5 group where to store the data
        title ............... a string with the title for the data
        desc ................ a string with the description of the data
        optattrs ............ a dictionary with optional attributes to store for the data
        comp ................ activate compression - true by default
        """
        
        closeFile=False
        if isinstance(h5f,tables.file.File):
            h5 = h5f
            if not h5.isopen:
                h5 = tables.openFile(h5f,mode='a')
                closeFile=True
        elif isinstance(h5f,str):
            h5 = tables.openFile(h5f,mode='a')
            closeFile=True
        else:
            raise InputError("h5f argument of wrong type was passed")
        
        #check if data object has been already written
        if self.data == None:
            raise InputError("XU.io.SPECScan.Save2HDF5: No data has been read so far - call ReadData method of the scan")
            return None

        #parse keyword arguments:
        if "group" in keyargs:
            if isinstance(keyargs["group"],str):
                rootgroup = h5.getNode(keyargs["group"])
            else:
                rootgroup = keyargs["group"]
        else:
            rootgroup = "/"

        if "comp" in keyargs:
            compflag = keyargs["comp"]
        else:
            compflag = True

        if "title" in keyargs:
            group_title = keyargs["title"]
        else:
            group_title = self.name
        group_title  = group_title.replace(".","_")

        if "desc" in keyargs:
            group_desc = keyargs["desc"]
        else:
            group_desc = self.command


        #create the dictionary describing the table
        tab_desc_dict = {}
        col_name_list = []
        for d in self.data.dtype.descr:
            cname = d[0]
            col_name_list.append(cname)
            if len(d[1:])==1:
                ctype = numpy.dtype((d[1]))
            else:
                ctype = numpy.dtype((d[1],d[2]))
            tab_desc_dict[cname] = tables.Col.from_dtype(ctype)


        #create the table object and fill it
        f = tables.Filters(complevel=7,complib="zlib",fletcher32=True)
        copy_count = 0
        while True:
            try:
                #if everything goes well the group will be created and the
                #loop stoped
                g = h5.createGroup(rootgroup,group_title,group_desc)
                break
            except:
                #if the group already exists the name must be changed and
                #another will be made to create the group.
                if self.ischanged:
                    g = h5.removeNode(rootgroup,group_title,recursive=True)
                else:
                    group_title = group_title + "_%i" %(copy_count)
                    copy_count = copy_count + 1

        if compflag:
            tab = h5.createTable(g,"data",tab_desc_dict,"scan data",filters=f)
        else:
            tab = h5.createTable(g,"data",tab_desc_dict,"scan data")

        for rec in self.data:
            for cname in rec.dtype.names:
                tab.row[cname] = rec[cname]
            tab.row.append()

        #finally after the table has been written to the table - commit the table to the file
        tab.flush()

        #write attribute data for the scan
        g._v_attrs.ScanNumber = numpy.uint(self.nr)
        g._v_attrs.Command = self.command
        g._v_attrs.Date = self.date
        g._v_attrs.Time = self.time

        #write the initial motor positions as attributes
        for k in self.init_motor_pos.keys():
            g._v_attrs.__setattr__(k,numpy.float(self.init_motor_pos[k]))

        #if scan contains MCA data write also MCA parameters
        g._v_attrs.mca_start_channel = numpy.uint(self.mca_start_channel)
        g._v_attrs.mca_stop_channel = numpy.uint(self.mca_stop_channel)
        g._v_attrs.mca_nof_channels = numpy.uint(self.mca_channels)

        if "optattrs" in keyargs:
            optattrs = keyargs["optattrs"]
            for k in optattrs.keys():
                g._v_attrs.__setattr__(k,opattrs[k])

        h5.flush()

        if closeFile:
            h5.close()

class SPECFile(object):
    """
    class SPECFile:
    This class represents a single SPEC file. The class provides
    methodes for updateing an already opened file which makes it particular
    interesting for interactive use.

    """
    def __init__(self,filename,**keyargs):
        self.filename = filename

        if "path" in keyargs:
            self.full_filename = os.path.join(keyargs["path"],filename)
        else:
            self.full_filename = filename

        self.filename = os.path.basename(self.full_filename)

        #list holding scan objects
        self.scan_list = []
        #open the file for reading
        try:
            self.fid = open(self.full_filename,"r")
            self.last_offset = self.fid.tell()
        except:
            self.fid = None
            self.last_offset = 0
            raise IOError("error opening SPEC file %s" %(self.full_filename))

        #initially parse the file
        self.init_motor_names = [] #this list will hold the names of the
                                   #motors saved in initial motor positions

        self.Parse()

    def __getitem__(self,index):
        return self.scan_list[index]

    def __len__(self):
        return scan_list.__len__()

    def __str__(self):
        ostr = ""
        for i in range(len(self.scan_list)):
            ostr = ostr + "%5i" %(i)
            ostr = ostr + self.scan_list[i].__str__()

        return ostr

    def Save2HDF5(self,h5f,**keyargs):
        """
        Save2HDF5(h5f):
        Save the entire file in an HDF5 file. For that purpose a group is set up in the root
        group of the file with the name of the file without extension and leading path.
        If the method is called after an previous update only the scans not written to the file meanwhile are
        saved.

        required arguments:
        h5f .................... a HDF5 file object or its filename

        optional keyword arguments:
        comp .................. activate compression - true by default
        name .................. optional name for the file group
        """

        closeFile=False
        if isinstance(h5f,tables.file.File):
            h5 = h5f
            if not h5.isopen:
                h5 = tables.openFile(h5f,mode='a')
                closeFile=True
        elif isinstance(h5f,str):
            h5 = tables.openFile(h5f,mode='a')
            closeFile=True
        else:
            raise InputError("h5f argument of wrong type was passed")
        
        try:
            g = h5.createGroup("/",os.path.splitext(self.filename)[0],"Data of SPEC - File %s" %(self.filename))
        except:
            g = h5.getNode("/"+os.path.splitext(self.filename)[0])

        if "comp" in keyargs:
            compflag = keyargs["comp"]
        else:
            compflag = True

        for s in self.scan_list:
            if ((not g.__contains__(s.name)) or s.ischanged):
                s.ReadData()
                if s.data != None:
                    s.Save2HDF5(h5,group=g,comp=compflag)
                    s.ClearData()
                    s.ischanged = False

        if closeFile:
            h5.close()

    def Update(self):
        """
        Update():
        reread the file and add newly added files. The parsing starts at the
        data offset of the last scan gathered during the last parsing run.
        """

        self.fid.close()
        try:
            self.fid = open(self.full_filename,"r")
        except:
            self.fid = None
            raise IOError("error opening SPEC file %s" %(self.full_filename))

        #before reparsing the SPEC file update the fids in all scan objects
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.SPECFile.Update: update FID for actual scans ...")
        for scan in self.scan_list:
            scan.fid = self.fid

        #reparse the SPEC file
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.io.SPECFile.Update: reparsing file for new scans ...")
        # mark last found scan as not saved to force reread
        idx = len(self.scan_list)
        if idx>0:
            lastscan = self.scan_list[idx-1]
            lastscan.ischanged=True
        self.Parse()


    def Parse(self):
        """
        method parse_file():
        Parses the file from the starting at last_offset and adding found scans
        to the scan list.
        """
        #move to the last read position in the file
        self.fid.seek(self.last_offset,0)
        scan_started = False
        scan_has_mca = False
        #list with the motors from whome the initial
        #position is stored.
        init_motor_values = []

        #read the file
        self.last_offset = self.fid.tell()
        while True:
            line_buffer = self.fid.readline()
            if line_buffer=="": break
            #remove trailing and leading blanks from the read line
            line_buffer = line_buffer.strip()

            #fill the list with the initial motor names
            if SPEC_newheader.match(line_buffer):
                self.init_motor_names = []

            elif SPEC_initmoponames.match(line_buffer):
                line_buffer = SPEC_initmoponames.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                self.init_motor_names = self.init_motor_names + SPEC_multi_blank2.split(line_buffer)

            #if the line marks the beginning of a new scan
            elif SPEC_scan.match(line_buffer) and not scan_started:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found scan")
                line_list = SPEC_multi_blank.split(line_buffer)
                scannr = int(line_list[1])
                scancmd = "".join(" "+x+" " for x in line_list[2:])
                scan_started = True
                scan_has_mca = False
                scan_header_offset = self.last_offset
                scan_status = "OK"
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.io.SPECFile.Parse: processing scan nr. %i ..." %scannr)

            #if the line contains the date and time information
            elif SPEC_datetime.match(line_buffer) and scan_started:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found date and time")
                #fetch the time from the line data
                time = SPEC_time_format.findall(line_buffer)[0]
                line_buffer = SPEC_time_format.sub("",line_buffer)
                line_buffer = SPEC_datetime.sub("",line_buffer)
                date = SPEC_multi_blank.sub(" ",line_buffer)

            #if the line contains the integration time
            elif SPEC_exptime.match(line_buffer) and scan_started:
                if config.VERBOSITY >= config.DEBUG: print("XU.io.SPECFile.Parse: found exposure time")
                itime = float(SPEC_num_value.findall(line_buffer)[0])
            #read the initial motor positions
            elif SPEC_initmopopos.match(line_buffer) and scan_started:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found initial motor positions")
                line_buffer = SPEC_initmopopos.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                line_list = SPEC_multi_blank.split(line_buffer)
                for value in line_list:
                    init_motor_values.append(float(value))

            #if the line contains the number of colunmns
            elif SPEC_nofcols.match(line_buffer) and scan_started:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found number of columns")
                line_buffer = SPEC_nofcols.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                nofcols = int(line_buffer)

            #if the line contains the column names
            elif SPEC_colnames.match(line_buffer) and scan_started:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found column names")
                line_buffer = SPEC_colnames.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                col_names = SPEC_multi_blank.split(line_buffer)

                #this is a fix in the case that blanks are allowed in
                #motor and detector names (only a single balanks is supported
                #meanwhile)
                if len(col_names)>nofcols:
                    col_names = SPEC_multi_blank2.split(line_buffer)

            elif SPEC_MCAFormat.match(line_buffer) and scan_started:
                mca_col_number = int(SPEC_num_value.findall(line_buffer)[0])
                scan_has_mca = True

            elif SPEC_MCAChannels.match(line_buffer) and scan_started:
                line_list = SPEC_num_value.findall(line_buffer)
                mca_channels = int(line_list[0])
                mca_start = int(line_list[1])
                mca_stop = int(line_list[2])

            elif SPEC_scanbroken.findall(line_buffer)!=[] and scan_started:
                #this is the case when a scan is broken and no data has been
                #written, but nevertheless a comment is in the file that tells
                #us that the scan was aborted
                try:
                    s = SPECScan("scan_%i" %(scannr),scannr,scancmd,
                                 date,time,itime,col_names,
                                 scan_header_offset,scan_data_offset,self.fid,
                                 self.init_motor_names,init_motor_values,"ABORTED")
                except:
                    scan_data_offset = self.last_offset
                    s = SPECScan("scan_%i" %(scannr),scannr,scancmd,
                                 date,time,itime,col_names,
                                 scan_header_offset,scan_data_offset,self.fid,
                                 self.init_motor_names,init_motor_values,"ABORTED")

                self.scan_list.append(s)

                #reset control flags
                scan_started = False
                scan_has_mca = False
                #reset initial motor positions flag
                init_motor_values = []


            elif SPEC_dataline.match(line_buffer) and scan_started:
                #this is now the real end of the header block.
                #at this point we know that there is enough information about the scan

                #save the data offset
                scan_data_offset = self.last_offset

                #create an SPECFile scan object and add it to the scan list
                #the name of the group consists of the prefix scan and the
                #number of the scan in the file - this shoule make it easier
                #to find scans in the HDF5 file.
                s = SPECScan("scan_%i" %(scannr),scannr,scancmd,
                             date,time,itime,col_names,
                             scan_header_offset,scan_data_offset,self.fid,
                             self.init_motor_names,init_motor_values,scan_status)
                if scan_has_mca:
                    s.SetMCAParams(mca_col_number,mca_channels,mca_start,
                                   mca_stop)

                self.scan_list.append(s)

                #reset control flags
                scan_started = False
                scan_has_mca = False
                #reset initial motor positions flag
                init_motor_values = []

            elif SPEC_scan.match(line_buffer) and scan_started:
                #this should only be the case when there are two consecutive file
                #headers in the data file without any data or abort notice of the
                #first scan
                # first store current scan as aborted than start new scan parsing

                try:
                    s = SPECScan("scan_%i" %(scannr),scannr,scancmd,
                                 date,time,itime,col_names,
                                 scan_header_offset,scan_data_offset,self.fid,
                                 self.init_motor_names,init_motor_values,"ABORTED")
                except:
                    scan_data_offset = self.last_offset
                    s = SPECScan("scan_%i" %(scannr),scannr,scancmd,
                                 date,time,itime,col_names,
                                 scan_header_offset,scan_data_offset,self.fid,
                                 self.init_motor_names,init_motor_values,"ABORTED")

                self.scan_list.append(s)

                #reset control flags
                scan_started = False
                scan_has_mca = False
                #reset initial motor positions flag
                init_motor_values = []

                # start parsing of new scan
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.io.SPECFile.Parse: found scan (after aborted scan)")
                line_list = SPEC_multi_blank.split(line_buffer)
                scannr = int(line_list[1])
                scancmd = "".join(" "+x+" " for x in line_list[2:])
                scan_started = True
                scan_has_mca = False
                scan_header_offset = self.last_offset
                scan_status = "OK"
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.io.SPECFile.Parse: processing scan nr. %i ..." %scannr)

            #store the position of the file pointer
            self.last_offset = self.fid.tell()

        #if reading of the file is finished store the data offset of the last scan as the last
        #offset for the next parsing run of the file
        self.last_offset = self.scan_list[-1].doffset


class SPECCmdLine(object):
    def __init__(self,n,prompt,cmdl,out):
        self.linenumber = n
        self.prompt = prompt
        self.command = cmdl
        self.out = out

    def __str__(self):
        ostr = "%i: %s %s" %(self.linenumber,self.prompt,self.command)
        return ostr

    def Save2HDF5(h5,**keyargs):
        pass

class SPECLog(object):
    def __init__(self,filename,prompt,**keyargs):
        self.filename = filename
        if "path" in keyargs:
            self.full_filename = os.path.join(keyargs["path"],self.filename)
        else:
            self.full_filename = self.filename

        try:
            self.fid = open(self.full_filename,"r")
        except:
            raise IOError("cannot open log file %s" %(self.full_filename))

        self.prompt = prompt
        self.prompt_re = re.compile(r"^"+self.prompt)

        self.cmdl_list = []
        self.last_offset = self.fid.tell()
        self.line_counter = 0

    def Update(self):
        pass

    def Parse(self):

        while True:
            line_buffer = self.fid.readline()
            if line_buffer == "":
                break

            line_buffer = line_buffer.strip()

            if self.prompt_re.match(line_buffer):
                line_buffer = self.prompt_re.sub("",line_buffer)
                line_buffer = line_buffer.strip()



    def __str__(self):
        ostr = ""

        for cmd in self.cmdl_list:
            ostr = ostr + cmd.__str__() + "\n"

        return ostr


def geth5_map(h5f,scans,*args,**kwargs):
    """
    function to obtain the omega and twotheta as well as intensity values
    for a reciprocal space map saved in an HDF5 file, which was created
    from a spec file by the Save2HDF5 method.

    further more it is possible to obtain even more positions from
    the data file if more than two string arguments with its names are given

    Parameters
    ----------
     h5f:     file object of a HDF5 file opened using pytables or its filename
     scans:   number of the scans of the reciprocal space map (int,tuple or list)
     *args:   names of the motors (strings)
        omname:  name of the omega motor (or its equivalent)
        ttname:  name of the two theta motor (or its equivalent)

     **kwargs (optional):
        samplename: string with the hdf5-group containing the scan data
                    if ommited the first child node of h5f.root will be used

    Returns
    -------
     [ang1,ang2,...],MAP:
                angular positions of the center channel of the position
                sensitive detector (numpy.ndarray 1D) together with all the
                data values as stored in the data file (includes the
                intensities e.g. MAP['MCA']).
    """
    
    closeFile=False
    if isinstance(h5f,tables.file.File):
        h5 = h5f
        if not h5.isopen:
            h5 = tables.openFile(h5f,mode='r')
            closeFile=True
    elif isinstance(h5f,str):
        h5 = tables.openFile(h5f,mode='r')
        closeFile=True
    else:
        raise InputError("h5f argument of wrong type was passed")

    if "samplename" in kwargs:
        h5g = h5.getNode(h5.root,kwargs["samplename"])
    else:
        h5g = h5.listNodes(h5.root)[0]

    if isinstance(scans,(list,tuple)):
        scanlist = scans
    else:
        scanlist = list([scans])

    angles = dict.fromkeys(args)
    for key in angles.keys():
        if not isinstance(key,str):
            raise InputError("*arg values need to be strings with motornames")
        angles[key] = numpy.zeros(0)
    buf=numpy.zeros(0)
    MAP = numpy.zeros(0)

    for nr in scanlist:
        h5scan = h5.getNode(h5g,"scan_%d" %nr)
        command = h5.getNodeAttr(h5scan,'Command')
        sdata = h5scan.data.read()
        if MAP.dtype == numpy.float64:  MAP.dtype = sdata.dtype
        # append scan data to MAP, where all data are stored
        MAP = numpy.append(MAP,sdata)
        #check type of scan
        notscanmotors = []
        for i in range(len(args)):
            motname = args[i]
            try:
                buf = sdata[motname]
                scanshape = buf.shape
                angles[motname] =numpy.concatenate((angles[motname],buf))
            except:
                notscanmotors.append(i)
        if len(notscanmotors) == len(args):
            scanshape = MAP.shape
        for i in notscanmotors:
            motname = args[i]
            buf = numpy.ones(scanshape) * h5.getNodeAttr(h5scan,"INIT_MOPO_%s" %motname)
            angles[motname] =numpy.concatenate((angles[motname],buf))

    retval = []
    for motname in args:
        #create return values in correct order
        retval.append(angles[motname])

    if closeFile:
        h5.close()

    return retval,MAP

