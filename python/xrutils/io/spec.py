#a threaded class for observing a SPEC data file
#Motivation:
# SPEC files can become quite large. Therefore, subsequently reading
#the entire file to extract a single scan is a quite cumbersome procedure.
#This module is a proof of concept code to write a file observer starting 
#a reread of the file starting from a stored offset (last known scan position)
#
#
#
#

import re
import numpy
import os
import time
import tables
try: from matplotlib import pylab
except RuntimeError: print "Warning: spec class plotting functionality not available"

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



DEBUG_FLAG = False;

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
            print "unknown scan status flag - set to CORRUPTED"
        
        #setup the initial motor positions dictionary - set the motor names
        self.init_motor_pos = {}    #dictionary holding the initial motor positions        
        for i in range(len(imopnames)):
            self.init_motor_pos["INIT_MOPO_"+imopnames[i]] = float(imopvalues[i])

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
            
        if DEBUG_FLAG:
            print "number of channels: ",self.mca_channels
            print "number of columns: ",self.mca_column_format
            print "number of lines to read for MCA: ",self.mca_nof_lines            

        

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
            print "scan has been aborted - no data available!"
            self.data = None
            return None

        if not self.has_mca:
            print "scan contains no MCA data"
        
        #save the actual position of the file pointer
        oldfid = self.fid.tell()
        self.fid.seek(self.doffset,0)
        
        #create dictionary to hold the data
        if self.has_mca:
        	type_desc = {"names":self.colnames+["MCA"],"formats":len(self.colnames)*[numpy.float32]+\
            	         [(numpy.uint32,self.mca_channels)]}
        else:
        	type_desc = {"names":self.colnames,"formats":len(self.colnames)*[numpy.float32]}
        	
        if DEBUG_FLAG: 
            print type_desc            

        record_list = [] #from this list the record array while be built
        
        mca_counter = 0
        scan_aborted_flag = False

        while True:
            line_buffer = self.fid.readline()
            line_buffer = line_buffer.strip()            
            if line_buffer=="": break #EOF
            #check if scan is broken
            if SPEC_scanbroken.findall(line_buffer) != [] or scan_aborted_flag:
                # need to check next line(s) to know if scan is resumed
                # read until end of comment block or end of file
                if not scan_aborted_flag:
                    scan_aborted_flag = True
                    self.scan_status = "ABORTED"
                    print "Scan aborted"
                    continue
                elif SPEC_scanresumed.match(line_buffer):
                    self.scan_status = "OK"
                    scan_aborted_flag = False
                    print "Scan resumed"
                    continue
                elif SPEC_commentline.match(line_buffer):
                    continue
                else:
                    break
            
            if SPEC_headerline.match(line_buffer) or SPEC_commentline.match(line_buffer): break

            if mca_counter == 0:
                #the line is a scalar data line                
                line_list = SPEC_num_value.findall(line_buffer)
                if DEBUG_FLAG: 
                    print line_buffer
                    print "read scalar values",line_list
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
        print "%s: %d %d %d" %(self.name,len(record_list),len(record_list[0]),len(type_desc["names"]))
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
            print "error: plot functionality not available"
            return

        if keyargs.has_key("newfig"):
            newfig = keyargs["newfig"]
        else:
            newfig = True

        if keyargs.has_key("logy"):
            logy = keyargs["logy"]
        else:
            logy = False

        try:
            xname = args[0]
            xdata = self.data[xname]
        except:
            print "name of the x-axis is invalid!"
            return Nont

        alist = args[1:]
        leglist = []
        
        if len(alist)%2 != 0:
            print "wrong number of yname/style arguments!"
            return None
    
        if newfig:
            pylab.figure()
            pylab.subplots_adjust(left=0.08,right=0.95)

        for i in range(0,len(alist),2):
            yname = alist[i]
            ystyle = alist[i+1]
            try:
                ydata = self.data[yname]
            except:
                print "no column with name %s exists!" %yname
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

            
        
        
    def Save2HDF5(self,h5file,**keyargs):
        """
        Save2HDF5(h5file,**keyargs):
        Save a SPEC scan to an HDF5 file. The method creates a group with the name of the 
        scan and stores the data there as a table object with name "data". By default the 
        scan group is created under the root group of the HDF5 file.  
        The title of the scan group is ususally the scan command. Metadata of the scan 
        are stored as attributes to the scan group. Additional custom attributes to the scan 
        group can be passed as a dictionary via the optattrs keyword argument.
        
        input arguments:
        h5file ..................... a HDF5 file object
        
        optional keyword arguments:
        group ...................... name or group object of the HDF5 group where to store the data
        title ............... a string with the title for the data
        desc ................ a string with the description of the data       
        optattrs ............ a dictionary with optional attributes to store for the data
        comp ................ activate compression - true by default
        """
        
        #check if data object has been already written
        if self.data == None:
            print "No data has been read so far - call ReadData method of the scan"
            return None
		
        #parse keyword arguments:
        if keyargs.has_key("group"):
            if isinstance(keyargs["group"],str):
                rootgroup = h5file.getNode(keyargs["group"])
            else:
                rootgroup = keyargs["group"]
        else:
            rootgroup = "/"
            
        if keyargs.has_key("comp"):
            compflag = keyargs["comp"]
        else:
            compflag = True
				
        if keyargs.has_key("title"):
            group_title = keyargs["title"]
        else:
            group_title = self.name
        group_title  = group_title.replace(".","_")
		
        if keyargs.has_key("desc"):
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
                g = h5file.createGroup(rootgroup,group_title,group_desc)
                break
            except:
                #if the group already exists the name must be changed and 
                #another will be made to create the group.
                if self.ischanged:
                    g = h5file.removeNode(rootgroup,group_title,recursive=True)
                else:
                    group_title = group_title + "_%i" %(copy_count)
                    copy_count = copy_count + 1
                
        if compflag:
            tab = h5file.createTable(g,"data",tab_desc_dict,"scan data",filters=f)        
        else:
            tab = h5file.createTable(g,"data",tab_desc_dict,"scan data")        
            
        for rec in self.data:
            for cname in rec.dtype.names:
                tab.row[cname] = rec[cname]					
            tab.row.append()
			
        #finally after the table has been written to the table - commit the table to the file
        tab.flush()		        

        #write attribute data for the scan
        g._v_attrs.ScanNumber = numpy.uint(self.nr)
        g._v_attrs.Command = self.command;
        g._v_attrs.Date = self.date
        g._v_attrs.Time = self.time
        
        #write the initial motor positions as attributes
        for k in self.init_motor_pos.keys():
            g._v_attrs.__setattr__(k,numpy.float(self.init_motor_pos[k]))
            
        #if scan contains MCA data write also MCA parameters
        g._v_attrs.mca_start_channel = numpy.uint(self.mca_start_channel)
        g._v_attrs.mca_stop_channel = numpy.uint(self.mca_stop_channel)
        g._v_attrs.mca_nof_channels = numpy.uint(self.mca_channels)
            
        if keyargs.has_key("optattrs"):
            optattrs = keyargs["optattrs"]
            for k in optattrs.keys():
                g._v_attrs.__setattr__(k,opattrs[k])
                        
        h5file.flush();            

class SPECFile(object):
    """
    class SPECFile:
    This class represents a single SPEC file. The class provides 
    methodes for updateing an already opened file which makes it particular 
    interesting for interactive use.
    
    """
    def __init__(self,filename,**keyargs):
        self.filename = filename
        
        if keyargs.has_key("path"):
            self.full_filename = os.path.join(keyargs["path"],filename)     
        else:
            self.full_filename = filename           
            
        #list holding scan objects
        self.scan_list = []
        #open the file for reading
        try:
            self.fid = open(self.full_filename,"r")            
            self.last_offset = self.fid.tell()
        except:
            print "error opening SPEC file %s" %(self.full_filename)
            self.fid = None
            self.last_offset = 0
            return
            
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
		
    def Save2HDF5(self,h5,**keyargs):
        """
        Save2HDF5(h5):
        Save the entire file in an HDF5 file. For that purpose a group is set up in the root 
        group of the file with the name of the file without extension and leading path.
        If the method is called after an previous update only the scans not written to the file meanwhile are 
        saved.
        
        required arguments:
        h5 .................... a HDF5 file object
        
        optional keyword arguments:
        comp .................. activate compression - true by default
        name .................. optional name for the file group
        """
        
        try:
            g = h5.createGroup("/",os.path.splitext(self.filename)[0],"Data of SPEC - File %s" %(self.filename))
        except:
            g = h5.getNode("/"+os.path.splitext(self.filename)[0])
            
        if keyargs.has_key("comp"):
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
            print "error opening SPEC file %s" %(self.full_filename)
            self.fid = None

        #before reparsing the SPEC file update the fids in all scan objects
        print "update FID for actual scans ..."
        for scan in self.scan_list:
            scan.fid = self.fid

        #reparse the SPEC file
        print "reparsing file for new scans ...";
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
            if SPEC_initmoponames.match(line_buffer) and not scan_started:
                line_buffer = SPEC_initmoponames.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                self.init_motor_names = self.init_motor_names + SPEC_multi_blank.split(line_buffer)                

            #if the line marks the beginning of a new scan
            elif SPEC_scan.match(line_buffer) and not scan_started: 
                if DEBUG_FLAG: print "found scan"               
                line_list = SPEC_multi_blank.split(line_buffer)
                scannr = int(line_list[1])
                scancmd = "".join(" "+x+" " for x in line_list[2:])
                scan_started = True
                scan_has_mca = False
                scan_header_offset = self.last_offset               
                scan_status = "OK"
                print "processing scan nr. %i ..." %scannr

            #if the line contains the date and time information
            elif SPEC_datetime.match(line_buffer) and scan_started:
                if DEBUG_FLAG: print "found date and time"
                #fetch the time from the line data
                time = SPEC_time_format.findall(line_buffer)[0]
                line_buffer = SPEC_time_format.sub("",line_buffer)
                line_buffer = SPEC_datetime.sub("",line_buffer)
                date = SPEC_multi_blank.sub(" ",line_buffer)                   

            #if the line contains the integration time
            elif SPEC_exptime.match(line_buffer) and scan_started:
                if DEBUG_FLAG: print "found exposure time"
                itime = float(SPEC_num_value.findall(line_buffer)[0])                                     
            #read the initial motor positions
            elif SPEC_initmopopos.match(line_buffer) and scan_started:
                if DEBUG_FLAG: print "found initial motor positions"
                line_buffer = SPEC_initmopopos.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                line_list = SPEC_multi_blank.split(line_buffer)
                for value in line_list:
                    init_motor_values.append(float(value))

            #if the line contains the number of colunmns
            elif SPEC_nofcols.match(line_buffer) and scan_started:
                if DEBUG_FLAG: print "found number of columns"
                line_buffer = SPEC_nofcols.sub("",line_buffer)
                line_buffer = line_buffer.strip()
                nofcols = int(line_buffer)

            #if the line contains the column names
            elif SPEC_colnames.match(line_buffer) and scan_started:
                if DEBUG_FLAG: print "found column names"
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
                #written 
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
        if keyargs.has_key("path"):
            self.full_filename = os.path.join(keyargs["path"],self.filename)
        else:
            self.full_filename = self.filename
            
        try:
            self.fid = open(self.full_filename,"r")
        except:
            print "cannot open log file %s" %(self.full_filename)
            
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

