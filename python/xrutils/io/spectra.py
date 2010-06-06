#module to handle spectra data

import numpy
import os
import re
import tables
import os.path
from numpy import rec
import glob

re_wspaces = re.compile(r"\s+")
re_colname = re.compile(r"^Col")

re_comment_section = re.compile(r"^%c")
re_parameter_section = re.compile(r"^%p")
re_data_section = re.compile(r"^%d")
re_end_section  = re.compile(r"^!")
re_unit = re.compile(r"\[.+\]")
re_column = re.compile(r"^Col")
re_col_name = re.compile(r"\d+\s+.+\s*\[")
re_col_index = re.compile(r"\d+\s+")
re_col_type = re.compile(r"\[.+\]")

dtype_map = {"FLOAT":"f4"}

_absorber_factors = None

class SPECTRAFileComments(dict):
    """
    class SPECTRAFileComments:
    Class that describes the comments in the header of a SPECTRA file.
    The different comments are accessible via the comment keys.
    """
    def __init__(self):
        pass
        
    def __getattr__(self,name):
        if self.has_key(name): return self[key]
        
class SPECTRAFileParameters(dict):
    def __init__(self):
        pass
        
    def __getattr__(self,name):
        if self.has_key(name):
            return self[name]
            
    def __str__(self):
        ostr = ""
        n = len(self.keys())
        lmax_key = 0 
        lmax_item = 0
        strlist = []
        
        #find the length of the longest key
        for k in self.keys():
            if len(k)>lmax_key: lmax_key = len(k)
            
            i = self[k]
            if not isinstance(i,str): 
                #if the item is not a string it must be converted
                i = "%f" %i
                
            if len(i)>lmax_item: lmax_item = len(i)
            
        #define the format string for a single key-value pair
        kvfmt = "|%%-%is = %%-%is" %(lmax_key,lmax_item)
        
        nc = 3
        nres = len(self.keys())%nc
        nrow = (len(self.keys())-nres)/nc                
        
        cnt = 0
        ostr += (3*(lmax_key+lmax_item+4)+1)*"-"+"\n"
        ostr += "|Parameters:" +(3*(lmax_key+lmax_item))*" "+"|\n"
        ostr += (3*(lmax_key+lmax_item+4)+1)*"-"+"\n"
        for key in self.keys():
            value = self[key]            
            if not isinstance(value,str): value = "%f" %value
            
            ostr += kvfmt %(key,value)
            cnt += 1
            if cnt==3:
                ostr += "|\n"
                cnt = 0
                
        if cnt!=0: ostr += "|\n"
        ostr += (3*(lmax_key+lmax_item+4)+1)*"-"+"\n"
            
        return ostr
  
  
class SPECTRAFileDataColumn(object):
    def __init__(self,index,name,unit,type):
        self.index = int(index)
        self.name  = name
        self.unit  = unit
        self.type  = type
        
    def __str__(self):
        ostr = "%i %s %s %s" %(self.index,self.name,self.unit,self.type)
        return ostr
        
        
class SPECTRAFileData(object):
    def __init__(self):
        self.collist = []
        self.data = None
        
    def append(self,col):
        self.collist.append(col)
        
    def __getitem__(self,key):
        try:
            return self.data[key]
        except:
            print "data contains no column named: %s!" %key
        
        
    def __str__(self):
        ostr = ""
        
        #determine the maximum lenght of every column string
        lmax = 0
        for c in self.collist:
            if len(c.__str__())>lmax: lmax = len(c.__str__())
            
        lmax += 3
        
        #want to print in three columns
        nc   = 3
        nres = len(self.collist)%nc
        nrows = (len(self.collist)-nres)/nc
    
        fmtstr = "| %%-%is| %%-%is| %%-%is|\n" %(lmax,lmax,lmax)
        
        ostr += (3*lmax+7)*"-"+"\n"
        ostr += "|Column names:"+(3*lmax-8)*" "+"|\n"
        ostr += (3*lmax+7)*"-"+"\n"
        for i in range(nrows):
            c1 = self.collist[i]
            c2 = self.collist[i+nrows]
            c3 = self.collist[i+2*nrows]            
            ostr +=  fmtstr %(c1.__str__(),c2.__str__(),c3.__str__())
            
        ostr += (3*lmax+7)*"-"+"\n"
        return ostr
        
        

class SPECTRAFile(object):
    """
    class SPECTRAFile:
    Represents a SPECTRA data file. The file is read during the 
    Constructor call. This class should work for data stored at 
    beamlines P08 and BW2 at HASYLAB.

    Required constructor arguments:
    ------------------------------
    filename ................ a string with the name of the SPECTRA file

    Optional keyword arguments:
    --------------------------
    mcatmp .................. template for the MCA files
    mcastart,mcastop ........ start and stop index for the MCA files, if 
                              not given, the class tries to determine the 
                              start and stop index automatically.
    """
    def __init__(self,filename,mcatmp=None,mcastart=None,mcastop=None):
        self.filename = filename
        self.comments = SPECTRAFileComments()
        self.params   = SPECTRAFileParameters()
        self.data     = SPECTRAFileData()
        self.mca = None
        self.mca_channels = None
        
        self.Read()

        if mcatmp!=None:
            self.mca_file_template = mcatmp

            if mcastart!=None and mcastop!=None:
                self.mca_start_index = mcastart
                self.mca_stop_index = mcastop
            else:
                #try to determine the number of scans automatically
                spat = self.mca_file_template.replace("%i","*")
                print spat
                l = glob.glob(spat)
                self.mca_start_index = 1
                self.mca_stop_index = len(l)

            self.ReadMCA()

    def Save2HDF5(self,h5file,name,group="/",description="SPECTRA scan",mcaname="MCA"):
        """
        Save2HDF5(h5file,group="/",name="",description="SPECTRA scan",
                  mcaname="MCA"):
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
        
        """
        if isinstance(h5file,str):
            try:
                h5 = tables.openFile(h5file,mode="a")
            except:
                print "cannot open file %s for writting!" %h5file

        else:
            h5 = h5file
        
        #create the group where to store the data    
        try:
            g = h5.createGroup(group,name,title=description,createparents=True)
        except:
            print "cannot create group %s for writting data!" %name
            
        
        #start with saving scan comments
        for k in self.comments.keys():
            try:
                h5.setNodeAttr(g,k,self.comments[k])
            except:
                print "cannot save file comment %s = %s to group %s!" %(k,self.comments[k],name)
                
        #save scan parameters
        for k in self.params.keys():
            try:
                h5.setNodeAttr(g,k,self.params[k])
            except:
                print "cannot save file parametes %s to group %s!" %(k,name)
                
        #----------finally we need to save the data -------------------
        
        #first save the data stored in the FIO file
        
        
        
        #if there is MCA data - store this 
        if self.mca:
            a = tables.Float32Atom()
            f = tables.Filter(complib="zlib",complevel=9,flechter32=True)
            c = h5.createCArray(g,mcaname,a,self.mca.shape)
            c[...] = self.mca[...]
            
            #set MCA specific attributes
            h5.setNodeAttr(c,"channels",self.mca_channels)
            h5.setNodeAttr(c,"nchannels",self.mca_channels.shape[0])

        h5.close()


    def ReadMCA(self):
        dlist = []
        for i in range(self.mca_start_index,self.mca_stop_index+1):
            fname = self.mca_file_template %i
            data = numpy.loadtxt(fname)
            
            if i==self.mca_start_index:
                self.mca_channels = data[:,0]

            dlist.append(data[:,1].tolist())

        self.mca= numpy.array(dlist,dtype=float)
        
    def __str__(self):
        ostr = self.params.__str__()
        ostr += self.data.__str__()
        
        return ostr
        
    def Read(self):
        """
        Read():
        Read the data from the file.
        """
        try:
            fid = open(self.filename,"r")
        except:
            print "cannot open data file %s for reading!" %(self.filename)
            return None
        
        col_names = ""
        col_units = []
        col_types = ""
        rec_list = []

        while True:
            lbuffer = fid.readline()
            if lbuffer=="": break
            lbuffer = lbuffer.strip()
            
            #read the next line if the line starts with a "!"
            if re_end_section.match(lbuffer): continue
            
            #select the which section to read
            if re_comment_section.match(lbuffer): 
                read_mode = 1                               
                continue
                 
            if re_parameter_section.match(lbuffer): 
                read_mode = 2                
                continue 
                
            if re_data_section.match(lbuffer): 
                read_mode = 3                
                continue
                
            #here we decide how to proceed with the data
            if read_mode == 1:
                #read the file comments
                try:
                    (key,value) = lbuffer.split("=")
                except:
                    print "cannot interpret the comment string: %s" %(lbuffer)
                    continue 
                    
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                except:
                    pass
                self.comments[key] = value
                
            elif read_mode == 2:
                #read scan parameters
                try:
                    (key,value) = lbuffer.split("=")
                except:
                    print "cannot interpret the parameter string: %s" %(lbuffer)
                    
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                except:
                    #if the conversion of the parameter to float 
                    #fails it will be saved as a string
                    pass
                    
                self.params[key] = value                
                
            elif read_mode == 3:
                if re_column.match(lbuffer):
                    try:
                        unit = re_unit.findall(lbuffer)[0]
                        (lval,rval ) = re_unit.split(lbuffer)
                        type = rval.strip()
                        l = re_wspaces.split(lval)
                        index = int(l[1])
                        name = "".join(l[2:])
                    except:
                        unit = "NONE"
                        l = re_wspaces.split(lbuffer)
                        index = int(l[1])
                        type = l[-1]
                        name = "".join(l[2:-1])

                    
                    #store columne definition 
                    self.data.append(SPECTRAFileDataColumn(index,name,unit,type))
                    
                    
                    col_names += "%s," %name                    
                    col_types  += "%s," %(dtype_map[type])
                else:
                    #read data
                    dlist = re_wspaces.split(lbuffer)
                    for i in range(len(dlist)):
                        dlist[i] = float(dlist[i])
                        
                    rec_list.append(dlist)
                    
        col_names = col_names[:-1]
        col_types = col_types[:-1]
        self.data.data = rec.fromrecords(rec_list,formats=col_types,
                                    names=col_names)    
        

class Spectra(object):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.h5_file = None
        self.h5_group = None
        self.abs_factors = None

    def set_abs_factors(self,ff):
        """
        set_abs_factors(ff):
        Set the global absorber factors in the module.


        """
        if isinstance(ff,list):
            self.abs_factors = numpy.array(ff,dtype=numpy.double)
        elif isinstance(ff,numpy.ndarray):
            self.abs_factors = ff



    def recarray2hdf5(self,h5g,rec,name,desc,**keyargs):
        """
        recarray2hdf5(h5,g,rec,**keyargs):
        Save a record array in an HDF5 file. A pytables table 
        object is used to store the data.

        required input arguments:
        h5g ................. HDF5 group object or path
        rec ................ record array
        name ............... name of the table in the file
        desc ............... description of the table in the file

        optional keyword arguments:

        return value:
        tab ................. a HDF5 table object
        """

        #to build the table data types and names must be extracted
        descr = rec.dtype.descr
        tab_desc = {}
        cname_list = []

        for d in descr:
            tab_desc[d[0]] = tables.Col.from_dtype(numpy.dtype(d[1]))
            cname_list.append(d[0])

        #create the table object
        try:
            tab = self.h5_file.createTable(h5g,name,tab_desc,desc)
        except:
            print "Error creating table object %s!"
            return None

        #fill in data values
        for i in range(rec.shape[0]):
            for k in cname_list:
                tab.row[k] = rec[k][i]

            tab.row.append()

        tab.flush()
        self.h5_file.flush()

    def spectra2hdf5(self,dir,fname,mcatemp,**keyargs):
        """
        sepctra2hdf5(h5,name,desc,dir,scanname,mcatemp):
        Convert SPECTRA scan data to a HDF5 format. 
        
        required input arguments:
        dir ............... directory where the scan is stored
        fname ............. name of the SPECTRA data file
        mcatemp ........... template for the MCA file names

        optional keyword arguments:
        name .............. optional name under which to save the data
        desc .............. optional description of the scan
        """

        (name,ext) = os.path.splitext(fname)
        mcadir = os.path.join(dir,name)

        #evaluate keyword arguments
        if keyargs.has_key("name"):
            sg_name = keyargs["name"]
        else:
            sg_name = name

        if keyargs.has_key("desc"):
            sg_desc = keyargs["desc"]
        else:
            sg_desc = "SPECTRA data"

        #check wether an MCA directory exists or not
        if os.path.exists(mcadir):
            has_mca = True
        else:
            has_mca = False

        fullfname = os.path.join(dir,fname)
        if not os.path.exists(fullfname):
            print "data file does not exist!"
            return None

        #read data file
        (data,hdr) = read_data(fullfname)

        #create a new group to save the scan data in
        #this group is created below the default group determined by
        #self.h5_group
        try:
            sg = self.h5_file.createGroup(self.h5_group,sg_name,sg_desc)
        except:
            print "cannot create scan group!"
            return None

        self.recarray2hdf5(sg,data,"data","SPECTRA tabular data")

        #write attribute data
        for k in hdr.keys():
            self.h5_file.setNodeAttr(sg,"MOPOS_"+k,hdr[k])

        if has_mca:
            mca = read_mca_dir(mcadir,mcatemp)
            a = tables.Float64Atom()
            filter = tables.Filters(complib="zlib",complevel=4,fletcher32=True)
            c = self.h5_file.createCArray(sg,"MCA",a,mca.shape,"MCA data",filters=filter)
            c[...] = mca[...]

        self.h5_file.flush()

        return sg


    def abs_corr(self,data,f,**keyargs):
        """
        abs_corr(data,f,**keyargs):
        Perform absorber correction. Data can be either a 1 dimensional data (point
        detector) or a 2D MCA array. In the case of an array the data array should
        be of shape (N,NChannels) where N is the number of points in the scan an 
        NChannels the number of channels of the MCA. The absorber values are passed
        to the function as a 1D array of N elements. 

        By default the absorber values are taken form a global variable stored in
        the module called _absorver_factors. Despite this, costume values can be
        passed via optional keyword arguments.

        required input arguments:
        mca ............... matrix with the MCA data
        f ................. filter values along the scan

        optional keyword arguments:
        ff ................ custome filter factors

        return value:
        Array with the same shape as mca with the corrected MCA data.
        """

        mcan = numpy.zeros(data.shape,dtype=numpy.double)

        if keyargs.has_key("ff"):
            ff = keyargs["ff"]
        else:
            ff = _absorber_factors
        
        if len(data.shape)==2:
            #MCA and matrix data
            data = data*ff[f][:,numpy.newaxis]
        elif len(data.shape) == 1:
            data = data*ff[f]

        return data

def get_spectra_files(dirname):
    """
    get_spectra_files(dirname):
    Return a list of spectra files within a directory.

    required input arguments:
    dirname .............. name of the directory to search 

    return values:
    list with filenames
    """

    fnlist = os.listdir(dirname)
    onlist = []

    for fname in fnlist:
        (name,ext) = os.path.splitext(fname)
        if ext == ".fio":
            onlist.append(fname)

    onlist.sort()
    return onlist

def read_mca_dir(dirname,filetemp,**keyargs):
    """
    read_mca_dir(dirname):
    Read all MCA files within a directory 
    """

    flist = get_spectra_files(dirname)

    if keyargs.has_key("sort"):
        sort_flag = keyargs["sort"]
    else:
        sort_flag = True
    
    #create a list with the numbers of the files
    nlist = []
    for fname in flist:
        (name,ext) = os.path.splitext(fname)
        name = name.replace(filetemp,"")
        nlist.append(int(name))

    if sort_flag:
        nlist.sort()

    dlist = []

    for i in nlist:
        fname = os.path.join(dirname,filetemp+"%i.fio")
        fname = fname %(i)
        d = read_mca(fname)
        dlist.append(d.tolist())
    
    return numpy.array(dlist)


def read_mca(fname):
    """
    read_mca(fname):
    Read a single SPECTRA MCA file. 

    required input arguments:
    fname ............... name of the file to read

    return value:
    data ................ a numpy array witht the MCA data
    """

    try:
        fid = open(fname)
    except:
        print "cannot open file %s!" %fname
        return None

    dlist = []
    hdr_flag = True

    while True:
        lbuffer = fid.readline()
        
        if lbuffer == "": break
        lbuffer = lbuffer.strip()
        if lbuffer == "%d": 
            hdr_flag = False
            lbuffer = fid.readline()
            continue

        if not hdr_flag:
            dlist.append(float(lbuffer))

    return numpy.array(dlist,dtype=numpy.double)

def read_mcas(ftemp,cntstart,cntstop):
    """
    read_mcas(ftemp,cntstart,cntstop):
    Read MCA data from a SPECTRA MCA directory. The filename is passed as a
    generic 
    """

    fnums = range(cntstart,cntstop+1)
    mcalist = []

    for i in fnums:
        fname = ftemp %i
        print "processing file %s ..." %fname
        mcalist.append(read_mca(fname))

    return numpy.array(mcalist,dtype=numpy.double)

    
def read_data(fname):
    """
    read_data(fname):
    Read a spectra data file (a file with now MCA data). 

    required input arguments:
    fname .................... name of the file to read

    return values:
    (data,hdr)
    data .......... numpy record array where the keys are the column names
    hdr ........... a dictionary with header information
    """

    try:
        fid = open(fname,"r")
    except:
        print "cannot open file %s!" %fname
        return None

    hdr_dict = {}
    hdr_flag = False
    data_flag = False
    col_cnt = 0        #column counter
    col_names = []     #list with column names
    data = []

    fname = os.path.basename(fname)
    fname,ext = os.path.splitext(fname)
    print fname

    while True:
        lbuffer = fid.readline()
        if lbuffer == "": break

        lbuffer = lbuffer.strip()
        #check for common break conditions
        #if the line is a comment skip it
        if lbuffer[0] == "!": continue

        #remove leading and trailing whitespace symbols
        lbuffer = lbuffer.strip()

        if lbuffer == "%p":
            hdr_flag = True
            continue

        if lbuffer == "%d":
            hdr_flag = False
            data_flag = True
            continue

        if hdr_flag:
            #read header data (initial motor positions)
            key,value = lbuffer.split("=")
            key = key.strip()
            value = value.strip()
            hdr_dict[key] = float(value)

        if data_flag:
            #have to read the column names first
            if re_colname.match(lbuffer):
                l = re_wspaces.split(lbuffer)
                col_names.append(l[2].replace(fname.upper()+"_",""))
            else:
                #read data values
                dlist = re_wspaces.split(lbuffer)
                #convert strings to float values
                for i in range(len(dlist)): dlist[i] = float(dlist[i])

                data.append(dlist)
            
    #create a record array to hold data
    data = numpy.rec.fromrecords(data,names=col_names)

    return (data,hdr_dict)

