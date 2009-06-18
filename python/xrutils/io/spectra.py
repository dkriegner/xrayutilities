#module to handle spectra data

import numpy
import os
import re
import tables
import os.path

re_wspaces = re.compile(r"\s+")
re_colname = re.compile(r"^Col")

_absorber_factors = None

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



    def recarray2hdf5(self,rec,name,desc,**keyargs):
        """
        recarray2hdf5(h5,g,rec,**keyargs):
        Save a record array in an HDF5 file. A pytables table 
        object is used to store the data.

        required input arguments:
        h5 ................. HDF5 file object
        g .................. group object or path
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
            tab = self.h5_file.createTable(self.h5_group,name,tab_desc,desc)
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

        self.recarray2hdf5(data,"data","SPECTRA tabular data")

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

