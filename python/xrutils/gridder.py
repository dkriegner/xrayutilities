
import numpy
import libxrayutils
import ctypes

unit_dict = {"kb":1024,"mb":1024**2,"gb":1024**3}

class Gridder(object):
    def __init__(self):
        self.nthreads = 0
        self.csize = 0
        self.cunit = 1024**2
        self.flags = 0 

    def SetThreads(self,n):
        self.nthreads = n

    def SetChunkSize(self,n):
        self.csize = n

    def SetChunkUnit(self,u):
        if u in unit_dict.keys():
            self.cunit = unit_dict[u]    
        else:
            print "Chunk size unit must be one of"
            print "kb, mb or gb"
            return None

    def Normalize(self,bool):
        self.flags = self.flags^4

    def KeepData(self,bool):
        if not bool==True or bool == False:
            raise TypeError,"Keep Data flag must be a boolan value (True/False)!"

        self.keep_data = bool


class Gridder2D(Gridder):
    def __init__(self,nx,ny):
        Gridder.__init__(self)

        self.nx = nx
        self.ny = ny
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0

        self.gdata = numpy.zeros((nx,ny),dtype=numpy.double)
        self.gnorm = numpy.zeros((nx,ny),dtype=numpy.double)

    def SetResolution(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def ClearData(self):
        self.gdata[...] = 0
        self.gnorm[...] = 0
    
    def GetXAxis(self):
        """
        GetXAxis():
        Return the x-axis if the gridded data as a numpy array of shape
        (nx).
        """
        dx = (self.xmax-self.xmin)/(self.nx-1)
        ax = self.xmin+dx*numpy.arange(0,self.nx)
        return ax

    def GetYAxis(self):
        """
        GetYAxis():
        Return the y-axis if the gridded data as a numpy array of shape (ny).
        """
        dy = (self.ymax-self.ymin)/(self.ny-1)
        ax = self.ymin + dy*numpy.arange(0,self.ny)
        return ax

    def GetXMatrix(self):
        """
        GetXMatrix():
        Return x axis in form of a matrix of shape (nx,ny). The axis value 
        vary along the first index (x,axis).
        """
        m = numpy.ones((self.nx,self.ny),dtype=numpy.double)
        a = self.GetXAxis()

        return m*a[:,numpy.newaxis]

    def GetYMatrix(self):
        """
        GetYMatrix():
        Return y axis in form of a matrx of shape (nx,ny) where the 
        axis values vary along the second index ny.
        """
        a = self.GetYAxis()
        m = numpy.ones((self.nx,self.ny),dtype=numpy.double)

        return m*a[numpy.newaxis,:]

    def GridData(self,x,y,data):
        """
        GridData(x,y,data):
        Perform gridding on a set of data. 

        required input argument:
        x ............... numpy array of arbitrary shape with x positions
        y ............... numpy array of arbitrary shape with y positions
        data ............ numpy array of arbitrary shape with data values
        """

        x = x.reshape(x.size)
        y = y.reshape(y.size)
        data = data.reshape(data.size)

        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = y.min()
        self.ymax = y.max()

        if self.nthreads != 0:
            #use threaded code
            print "using threaded code ..."
            libxrayutils._gridder2d_th(ctypes.c_uint(self.nthreads),x,y,data,ctypes.c_uint(x.size),
                                      ctypes.c_uint(self.nx),ctypes.c_uint(self.ny),
                                      ctypes.c_double(self.xmin),ctypes.c_double(self.xmax),
                                      ctypes.c_double(self.ymin),ctypes.c_double(self.ymax),
                                      self.gdata,self.gnorm,self.flags)
        else:
            #use sequential code - good for small data
            print "using sequential code ..."
            print self.flags
            libxrayutils._gridder2d(x,y,data,ctypes.c_uint(x.size),
                                   ctypes.c_uint(self.nx),ctypes.c_uint(self.ny),
                                   ctypes.c_double(self.xmin),ctypes.c_double(self.xmax),
                                   ctypes.c_double(self.ymin),ctypes.c_double(self.ymax),
                                   self.gdata,self.gnorm,ctypes.c_int(self.flags))

    def GridDataChunked(self,xobj,yobj,zobj):
        pass

def Gridder3D(Gridder2D):
    def __init__(self,nx,ny,nz):
        Gridder2D.__init__(self,nx,ny)

        self.nz = nz
        self.gdata = numpy.zeros((nx,ny,nz),dtype=numpy.double)
        self.gnorm = numpy.zeros((nx,ny,nz),dtype=numpy.double)

    def GetZMatrix(self):
        pass

    def GetZAxis(self):
        pass

  

def grid2dmap(x,y,data,nx,ny,**keyargs):
    """
    grid2dmap(x,y,z,data,nx,ny,nz):
    grid2dmap grids data stored on a nonregular grid of any dimension onto a
    2D grid. Using the optional keyword arguments the routine can be used
    on the same data grid in subsquent runs. This makes sense in the case, that
    the original dataset is to large to be stored in the main memory as a whole,
    and therefore must be splitted into several chunks which should be all
    gridded into one matrix. On deman axes values or axis matrices will be 
    returned.
    
    required input arguments:
    x ............... matrix with x-values to be gridded
    y ............... matrix with y-values to be gridded
    data ............ matrix with data values to be gridded
    nx .............. number of grid points in x direction
    ny .............. number of grid points in y direction

    optional keyword arguments:
    normalize ....... True/False perform normalization of the gridded data
    dgrid ........... data grid from a previous run
    ngrid ........... normalization grid from a previous run
    axmat ........... axis values in a matrix of same shape as the result are returned
    axval ........... axis values are returned as vectors with the length of the matrix dimension
                      the axis belongs to are returned
    threads ......... number of threads to use in case of the threaded version
    
    
    return values: [datagrid,datanorm] or datagrid
    datagrid ........ (nx,ny,nz) numpy array with the gridded data
    datanorm ........ (nx,ny,nz) numpy array with the normalizatio matrix
                       datanorm is only returnd if no normalization is performed (normalize=False).
    """
    
        #evaluate keyword arguments:
    if keyargs.has_key("normalize"):
        perform_normalization_flag = keyargs["normalize"];
    else:
        perform_normalization_flag = True;
        
    if keyargs.has_key("axval"):
        axval = keyargs["axval"]
    else:
        axval = False
        
    if keyargs.has_key("axmat"):
        axmat = keyargs["axmat"]
    else:
        axmat = False

    dx = abs((xmax-xmin)/(nx-1))
    dy = abs((ymax-ymin)/(ny-1))

    if keyargs.has_key("dgrid"):
        datagrid = keyargs["dgrid"]
    else:
        datagrid = numpy.zeros((ny,nx),numpy.float)

    if keyargs.has_key("ngrid"):
        datanorm = keyargs["ngrid"]
    else:
        datanorm = numpy.zeros((ny,nx),numpy.int)
                
            
    outlist = []

    #perform normalization if requested (Default)
    if perform_normalization_flag:
        print "perform normalization ..."
        numpy.putmask(datanorm,datanorm==0.0,1.0);
        datagrid = datagrid/datanorm.astype(numpy.float);
        print datagrid.shape
        outlist.append(datagrid);
    else:
        outlist.append(datagrid);
        outlist.append(datanorm);
        
    #build axis values or matrices if requested by the user
    if axval:
        xaxis = numpy.arange(xmin,xmax+0.1*dx,dx,dtype=numpy.double);
        yaxis = numpy.arange(ymin,ymax+0.1*dx,dy,dtype=numpy.double);
        outlist.append(xaxis);
        outlist.append(yaxis);
        
    if axmat:
        xaxis = numpy.arange(xmin,xmax+0.1*dx,dx,dtype=numpy.double)
        yaxis = numpy.arange(ymin,ymax+0.1*dy,dy,dtype=numpy.double)
        m = numpy.ones(datagrid.shape,numpy.double)
        print xmin,xmax,xaxis.shape
        print ymin,ymax,yaxis.shape
        print m.shape
        xmat = m*xaxis[numpy.newaxis,:]
        ymat = m*yaxis[:,numpy.newaxis]
        outlist.append(xmat)
        outlist.append(ymat)
        
    return outlist;
    

def grid3dmap(x,y,z,data,nx,ny,nz,**keyargs):
    """
    grid3dmap(x,y,z,data,nx,ny,nz):
    grid3d grids data stored on a nonregular grid of any dimension onto a
    3D grid. Using the optional keyword arguments the routine can be used
    on the same data grid in subsquent runs. This makes sense in the case, that
    the original dataset is to large to be stored in the main memory as a whole,
    and therefore must be splitted into several chunks which should be all
    gridded into one matrix.
    
    required input arguments:
    x ............... matrix with x-values to be gridded
    y ............... matrix with y-values to be gridded
    z ............... matrix with z-values to be gridded
    data ............ matrix with data values to be gridded
    nx .............. number of grid points in x direction
    ny .............. number of grid points in y direction
    nz .............. number of grid points in z direction

    optional keyword arguments:
    normalize ....... True/False perform normalization of the grided data
    xrange .......... [xmin,xmax] range in x-direction
    yrange .......... [ymin,ymax] range in y-direction
    zrange .......... [zmin,zmax] range in z-direction
    dgrid ........... data grid from a previous run
    ngrid ........... normalization grid from a previous run
    
    return values: [datagrid,datanorm] or datagrid
    datagrid ........ (nx,ny,nz) numpy array with the gridded data
    datanorm ........ (nx,ny,nz) numpy array with the normalizatio matrix
                       datanorm is only returnd if no normalization is performed (normalize=False).
    """
    
        #evaluate keyword arguments:
    if keyargs.has_key("normalize"):
        perform_normalization_flag = keyargs["normalize"];
    else:
        perform_normalization_flag = True;
        
    if keyargs.has_key("axval"):
        axval = keyargs["axval"];
    else:
        axval = False;
        
    if keyargs.has_key("axmat"):
        axmat = keyargs["axmat"];
    else:
        axmat = False;
        
    if keyargs.has_key("xrange"):
        xmin = keyargs["xrange"][0];
        xmax = keyargs["xrange"][1];
    else:
        xmin = x.min();
        xmax = x.max();

    if keyargs.has_key("yrange"):
        ymin = keyargs["yrange"][0];
        ymax = keyargs["yrange"][1];
    else:
        ymin = y.min();
        ymax = y.max();

    if keyargs.has_key("zrange"):
        zmin = keyargs["zrange"][0];
        zmax = keyargs["zrange"][1];
    else:
        zmin = z.min();
        zmax = z.max();

    dx = abs((xmax-xmin)/(nx-1));
    dy = abs((ymax-ymin)/(ny-1));
    dz = abs((zmax-zmin)/(nz-1));

    if keyargs.has_key("dgrid"):
        datagrid = keyargs["dgrid"];
    else:
        datagrid = numpy.zeros((ny,nx,nz),numpy.float);

    if keyargs.has_key("ngrid"):
        datanorm = keyargs["ngrid"];
    else:
        datanorm = numpy.zeros((ny,nx,nz),numpy.int);
                
    xindex = numpy.floor((x-xmin)/dx).astype(numpy.int);
    yindex = numpy.floor((y-ymin)/dy).astype(numpy.int);
    zindex = numpy.floor((z-zmin)/dz).astype(numpy.int);
    
    for i in range(data.size):            

        try:
            #I use here exception tracking since it is possible from the above calculation that some
            #points are maybe outside the range of the grid. In this case they are simply ignored.
            #this is obvioulsy the case if custom ranges are provided by the user.
            datagrid[yindex.flat[i],\
                     xindex.flat[i],\
                     zindex.flat[i]] = datagrid[yindex.flat[i],xindex.flat[i],zindex.flat[i]]+data.flat[i];
            datanorm[yindex.flat[i],\
                     xindex.flat[i],\
                     zindex.flat[i]] = datanorm[yindex.flat[i],xindex.flat[i],zindex.flat[i]] + 1;
        except:
            pass

    outlist = [];
    #perform normalization if requested (Default)
    if perform_normalization_flag:
        print "perform normalization ..."
        numpy.putmask(datanorm,datanorm==0.0,1.0);
        datagrid = datagrid/datanorm.astype(numpy.float);
        outlist.append(datagrid);
    else:
        outlist.append(datagrid);
        outlist.append(datanorm);

        
    #build axis values or matrices if requested by the user
    if axval:
        xaxis = numpy.arange(xmin,xmax+0.1*dx,dx,dtype=numpy.double);
        yaxis = numpy.arange(ymin,ymax+0.1*dx,dy,dtype=numpy.double);
        zaxis = numpy.arange(zmin,zmax+0.1*dx,dz,dtype=numpy.double);
        outlist.append(xaxis);
        outlist.append(yaxis);
        outlist.append(zaxis);
        
    if axmat:
        xaxis = numpy.arange(xmin,xmax+0.1*dx,dx,dtype=numpy.double);
        yaxis = numpy.arange(ymin,ymax+0.1*dy,dy,dtype=numpy.double);
        zaxis = numpy.arange(zmin,zmax+0.1*dx,dz,dtype=numpy.double);
        m = numpy.ones(datagrid.shape,numpy.double);
        print xaxis.shape
        print yaxis.shape
        print zaxis.shape
        print m.shape
        xmat = m*xaxis[:,numpy.newaxis,numpy.newaxis];
        ymat = m*yaxis[numpy.newaxis,:,numpy.newaxis];
        zmat = m*zaxis[numpy.newaxis,numpy.newaxis,:];
        outlist.append(xmat);
        outlist.append(ymat);
        outlist.append(zmat);
        
    return outlist;

