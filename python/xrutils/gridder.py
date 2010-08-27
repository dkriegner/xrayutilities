
import numpy
import ctypes

from . import libxrayutils

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
        self.gdata = numpy.zeros((nx,ny),dtype=numpy.double)
        self.gnorm = numpy.zeros((nx,ny),dtype=numpy.double)

    def __get_xaxis(self):
        dx = (self.xmax-self.xmin)/(self.nx-1)
        ax = self.xmin+dx*numpy.arange(0,self.nx)
        return ax

    def __get_yaxis(self):
        dy = (self.ymax-self.ymin)/(self.ny-1)
        ax = self.ymin + dy*numpy.arange(0,self.ny)
        return ax

    def __get_xmatrix(self):
        m = numpy.ones((self.nx,self.ny),dtype=numpy.double)
        a = self.xaxis

        return m*a[:,numpy.newaxis]

    def __get_ymatrix(self):
        a = self.yaxis
        m = numpy.ones((self.nx,self.ny),dtype=numpy.double)

        return m*a[numpy.newaxis,:]

    def __get_data(self):
        return self.gdata.copy()

    yaxis = property(__get_yaxis)
    xaxis = property(__get_xaxis)
    xmatrix = property(__get_xmatrix)
    ymatrix = property(__get_ymatrix)
    data = property(__get_data)


    def Clear(self):
        self.gdata[...] = 0
        self.gnorm[...] = 0


    def __call__(self,*args):
        """
        GridData(x,y,data):
        Perform gridding on a set of data. 

        required input argument:
        x ............... numpy array of arbitrary shape with x positions
        y ............... numpy array of arbitrary shape with y positions
        data ............ numpy array of arbitrary shape with data values
        """

        x = args[0]
        y = args[1]
        data = args[2]
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        data = data.reshape(data.size)

        # require correct aligned memory for input arrays
        x = numpy.require(x,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        y = numpy.require(y,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        data = numpy.require(data,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])

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

class Gridder3D(Gridder2D):
    def __init__(self,nx,ny,nz):
        Gridder2D.__init__(self,nx,ny)

        self.nz = nz
        self.gdata = numpy.zeros((nx,ny,nz),dtype=numpy.double)
        self.gnorm = numpy.zeros((nx,ny,nz),dtype=numpy.double)
        
        self.zmin = 0
        self.zmax = 0
        
    def SetResolution(self,nx,ny,nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.gdata = numpy.zeros((nx,ny,nz),dtype=numpy.double)
        self.gnorm = numpy.zeros((nx,ny,nz),dtype=numpy.double)
    
    
    def __get_zaxis(self):        
        dz = (self.zmax-self.zmin)/(self.nz-1)
        az = self.zmin + dz*numpy.arange(0,self.nz)
        return az

    def __get_zmatrix(self):        
        a = self.GetZAxis()
        m = numpy.ones((self.nx,self.ny,self.nz),dtype=numpy.double)

        return m*a[numpy.newaxis,numpy.newaxis,:]
        
    zaxis = property(__get_zaxis)
    zmatrix = property(__get_zmatrix)
        

    def __call__(self,x,y,z,data):
        """
        GridData(x,y,data):
        Perform gridding on a set of data. After running the gridder 
        the gdata object in the class is holding the gridded data.

        required input argument:
        x ............... numpy array of arbitrary shape with x positions
        y ............... numpy array of arbitrary shape with y positions
        z ............... numpy array fo arbitrary shape with z positions
        data ............ numpy array of arbitrary shape with data values
        """

        x = x.reshape(x.size)
        y = y.reshape(y.size)
        z = z.reshape(z.size)
        data = data.reshape(data.size)
        
        # require correct aligned memory for input arrays
        x = numpy.require(x,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        y = numpy.require(y,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        z = numpy.require(z,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        data = numpy.require(data,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        
        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = y.min()
        self.ymax = y.max()
        self.zmin = z.min()
        self.zmax = z.max()

        if self.nthreads != 0:
            #use threaded code
            print "using threaded code ..."
            print self.flags
            libxrayutils._gridder3d_th(ctypes.c_uint(self.nthreads),x,y,z,data,ctypes.c_uint(x.size),
                                      ctypes.c_uint(self.nx),ctypes.c_uint(self.ny),ctypes.c_uint(self.nz),
                                      ctypes.c_double(self.xmin),ctypes.c_double(self.xmax),
                                      ctypes.c_double(self.ymin),ctypes.c_double(self.ymax),
                                      ctypes.c_double(self.zmin),ctypes.c_double(self.zmax),
                                      self.gdata,self.gnorm,self.flags)
        else:
            #use sequential code - good for small data
            print "using sequential code ..."
            print self.flags
            libxrayutils._gridder3d(x,y,z,data,ctypes.c_uint(x.size),
                                   ctypes.c_uint(self.nx),ctypes.c_uint(self.ny),ctypes.c_uint(self.nz),
                                   ctypes.c_double(self.xmin),ctypes.c_double(self.xmax),
                                   ctypes.c_double(self.ymin),ctypes.c_double(self.ymax),
                                   ctypes.c_double(self.zmin),ctypes.c_double(self.zmax),
                                   self.gdata,self.gnorm,self.flags)

  

