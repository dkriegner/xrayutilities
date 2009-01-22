#module handling crystall lattice structures

import numpy

class LatticeBase(list):
    """
    The LatticeBase class implements a container for a set of 
    points that form the base of a crystal lattice. An instance of this class
    can be treated as a simple container objects. 
    """
    def __init__(self,*args,**keyargs):
       list.__init__(self,*args,**keyargs) 

    def append(self,point):
        if isinstance(point,list):
            p = numpy.array(point,dtype=numpy.double)
        elif isinstance(point,numpy.ndarray):
            p = point
        else:
            raise TypeError,"point must be a list or numpy array of shape (3)"

        list.append(self,p)


    def __setitem__(self,key,point):
        if isinstance(point,list):
            p = numpy.array(point,dtype=numpy.double)
        elif isinstance(point,numpy.ndarray):
            p = point
        else:
            raise TypeError,"point must be a list or numpy array of shape (3)"

        list.__setitem__(self,key,p)

    def __str__(self):
        ostr = ""
        for i in range(list.__len__(self)):
            p = list.__getitem__(self,i)
            ostr += "Base point %i: (%f %f %f)\n" %(i,p[0],p[1],p[2])

        return ostr

class Lattice(object):
    def __init__(self,a1,a2,a3):
        if isinstance(a1,list):
            self.a1 = numpy.array(a1,dtype=numpy.double)
        elif isinstance(a1,numpy.ndarray):
            self.a1 = a1
        else:
            raise TypeError,"a1 must be a list or a numpy array"

        if isinstance(a2,list):
            self.a2 = numpy.array(a2,dtype=numpy.double)
        elif isinstance(a1,numpy.ndarray):
            self.a2 = a2
        else:
            raise TypeError,"a2 must be a list or a numpy array"
        
        if isinstance(a3,list):
            self.a3 = numpy.array(a3,dtype=numpy.double)
        elif isinstance(a3,numpy.ndarray):
            self.a3 = a3
        else:
            raise TypeError,"a3 must be a list or a numpy array"

    def ApplyStrain(self,eps):
        """
        ApplyStrain(eps):
        Applies a certain strain on a lattice. The result is a change 
        in the base vectors. 

        requiered input arguments:
        eps .............. a vector with the 6 independent strain components
        """

        if isinstance(eps,list):
            eps = numpy.array(eps,dtype=numpy.double)
        
        u1 = self.a1*esp[:3]
        self.a1 = self.a1 + u1
        u2 = self.a2*eps[:3]
        self.a2 = self.a2 + u2
        u3 = self.a3*eps[:3]
        self.a3 = self.a3 + u3


    def ReciprocalLattice(self):
        V = (self.a3*numpy.cross(self.a1,self.a2)).sum()
        p = 2.*numpy.pi/V
        b1 = p*numpy.cross(self.a2,self.a3)
        b2 = p*numpy.cross(self.a3,self.a1)
        b3 = p*numpy.cross(self.a1,self.a2)

        return Lattice(b1,b2,b3)

    def GetPoint(self,*args):
        if len(args)<3:
            raise IndexError,"need 3 indices for the lattice point"

        return args[0]*self.a1+args[1]*self.a2+args[2]*self.a3

    def __str__(self):
        ostr = ""
        ostr += "a1 = (%f %f %f)\n" %(self.a1[0],self.a1[1],self.a1[2])
        ostr += "a2 = (%f %f %f)\n" %(self.a2[0],self.a2[1],self.a2[2])
        ostr += "a3 = (%f %f %f)\n" %(self.a3[0],self.a3[1],self.a3[2])

        return ostr

class Crystal(object):
    def __init__(self,base,lat):
        self.Lattice = lat
        self.Base = base

class Transform(object):
    def __init__(self,matrix):
        self.matrix = matrix
        try:
            self.imatrix = numpy.linalg.inv(matrix)
        except:
            print "matrix cannot be inverted - seems to be singular"
            self.imatrix = None

    def __call__(self,*args):
        olist = []
        for a in args:
            if isinstance(a,list):
                p = numpy.array(a,dtype=numpy.double)
            elif isinstance(a,numpy.ndarray):
                p = a
            else:
                raise TypeError,"Argument must be a list or numpy array!"

            #matrix product in pure array notation
            if len(p.shape)==1:
                #argument is a vector
                print "transform a vector ..."
                b = (self.matrix*p[numpy.newaxis,:]).sum(axis=1)
                olist.append(b)
            elif len(p.shape)==2 and p.shape[0]==3 and p.shape[1]==3:
                #argument is a matrix
                print "transform a matrix ..."
                b = numpy.zeros(p.shape,dtype=numpy.double)
                b2 = numpy.zeros(p.shape,dtype=numpy.double)
                for i in range(3):
                    for j in range(3):
                        b[i,j] = (self.matrix[i,:]*p[:,j]).sum()

                #perform multiplication with the inverse matrix
                for i in range(3):
                    for j in range(3):
                        b2[i,j] = (b[i,:]*self.imatrix[:,j]).sum()

                olist.append(b2)

    
        if len(args) == 1:
            return olist[0]
        else:
            return olist
    def __str__(self):
        ostr = ""
        ostr += "Transformation matrix:\n"
        ostr += "%f %f %f\n" %(self.matrix[0,0],self.matrix[0,1],self.matrix[0,2])
        ostr += "%f %f %f\n" %(self.matrix[1,0],self.matrix[1,1],self.matrix[1,2])
        ostr += "%f %f %f\n" %(self.matrix[2,0],self.matrix[2,1],self.matrix[2,2])

        return ostr

def CoordinateTransform(v1,v2,v3):
    """
    CoordinateTransform(v1,v2,v3):
    Create a Transformation object which transforms a point into a new 
    coordinate frame. The new frame is determined by the three vectors
    v1, v2 and v3.

    required input arguments:
    v1 ............. list or numpy array with new base vector 1
    v2 ............. list or numpy array with new base vector 2 
    v2 ............. list or numpy array with new base vector 3

    return value:
    An instance of a Transform class
    """

    if isinstance(v1,list):
        e1 = numpy.array(v1,dtype=numpy.double)
    elif isinstance(v1,numpy.ndarray):
        e1 = v1
    else:
        raise TypeError,"vector must be a list or numpy array"
    
    if isinstance(v2,list):
        e2 = numpy.array(v2,dtype=numpy.double)
    elif isinstance(v2,numpy.ndarray):
        e2 = v2
    else:
        raise TypeError,"vector must be a list or numpy array"
    
    if isinstance(v3,list):
        e3 = numpy.array(v3,dtype=numpy.double)
    elif isinstance(v3,numpy.ndarray):
        e3 = v3
    else:
        raise TypeError,"vector must be a list or numpy array"

    #normalize base vectors
    e1 = e1/numpy.sqrt((e1**2).sum())
    e2 = e2/numpy.sqrt((e2**2).sum())
    e3 = e3/numpy.sqrt((e3**2).sum())

    #assemble the transformation matrix
    m = numpy.array([e1,e2,e3])
    
    return Transform(m)
    

#some idiom functions to simplify lattice creation

def CubicLattice(a):
    """
    CubicLattice(a):
    Returns a Lattice object representing a simple cubic lattice.
    
    required input arguments:
    a ................ lattice parameter

    return value:
    an instance of  Lattice class
    """

    return Lattice([a,0,0],[0,a,0],[0,0,a])

#some lattice related functions




