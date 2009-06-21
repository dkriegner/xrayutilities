#module handling crystall lattice structures

import numpy

class Atom(object):
    def __init__(self,name,pos):
        self.name = name
        if isinstance(pos,list):
            self.pos = numpy.array(list,dtype=numpy.double)
        elif isinstance(pos,numpy.ndarray):
            self.pos = pos
        else:
            raise TypeError,"Atom position must be array or list!"



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
    """
    class Lattice:
    This object represents a Bravais lattice. A lattice consists of a 
    base 
    """
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
        eps .............. a 3x3 matrix independent strain components
        """

        if isinstance(eps,list):
            eps = numpy.array(eps,dtype=numpy.double)
        
        u1 = (eps*self.a1[numpy.newaxis,:]).sum(axis=1)
        self.a1 = self.a1 + u1
        u2 = (eps*self.a2[numpy.newaxis,:]).sum(axis=1)
        self.a2 = self.a2 + u2
        u3 = (eps*self.a3[numpy.newaxis,:]).sum(axis=1)
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

def DiamondLattice(a):
	pass
	
def FCCLattice(a):
	pass
	
def BCCLattice(a):
	pass




