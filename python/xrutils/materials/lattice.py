#module handling crystall lattice structures
#

import numpy
from numpy.linalg import norm
import atexit

from . import __path__
from . import database

_db = database.DataBase(__path__[0]+"/data/test.db")
_db.Open()

def _db_cleanup():
    _db.Close()

atexit.register(_db_cleanup)

class Atom(object):
    #{{{1
    def __init__(self,name,num):
        self.name = name
        self.num = num
        
    def f0(self,q):
        _db.SetMaterial(self.name)
        
        if isinstance(q,numpy.ndarray) or isinstance(q,list):
            d = numpy.zeros((len(q)),dtype=numpy.double)
            for i in range(len(q)):
                d[i] = _db.GetF0(q[i])
                
            return d
        else:
            return _db.GetF0(q)
                
    def f1(self,en):
        _db.SetMaterial(self.name)
        
        if isinstance(en,numpy.ndarray) or isinstance(en,list):
            d = numpy.zeros((len(en)),dtype=numpy.double)
            for i in range(len(en)):
                d[i] = _db.GetF1(en[i])
                
            return d
        else:
            return _db.GetF1(en)
        
    def f2(self,en):
        _db.SetMaterial(self.name)
        
        if isinstance(en,numpy.ndarray) or isinstance(en,list):
            d = numpy.zeros((len(en)),dtype=numpy.double)
            for i in range(len(en)):
                d[i] = _db.GetF2(en[i])
                
            return d
        else:
            return _db.GetF2(en)
    
    def f(self,q,en):
        #{{{2
        """
        function to calculate the atomic structure factor F

        Parameter
        ---------
         q:     momentum transfer 
         en:    energy for which F should be calculated

        Returns
        -------
         f (float)
        """
        f = self.f0(norm(q))+self.f1(en)+1.j*self.f2(en)
        return f
        #}}}2

    def __str__(self):
        ostr = self.name
        ostr += " (%2d)" %self.num
        return ostr 
    #}}}1    



class LatticeBase(list):
    #{{{1
    """
    The LatticeBase class implements a container for a set of 
    points that form the base of a crystal lattice. An instance of this class
    can be treated as a simple container object. 
    """
    def __init__(self,*args,**keyargs):
       list.__init__(self,*args,**keyargs) 

    def append(self,atom,pos):
        if not isinstance(atom,Atom):           
            raise TypeError,"atom must be an instance of class Atom"
            
        if isinstance(pos,list):
            pos = numpy.array(pos,dtype=numpy.double)
        elif isinstance(pos,numpy.ndarray):
            pos = pos
        else:
            raise TypeError,"Atom position must be array or list!"

        list.append(self,(atom,pos))


    def __setitem__(self,key,data):
        (atom,pos) = data
        if not isinstance(atom,Atom):
            raise TypeError,"atom must be an instance of class Atom!"
            
        if isinstance(pos,list):
            p = numpy.array(pos,dtype=numpy.double)
        elif isinstance(pos,numpy.ndarray):
            p = pos
        else:
            raise TypeError,"point must be a list or numpy array of shape (3)"

        list.__setitem__(self,key,(atom,p))

    def __str__(self):
        ostr = ""
        for i in range(list.__len__(self)):
            (atom,p) = list.__getitem__(self,i)
            
            ostr += "Base point %i: %s (%f %f %f)\n" %(i,atom.__str__(),p[0],p[1],p[2])

        return ostr
    #}}}1
    

class Lattice(object):
    #{{{1
    """
    class Lattice:
    This object represents a Bravais lattice. A lattice consists of a 
    base 
    """
    def __init__(self,a1,a2,a3,base=None):
        #{{{2
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
            
        if base!=None:
            if not isinstance(base,LatticeBase):
                raise TypeError,"lattice base must be an instance of class LatticeBase"
            else:
                self.base = base
        else:
            self.base = None
        #}}}2

    def ApplyStrain(self,eps):
        #{{{2
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
        #}}}2

    def ReciprocalLattice(self):
        V = self.UnitCellVolume()
        p = 2.*numpy.pi/V
        b1 = p*numpy.cross(self.a2,self.a3)
        b2 = p*numpy.cross(self.a3,self.a1)
        b3 = p*numpy.cross(self.a1,self.a2)

        return Lattice(b1,b2,b3)

    def UnitCellVolume(self):
        #{{{2
        """
        function to calculate the unit cell volume of a lattice (angstrom^3)
        """
        V = numpy.dot(self.a3,numpy.cross(self.a1,self.a2))
        return V
        #}}}2

    def GetPoint(self,*args):
        if len(args)<3:
            args = args[0]
            if len(args)<3:
                raise IndexError,"need 3 indices for the lattice point"

        return args[0]*self.a1+args[1]*self.a2+args[2]*self.a3

    def __str__(self):
        ostr = ""
        ostr += "a1 = (%f %f %f)\n" %(self.a1[0],self.a1[1],self.a1[2])
        ostr += "a2 = (%f %f %f)\n" %(self.a2[0],self.a2[1],self.a2[2])
        ostr += "a3 = (%f %f %f)\n" %(self.a3[0],self.a3[1],self.a3[2])
        
        if self.base:
            ostr += "Lattice base:\n"
            ostr += self.base.__str__()

        return ostr
            
    #}}}1
    
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

def ZincBlendeLattice(aa,ab,a):
    #{{{1 
    #create lattice base
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0])
    lb.append(aa,[0.5,0,0.5])
    lb.append(aa,[0,0.5,0.5])
    lb.append(ab,[0.25,0.25,0.25])
    lb.append(ab,[0.75,0.75,0.25])
    lb.append(ab,[0.75,0.25,0.75])
    lb.append(ab,[0.25,0.75,0.75])
    
    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,a]
    
    l = Lattice(a1,a2,a3,base=lb)    
    
    return l
    #}}}1

def DiamondLattice(aa,a):
    #{{{1
    # Diamond is ZincBlende with two times the same atom
    return ZincBlendeLattice(aa,aa,a)
    #}}}1
    
def FCCLattice(aa,a):
    #{{{1
    #create lattice base
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0])
    lb.append(aa,[0.5,0,0.5])
    lb.append(aa,[0,0.5,0.5])
    
    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,a]
    
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1
    
def BCCLattice(aa,a):
    #{{{1
    #create lattice base
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0.5])

    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,a]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def RockSaltLattice(aa,ab,a):
    #{{{1
    #create lattice base; data from http://cst-www.nrl.navy.mil/lattice/index.html
    print("Warning: NaCl lattice is not using a cubic lattice structure") 
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(ab,[0.5,0.5,0.5])

    #create lattice vectors
    a1 = [0,0.5*a,0.5*a]
    a2 = [0.5*a,0,0.5*a]
    a3 = [0.5*a,0.5*a,0]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def RockSalt_Cubic_Lattice(aa,ab,a):
    #{{{1
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0])
    lb.append(aa,[0,0.5,0.5])
    lb.append(aa,[0.5,0,0.5])

    lb.append(ab,[0.5,0,0])
    lb.append(ab,[0,0.5,0])
    lb.append(ab,[0,0,0.5])
    lb.append(ab,[0.5,0.5,0.5])

    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,a]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def RutileLattice(aa,ab,a,c,u):
    #{{{1
    #create lattice base; data from http://cst-www.nrl.navy.mil/lattice/index.html
    # P4_2/mmm(136) aa=2a,ab=4f; x \approx 0.305 (VO_2)
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0.5])
    lb.append(ab,[u,u,0.])
    lb.append(ab,[-u,-u,0.])
    lb.append(ab,[0.5+u,0.5-u,0.5])
    lb.append(ab,[0.5-u,0.5+u,0.5])

    #create lattice vectors
    a1 = [a,0.,0.]
    a2 = [0.,a,0.]
    a3 = [0.,0.,c]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def BaddeleyiteLattice(aa,ab,a,b,c,beta,deg=True):
    #{{{1
    #create lattice base; data from http://cst-www.nrl.navy.mil/lattice/index.html
    # P2_1/c(14), aa=4e,ab=2*4e  
    lb = LatticeBase()
    lb.append(aa,[0.242,0.975,0.025])
    lb.append(aa,[-0.242,0.975+0.5,-0.025+0.5])
    lb.append(aa,[-0.242,-0.975,-0.025])
    lb.append(aa,[0.242,-0.975+0.5,0.025+0.5])
    
    lb.append(ab,[0.1,0.21,0.20])
    lb.append(ab,[-0.1,0.21+0.5,-0.20+0.5])
    lb.append(ab,[-0.1,-0.21,-0.20])
    lb.append(ab,[0.1,-0.21+0.5,0.20+0.5])

    lb.append(ab,[0.39,0.69,0.29])
    lb.append(ab,[-0.39,0.69+0.5,-0.29+0.5])
    lb.append(ab,[-0.39,-0.69,-0.29])
    lb.append(ab,[0.39,-0.69+0.5,0.29+0.5])

    #create lattice vectors
    d2r= numpy.pi/180.
    if deg: beta=beta*d2r
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([0.,b,0.],dtype=numpy.double)
    a3 = numpy.array([c*numpy.cos(beta),0.,c*numpy.sin(beta)],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def WurtziteLattice(aa,ab,a,c):
    #{{{1
    #create lattice base: data from laue atlas (hexagonal ZnS)
    # P63mc; aa=4e,ab=4e  
    lb = LatticeBase()
    lb.append(aa,[0.,0.,0.])
    lb.append(aa,[1/3.,2/3.,0.5])
    
    lb.append(ab,[0.,0,3/8.])
    lb.append(ab,[1/3.,2/3.,3/8.+0.5])

    #create lattice vectors
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([-a/2.,numpy.sqrt(3)*a/2.,0.],dtype=numpy.double)
    a3 = numpy.array([0.,0.,c],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def Hexagonal4HLattice(aa,ab,a,c):
    #{{{1
    #create lattice base: data from laue atlas (hexagonal ZnS) + brainwork by B. Mandl and D. Kriegner
    # ABAC
    lb = LatticeBase()
    lb.append(aa,[0.,0.,0.]) # A
    lb.append(aa,[1/3.,2/3.,0.25]) # B
    lb.append(aa,[0.,0.,0.5]) # A
    lb.append(aa,[2/3.,1/3.,0.75]) # C
    
    lb.append(ab,[0.,0.,0.+3/16.]) # A
    lb.append(ab,[1/3.,2/3.,0.25+3/16.]) # B
    lb.append(ab,[0.,0.,0.5+3/16.]) # A
    lb.append(ab,[2/3.,1/3.,0.75+3/16.]) # C

    #create lattice vectors
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([-a/2.,numpy.sqrt(3)*a/2.,0.],dtype=numpy.double)
    a3 = numpy.array([0.,0.,c],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def Hexagonal6HLattice(aa,ab,a,c):
    #{{{1
    #create lattice base: https://www.ifm.liu.se/semicond/new_page/research/sic/Chapter2.html + brainwork by B. Mandl and D. Kriegner
    # ABCACB
    lb = LatticeBase()
    lb.append(aa,[0.,0.,0.]) # A
    lb.append(aa,[1/3.,2/3.,1/6.]) # B
    lb.append(aa,[2/3.,1/3.,2/6.]) # C
    lb.append(aa,[0.,0.,3/6.]) # A
    lb.append(aa,[2/3.,1/3.,4/6.]) # C
    lb.append(aa,[1/3.,2/3.,5/6.]) # B

    lb.append(ab,[0.,0.,0.+3/24.]) # A
    lb.append(ab,[1/3.,2/3.,1/6.+3/24.]) # B
    lb.append(ab,[2/3.,1/3.,2/6.+3/24.]) # C
    lb.append(ab,[0.,0.,3/6.+3/24.]) # A
    lb.append(ab,[2/3.,1/3.,4/6.+3/24.]) # C
    lb.append(ab,[1/3.,2/3.,5/6.+3/24.]) # B

    #create lattice vectors
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([-a/2.,numpy.sqrt(3)*a/2.,0.],dtype=numpy.double)
    a3 = numpy.array([0.,0.,c],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def TrigonalR3mh(aa,a,c):
    #{{{1
    # create Lattice base from american mineralogist: R3mh (166)
    # http://rruff.geo.arizona.edu/AMS/download.php?id=12092.amc&down=amc
    lb = LatticeBase()
    lb.append(aa,[0.,0.,0.23349])
    lb.append(aa,[2/3.,1/3.,1/3.+0.23349])
    lb.append(aa,[1/3.,2/3.,2/3.+0.23349])

    #create lattice vectors
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([a/2.,numpy.sqrt(3)*a/2.,0.],dtype=numpy.double)
    a3 = numpy.array([0.,0.,c],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)
    
    return l
    #}}}1

def Hexagonal3CLattice(aa,ab,a,c):
    #{{{1
    #create lattice base: data from laue atlas (hexagonal ZnS) + brainwork by B. Mandl and D. Kriegner
    lb = LatticeBase()
    lb.append(aa,[0.,0.,0.])
    lb.append(aa,[1/3.,2/3.,1/3.])
    lb.append(aa,[2/3.,1/3.,2/3.])
    
    lb.append(ab,[0.,0.,0.+1/4.])
    lb.append(ab,[1/3.,2/3.,1/3.+1/4.])
    lb.append(ab,[2/3.,1/3.,2/3.+1/4.])

    #create lattice vectors
    a1 = numpy.array([a,0.,0.],dtype=numpy.double)
    a2 = numpy.array([-a/2.,numpy.sqrt(3)*a/2.,0.],dtype=numpy.double)
    a3 = numpy.array([0.,0.,c],dtype=numpy.double)
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def QuartzLattice(aa,ab,a,b,c):
    #{{{1
    #create lattice base: data from american mineralogist 65 (1980) 920-930
    lb = LatticeBase()
    lb.append(aa,[0.4697,0.,0.])
    lb.append(aa,[0.,0.4697,2/3.])
    lb.append(aa,[-0.4697,-0.4697,1/3.])

    lb.append(ab,[0.4135,0.2669,0.1191])
    lb.append(ab,[0.2669,0.4135,2/3.-0.1191])
    lb.append(ab,[-0.2669,0.4135-0.2669,2/3.+0.1191])
    lb.append(ab,[-0.4135,-0.4135+0.2669,1/3.-0.1191])
    lb.append(ab,[-0.4135+0.2669,-0.4135,1/3.+0.1191])
    lb.append(ab,[0.4135-0.2669,-0.2669,-0.1191])

    #create lattice vectors alpha=beta=90 gamma=120
    ca = numpy.cos(numpy.radians(90))
    cb = numpy.cos(numpy.radians(90))
    cg = numpy.cos(numpy.radians(120))
    sa = numpy.sin(numpy.radians(90))
    sb = numpy.sin(numpy.radians(90))
    sg = numpy.sin(numpy.radians(120))

    a1 = a*numpy.array([1,0,0],dtype=numpy.double)
    a2 = b*numpy.array([cg,sg,0],dtype=numpy.double)
    a3 = c*numpy.array([cb , (ca-cb*cg)/sg , numpy.sqrt(1-ca**2-cb**2-cg**2+2*ca*cb*cg)/sg],dtype=numpy.double)    
    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def TetragonalIndiumLattice(aa,a,c):
    #{{{1
    #create lattice base: I4/mmm (139) site symmetry (2a) 
    # data from: Journal of less common-metals 7 (1964) 17-22 (see american mineralogist database) 
    lb = LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0.5])

    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,c]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1

def NaumanniteLattice(aa,ab,a,b,c):
    #{{{1
    #create lattice base: P 21 21 21 
    # data from: american mineralogist 
    # http://rruff.geo.arizona.edu/AMS/download.php?id=00261.amc&down=amc
    lb = LatticeBase()
    lb.append(aa,[0.107,0.369,0.456])
    lb.append(aa,[0.5-0.107,0.5+0.369,0.5-0.456])
    lb.append(aa,[0.5+0.107,-0.369,-0.456])
    lb.append(aa,[-0.107,0.5-0.369,0.5+0.456])
    
    lb.append(aa,[0.728,0.029,0.361])
    lb.append(aa,[0.5-0.728,0.5+0.029,0.5-0.361])
    lb.append(aa,[0.5+0.728,-0.029,-0.361])
    lb.append(aa,[-0.728,0.5-0.029,0.5+0.361])

    lb.append(ab,[0.358,0.235,0.149])
    lb.append(ab,[0.5-0.358,0.5+0.235,0.5-0.149])
    lb.append(ab,[0.5+0.358,-0.235,-0.149])
    lb.append(ab,[-0.358,0.5-0.235,0.5+0.149])
    
    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,b,0]
    a3 = [0,0,c]

    l = Lattice(a1,a2,a3,base=lb)

    return l
    #}}}1



