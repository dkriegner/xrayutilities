#class module impolements a certain material

import lattice
import copy
import numpy
from numpy import linalg

map_ijkl2ij = {"00":0,"11":1,"22":2,
               "12":3,"20":4,"01":5,
               "21":6,"02":7,"10":8}
map_ij2ijkl = {"0":[0,0],"1":[1,1],"2":[2,2],
        "3":[1,2],"4":[2,0],"5":[0,1],
        "6":[2,1],"7":[0,2],"8":[1,0]}

def index_map_ijkl2ij(i,j):
    return map_ijkl2ij["%i%i" %(i,j)] 

def index_map_ij2ijkl(ij):
    return map_ij2ijkl["%i" %ij]


def Cij2Cijkl(cij):
    #{{{1
    """
    Cij2Cijkl(cij):
    Converts the elastic constants matrix (tensor of rank 2) to 
    the full rank 4 cijkl tensor.

    required input arguments:
    cij ................ (6,6) cij matrix as a numpy array

    return value:
    cijkl .............. (3,3,3,3) cijkl tensor as numpy array
    """

    #first have to build a 9x9 matrix from the 6x6 one
    m = numpy.zeros((9,9),dtype=numpy.double)
    m[0:6,0:6] = cij[:,:]
    m[6:9,0:6] = cij[3:6,:]
    m[0:6,6:9] = cij[:,3:6]
    m[6:9,6:9] = cij[3:6,3:6]

    #now create the full tensor
    cijkl = numpy.zeros((3,3,3,3),dtype=numpy.double)

    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    mi = index_map_ijkl2ij(i,j)
                    mj = index_map_ijkl2ij(k,l)
                    cijkl[i,j,k,l] = m[mi,mj]
    return cijkl
    #}}}1

def Cijkl2Cij(cijkl):
    #{{{1
    """
    Cijkl2Cij(cijkl):
    Converts the full rank 4 tensor of the elastic constants to 
    the (6,6) matrix of elastic constants.

    required input arguments:
    cijkl .............. (3,3,3,3) cijkl tensor as numpy array

    return value:
    cij ................ (6,6) cij matrix as a numpy array
    """
    
    #build the temporary 9x9 matrix
    m = numpy.zeros((9,9),dtype=numpy.double)

    for i in range(0,9):
        for j in range(0,9):
            ij = index_map_ij2ijkl(i)
            kl = index_map_ij2ijkl(j)
            m[i,j] = cijkl[ij[0],ij[1],kl[0],kl[1]]

    cij = m[0:6,0:6]

    return cij
    #}}}1

class Material(object):
    #{{{1
    def __init__(self,name,lat,cij):
        #{{{2
        if isinstance(cij,list):
            self.cij = numpy.array(cij,dtype=numpy.double)
        elif isinstance(cij,numpy.ndarray):
            self.cij = cij
        else:
            raise TypeError,"Elastic constants must be a list or numpy array!"

        self.name = name
        self.lattice = lat
        self.rlattice = lat.ReciprocalLattice()
        self.cijkl = Cij2Cijkl(self.cij)
        self.transform = None
        #}}}2

    def __getattr__(self,name):
        if name.startswith("c"):
            index = name[1:]
            if len(index)>2:
                raise AttributeError,"Cij indices must be between 1 and 6"

            i=int(index[0])
            j=int(index[1])

            if i>6 or i<1 or j>6 or j<1:
                raise AttributeError,"Cij indices must be between 1 and 6"

            if self.transform:
                cij = self.transform(self.cij)
            else:
                cij = self.cij

            return cij[i-1,j-1]
    
    def _getmu(self):
        return self.cij[3,3]

    def _getlam(self):
        return self.cij[0,1]

    def _getnu(self):
        return self.lam/2./(self.mu+self.lam)

    def _geta1(self):
        return self.lattice.a1
    
    def _geta2(self):
        return self.lattice.a2
    
    def _geta3(self):
        return self.lattice.a3
    
    def _getb1(self):
        return self.rlattice.b1
    
    def _getb2(self):
        return self.rlattice.b2
    
    def _getb3(self):
        return self.rlattice.b3

    mu  = property(_getmu)
    lam = property(_getlam)
    nu  = property(_getnu)
    a1 = property(_geta1)
    a2 = property(_geta2)
    a3 = property(_geta3)
    b1 = property(_getb1)
    b2 = property(_getb2)
    b3 = property(_getb3)


    def Q(self,hkl):
        #{{{2
        """
        Q(hkl,**keyargs):
        Return the Q-space position for a certain material.

        required input arguments:
        hkl ............. list or numpy array with the Miller indices

        """

        p = self.rlattice.GetPoint(hkl[0],hkl[1],hkl[2])
        if self.transform: p = self.transform(p)

        return p
        #}}}2

    def __str__(self):
        #{{{2
        ostr ="Material: %s\n" %self.name
        ostr += "Elastic constants:\n"
        ostr += "c11 = %e\n" %self.c11
        ostr += "c12 = %e\n" %self.c12
        ostr += "c44 = %e\n" %self.c44
        ostr += "mu  = %e\n" %self.mu
        ostr += "lam = %e\n" %self.lam
        ostr += "nu  = %e\n" %self.nu
        ostr += "Lattice:\n"
        ostr += self.lattice.__str__()
        ostr += "Reciprocal lattice:\n"
        ostr += self.rlattice.__str__()

        return ostr
        #}}}2

    def ApplyStrain(self,strain,**keyargs):
        #let strain act on the base vectors
        self.lattice.ApplyStrain(strain)
        #recalculate the reciprocal lattice
        self.rlattice = self.lattice.ReciprocalLattice()

    def GetMissmatch(self,mat):
        """
        GetMissmatch(mat):
        Calculate the mismatch strain between  
        """
        print "not implemented yet"
    #}}}1


def CubicElasticTensor(c11,c12,c44):
    """
    CubicElasticTensor(c11,c12,c44):
    Assemble the 6x6 matrix of elastic constants for a cubic material.


    """
    m = numpy.zeros((6,6),dtype=numpy.double)
    m[0,0] = c11; m[1,1] = c11; m[2,2] = c11;
    m[3,3] = c44; m[4,4] = c44; m[5,5] = c44;
    m[0,1] = m[0,2] = c12
    m[1,0] = m[1,2] = c12
    m[2,0] = m[2,1] = c12

    return m

#calculate some predifined materials
Si = Material("Si",lattice.CubicLattice(5.43104),
                   CubicElasticTensor(165.77e+9,63.93e+9,79.62e+9))
Ge = Material("Ge",lattice.CubicLattice(5.65785),
                   CubicElasticTensor(124.0e+9,41.3e+9,68.3e+9))
InAs = Material("InAs",lattice.CubicLattice(6.0583),
                   CubicElasticTensor(8.34e+11,4.54e+11,3.95e+11))
InP  = Material("InP",lattice.CubicLattice(5.8687),
                   CubicElasticTensor(10.11e+11,5.61e+11,4.56e+11))
GaAs = Material("GaAs",lattice.CubicLattice(5.65325),
                   CubicElasticTensor(11.9e+11,5.34e+11,5.96e+11))
CdTe = Material("CdTe",lattice.CubicLattice(6.48),
                   CubicElasticTensor(53.5,36.7,19.9))
PbTe = Material("PbTe",lattice.CubicLattice(6.462),
                   CubicElasticTensor(93.6,7.7,13.4))
PbSe = Material("PbSe",lattice.CubicLattice(6.126),
                   CubicElasticTensor(123.7,19.3,15.9))

class AlloyAB(Material):
    #{{{1
    def __init__(self,matA,matB,x):
        #{{{2
        Material.__init__(self,"None",copy.deepcopy(matA.lattice),matA.cij)
        self.matA = matA
        self.matB = matB
        self.xb = 0
        self._setxb(x)
        #}}}2

    def _getxb(self):
        return self.xb

    def _setxb(self,x):
        #{{{2
        self.xb = x
        self.name = "%s(%2.2f)%s(%2.2f)" %(self.matA.name,1.-x,self.matB.name,x)
        #modify the lattice
        self.lattice.a1 = (self.matB.lattice.a1-self.matA.lattice.a1)*x+\
                          self.matA.lattice.a1
        self.lattice.a2 = (self.matB.lattice.a2-self.matA.lattice.a2)*x+\
                          self.matA.lattice.a2
        self.lattice.a3 = (self.matB.lattice.a3-self.matA.lattice.a3)*x+\
                          self.matA.lattice.a3
        self.rlattice = self.lattice.ReciprocalLattice()

        #set elastic constants
        self.cij = (self.matB.cij-self.matA.cij)*x+self.matA.cij
        self.cijkl = (self.matB.cijkl-self.matA.cijkl)*x+self.matA.cijkl
        
        #}}}2

    x = property(_getxb,_setxb)
    #}}}1


class SiGe(AlloyAB):
    #{{{1
    def __init__(self,x):
        AlloyAB.__init__(self,Si,Ge,x)

    def _setxb(self,x):
        #{{{2
        a = self.matA.lattice.a1[0]
        print "jump to base class"
        AlloyAB._setxb(self,x)
        print "back from base class"
        #the lattice parameters need to be done in a different way
        a = a+0.2*x+0.027*x**2
        self.lattice = lattice.CubicLattice(a)
        self.rlattice = self.lattice.ReciprocalLattice()
        #}}}2
   
    x = property(AlloyAB._getxb,_setxb)
    #}}}1

def PseudomorphicMaterial(submat,layermat):
    #{{{1
    """
    PseudomprohicMaterial(submat,layermat):
    This function returns a material whos lattice is pseudomorphic on a
    particular substrate material.
    This function works meanwhile only for cubit materials.

    required input arguments:
    submat .................... substrate material
    layermat .................. bulk material of the layer

    return value:
    An instance of Material holding the new pseudomorphically strained 
    material.
    """
    pmat = copy.copy(layermat)

    a1 = submat.lattice.a1
    a2 = submat.lattice.a2

    #calculate the normal lattice parameter

    abulk = layermat.lattice.a1[0] 
    ap    = a1[0]
    c11 = layermat.c11
    c12 = layermat.c12
    a3 = numpy.zeros(3,dtype=numpy.double)
    a3[2] =  abulk-2.0*c12*(ap-abulk)/c11
    #set the new lattice for the pseudomorphic material
    pmat.lattice = lattice.Lattice(a1,a2,a3)
    pmat.rlattice = pmat.lattice.ReciprocalLattice()

    return pmat
    #}}}1


def AssembleCubicElasticConstants(c11,c12,c13,c23,c44,c55,c66):
    #{{{1
    """
    AssembleCubicElasticConstants(c11,c12,c13,c23,c44,c55,c66):
    Assemble the elastic constants matrix for a cubic system. This function 
    simply reduces writting work and therefore the risk of typos.
    """

    m = numpy.array([[c11,c12,c12,0,0,0],[c12,c11,c12,0,0,0],[c12,c12,c11,0,0,0],
                     [0,0,0,c44,0,0],[0,0,0,0,c44,0],[0,0,0,0,0,c44]],
                     dtype=numpy.double)
    return m
    #}}}1


