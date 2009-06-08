#class module impolements a certain material

import lattice
import copy
import numpy
from numpy import linalg

class Material(object):
    #{{{1
    def __init__(self,name,lat,c11,c12,c44):
        #{{{2
        self.name = name
        self.lattice = lat
        self.rlattice = lat.ReciprocalLattice()
        self._c11 = c11
        self._c12 = c12
        self._c44 = c44
        #}}}2

    def _getc11(self):
        return self._c11

    def _setc11(self,x):
        self._c11 = x

    def _getc12(self):
        return self._c12

    def _setc12(self,x):
        self._c12 = x

    def _getc44(self):
        return self._c44

    def _setc44(self,x):
        self._c44 = x

    def _getnu(self):
        return self.lam/2./(self.mu+self.lam)
    #handle elastic constants
    c11 = property(_getc11,_setc11)
    c12 = property(_getc12,_setc12)
    c44 = property(_getc44,_setc44)
    mu  = property(_getc44,_setc44)
    lam = property(_getc12,_setc12)
    nu  = property(_getnu)

    def Q(self,hkl):
        #{{{2
        """
        Q(hkl,**keyargs):
        Return the Q-space position for a certain material.

        required input arguments:
        hkl ............. list or numpy array with the Miller indices

        """

        p = self.rlattice.GetPoint(hkl[0],hkl[1],hkl[2])

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


#calculate some predifined materials
Si = Material("Si",lattice.CubicLattice(5.43104),165.77e+9,63.93e+9,79.62e+9)
Ge = Material("Ge",lattice.CubicLattice(5.65785),124.0e+9,41.3e+9,68.3e+9)
InAs = Material("InAs",lattice.CubicLattice(6.0583),8.34e+11,4.54e+11,3.95e+11)
InP  = Material("InP",lattice.CubicLattice(5.8687),10.11e+11,5.61e+11,4.56e+11)
GaAs = Material("GaAs",lattice.CubicLattice(5.65325),11.9e+11,5.34e+11,5.96e+11)
CdTe = Material("CdTe",lattice.CubicLattice(6.48),53.5,36.7,19.9)
PbTe = Material("PbTe",lattice.CubicLattice(6.462),93.6,7.7,13.4)
PbSe = Material("PbSe",lattice.CubicLattice(6.126),123.7,19.3,15.9)

class AlloyAB(Material):
    #{{{1
    def __init__(self,matA,matB,x):
        #{{{2
        Material.__init__(self,"None",copy.deepcopy(matA.lattice),matA.c11,matA.c12,matA.c44)
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
        self.c11 = (self.matB.c11-self.matA.c11)*x+self.matA.c11
        self.c12 = (self.matB.c12-self.matA.c12)*x+self.matA.c12
        self.c44 = (self.matB.c44-self.matA.c44)*x+self.matA.c44
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


