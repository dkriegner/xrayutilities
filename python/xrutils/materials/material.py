#class module impolements a certain material

import lattice
import copy

class Material(object):
    def __init__(self,name,lat,c11,c12,c44):
        self.name = name
        self.lattice = lat
        self.rlattice = lat.ReciprocalLattice()
        self._c11 = c11
        self._c12 = c12
        self._c44 = c44

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
        """
        Q(hkl,**keyargs):
        Return the Q-space position for a certain material.

        required input arguments:
        hkl ............. list or numpy array with the Miller indices

        """

        p = self.rlattice.GetPoint(hkl[0],hkl[1],hkl[2])

        return p

    def __str__(self):
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


#calculate some predifined materials
Si = Material("Si",lattice.CubicLattice(5.43104),165.77e+9,63.93e+9,79.62e+9)
Ge = Material("Ge",lattice.CubicLattice(5.65785),124.0e+9,41.3e+9,68.3e+9)
InAs = Material("InAs",lattice.CubicLattice(6.0583),8.34e+11,4.54e+11,3.95e+11)
InP  = Material("InP",lattice.CubicLattice(5.8687),10.11e+11,5.61e+11,4.56e+11)
GaAs = Material("GaAs",lattice.CubicLattice(5.65325),11.9e+11,5.34e+11,5.96e+11)

class AlloyAB(Material):
    def __init__(self,matA,matB,x):
        Material.__init__(self,"None",copy.deepcopy(matA.lattice),matA.c11,matA.c12,matA.c44)
        self.matA = matA
        self.matB = matB
        self.xb = 0
        self._setxb(x)

    def _getxb(self):
        return self.xb

    def _setxb(self,x):
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

    x = property(_getxb,_setxb)


class SiGe(AlloyAB):
    def __init__(self,x):
        AlloyAB.__init__(self,Si,Ge,x)

    def _setxb(self,x):
        a = self.matA.lattice.a1[0]
        print "jump to base class"
        AlloyAB._setxb(self,x)
        print "back from base class"
        #the lattice parameters need to be done in a different way
        a = a+0.2*x+0.027*x**2
        self.lattice = lattice.CubicLattice(a)
        self.rlattice = self.lattice.ReciprocalLattice()
   
    x = property(AlloyAB._getxb,_setxb)
