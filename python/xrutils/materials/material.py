#class module impolements a certain material

import lattice

class Material(object):
    def __init__(self,lat,c11,c12,c44):
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

    def Q(self,hkl,**keyargs):
        """
        Q(hkl,**keyargs):
        Return the Q-space position for a certain material.

        required input arguments:
        hkl ............. list or numpy array with the Miller indices

        optional keyword arguments:
        azimuth ......... a transformation describing the azimuthal direction
        """

        p = self.rlattice.GetPoint(hkl[0],hkl[1],hkl[2])
        if keyargs.has_key("azimuth"):
            p = keyargs["azimuth"](p)

        return p


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

        pass

#calculate some predifined materials
Si = Material(lattice.CubicLattice(5.43104),165.77e+9,63.93e+9,79.62e+9)
Ge = Material(lattice.CubicLattice(5.65785),124.0e+9,41.3e+9,68.3e+9)

class SiGeAlloy(Material):
    SiComp = Si
    GeComp = Ge
    def __init__(self,x):
        Material.__init__(self,self.SiComp.lattice,
                          self.SiComp.c11,self.SiComp.c12,
                          self.SiComp.c44)
        self.xge = x
        self.__update_xge__()


    def _getxge(self):
        return self.xge

    def _setxge(self,x):
        self.xge = x
        self.__update_xge__()
    
    x = property(_getxge,_setxge)

    def __update_xge__(self):
        a = self.Si.lattice.a1[0]+0.2*self.xge+0.027*self.xge**2
        self.lattice = lattice.CubicLattice(a)
        self.rlattice = self.lattice.ReciprocalLattice()
        self.c11 = (self.GeComp.c11-self.SiComp.c11)*self.xge + \
                   self.SiComp.c11
        self.c12 = (self.GeComp.c12-self.SiComp.c12)*self.xge + \
                   self.SiComp.c12
        self.c44 = (self.GeComp.c44-self.SiComp.c44)*self.xge + \
                   self.SiComp.c44

   
class AnisotropicMaterial(Material):
    def __init__(self):
        pass
