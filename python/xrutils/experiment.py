#module describing ex parameters

import numpy
import math
    
_e_const = 1.60219e-19
_h_const = 6.62602e-34
_c_const = 2.997925e8

class Experiment(object):
    def __init__(self,ipdir,ndir,**keyargs):
        if isinstance(ipdir,list):
            self.idir = numpy.array(ipdir,dtype=numpy.double)
        elif isinstance(ipdir,numpy.ndarray):
            self.idir = ipdir
        else:
            raise TypeError,"Inplane direction must be list or numpy array"
        
        if isinstance(ndir,list):
            self.ndir = numpy.array(ndir,dtype=numpy.double)
        elif isinstance(ndir,numpy.ndarray):
            self.ndir = ndir
        else:
            raise TypeError,"normal direction must be list or numpy array"
        
        #set the coordinate transform for the azimuth used in the experiment
        v1 = numpy.cross(self.ndir,self.idir)
        self.transform = math.CoordinateTransform(v1,self.idir,self.ndir)
        
        #calculate the energy from the wavelength
        if keyargs.has_key("wl"):
            self._wl = keyargs["wl"]
        else:
            self._wl = 1.5406

        if keyargs.has_key("en"):
            self._en = keyargs["en"]
            self._wl = _c_const*_h_const/self._en/_e_const
        else:
            self._en = _c_const*_h_const/self._wl/_e_const

        self.k0 = numpy.pi*2./self._wl


    def __str__(self):
        ostr = "inplane azimuth: (%f %f %f)\n" %(self.idir[0],
                                                 self.idir[1],
                                                 self.idir[2])
        ostr += "surface normal: (%f %f %f)\n" %(self.ndir[0],
                                                 self.ndir[1],
                                                 self.ndir[2])
        ostr += "energy: %f (eV)\n" %self._en
        ostr += "wavelength: %f (Anstrom)\n" %(self._wl)

        return ostr

    def _set_energy(self,energy):
        self._en = energy
        self._wl = _c_const*_h_const/self._en/_e_const
        self.k0 = numpy.pi*2./self._wl

    def _set_wavelength(self,wl):
        self._wl = wl
        self._en = _c_const*_h_const/self._wl/_e_const
        self.k0 = numpy.pi*2./self._wl

    def _get_energy(self):
        return self.en

    def _get_wavelength(self):
        return self.wl

    energy = property(_get_energy,_set_energy)
    wavelength = property(_get_wavelength,_set_wavelength)

    def _set_inplane_direction(self,dir):
        if isinstance(dir,list):
            self.idir = numpy.ndarray(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.idir = dir
        else:
            raise TypeError,"Inplane direction must be list or numpy array"

        v1 = numpy.cross(self.ndir,self.idir)
        self.transform = math.CoordinateTransform(v1,self.idir,self.ndir)

    def _get_inplane_direction(self):
        return self.idir

    def _set_normal_direction(self,dir):
        if isinstance(dir,list):
            self.ndir = numpy.ndarray(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.ndir = dir
        else:
            raise TypeError,"Surface normal must be list or numpy array"

        v1 = numpy.cross(self.ndir,self.idir)
        self.transform = math.CoordinateTransform(v1,self.idir,self.ndir)

    def _get_normal_direction(self):
        return self.ndir

    def Ang2Q(self,a1,a2,a3):
        pass

    def Q2Ang(self,q1,q2,q3):
        pass

    def Transform(self,v):
        return self.transform(v)

    def AlignIntensity(self,data):
        pass

    def Align2DMatrix(self,data):
        return numpy.flipud(numpy.rot90(data))



class HXRD(Experiment):
    def __init__(self,idir,ndir,**keyargs):
        Experiment.__init__(self,idir,ndir,**keyargs)

    def Ang2Q(self,om,tth,delta,deg=true,dom=0.,dtth=0.,ddel=0.):
        """
        Ang2Q(om,tth,delta,deg=true,dom=0.,dtth=0.,ddel=0.):
        Convert angular positions into Q-space positions.

        """
        
        deg2rad = numpy.pi/180.0;
        if deg:
            ltth = (tth-dtth)*deg2rad
            lom  = (om-dom)*deg2rad

        qx=2.0*self.k0*numpy.sin(ltth*0.5)*numpy.sin(lom-0.5*ltth);
        qz=2.0*self.k0*numpy.sin(0.5*ltth)*numpy.cos(lom-0.5*ltth);        

        return [qx,qz];    

    def Q2Ang(self,Q,trans=False,geom="hi_lo"):
        """
        Q2Ang(Q,trans=False):
        Convert a reciprocal space vector Q to scattering angles.
        The keyword argument trans determines wether Q should be transformed 
        to the experimental coordinate frame or not. 

        required input arguments:
        Q .................... a list or numpy array of shape (3) with 
                               q-space vector components

        optional keyword arguments:
        trans ................. True/Flase apply coordinate transformation 
                                on Q
        geom .................. determines the scattering geometry:
                                "hi_lo" (default) high incidencet-low exit
                                "lo_hi" low incidence - high exit

        return value:
        a numpy array of shape (3) with all three scattering angles.
        """

        if isinstance(Q,list):
            q = numpy.array(Q,dtype=numpy.double)
        elif isinstance(Q,numpy.ndarray):
            q = Q
        else:
            raise TypeError,"Q vector must be a list or numpy array"

        if trans:
            q = self.transform(q)

        qa = math.VecNorm(q)



class GID(Experiment):
    pass



    

