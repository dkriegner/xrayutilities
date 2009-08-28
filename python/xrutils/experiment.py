#module describing ex parameters

import numpy
import math
    
_e_const = 1.60219e-19
_h_const = 6.62602e-34
_c_const = 2.997925e8

class Experiment(object):
    #{{{1
    def __init__(self,ipdir,ndir,**keyargs):
        #{{{2
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
            self._wl = _c_const*_h_const/self._en/_e_const/1.e-10
        else:
            self._en = _c_const*_h_const/self._wl/1.e-10/_e_const

        self.k0 = numpy.pi*2./self._wl
        #}}}2


    def __str__(self):
        #{{{2
        ostr = "inplane azimuth: (%f %f %f)\n" %(self.idir[0],
                                                 self.idir[1],
                                                 self.idir[2])
        ostr += "surface normal: (%f %f %f)\n" %(self.ndir[0],
                                                 self.ndir[1],
                                                 self.ndir[2])
        ostr += "energy: %f (eV)\n" %self._en
        ostr += "wavelength: %f (Anstrom)\n" %(self._wl)

        return ostr
        #}}}2

    def _set_energy(self,energy):
        #{{{2
        self._en = energy
        self._wl = _c_const*_h_const/self._en/_e_const
        self.k0 = numpy.pi*2./self._wl
        #}}}2

    def _set_wavelength(self,wl):
        #{{{2
        self._wl = wl
        self._en = _c_const*_h_const/self._wl/_e_const
        self.k0 = numpy.pi*2./self._wl
        #}}}2

    def _get_energy(self):
        return self._en

    def _get_wavelength(self):
        return self._wl

    energy = property(_get_energy,_set_energy)
    wavelength = property(_get_wavelength,_set_wavelength)

    def _set_inplane_direction(self,dir):
        #{{{2
        if isinstance(dir,list):
            self.idir = numpy.ndarray(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.idir = dir
        else:
            raise TypeError,"Inplane direction must be list or numpy array"

        v1 = numpy.cross(self.ndir,self.idir)
        self.transform = math.CoordinateTransform(v1,self.idir,self.ndir)
        #}}}2

    def _get_inplane_direction(self):
        return self.idir

    def _set_normal_direction(self,dir):
        #{{{2
        if isinstance(dir,list):
            self.ndir = numpy.ndarray(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.ndir = dir
        else:
            raise TypeError,"Surface normal must be list or numpy array"

        v1 = numpy.cross(self.ndir,self.idir)
        self.transform = math.CoordinateTransform(v1,self.idir,self.ndir)
        #}}}2

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

    def TiltAngle(self,q,deg=True):
        """
        TiltAngle(q,deg=True):
        Return the angle between a q-space position and the surface normal.

        required input arguments:
        q ...................... list or numpy array with the reciprocal space 
                                 position
        """

        if isinstance(q,list):
            qt = numpy.array(q,dtype=numpy.double)
        elif isinstance(q,numpy.ndarray):
            qt = q
        else:
            raise TypeError,"q-space position must be list or numpy array"

        return math.VecAngle(self.ndir,qt,deg)

    #}}}1



class HXRD(Experiment):
    def __init__(self,idir,ndir,**keyargs):
        Experiment.__init__(self,idir,ndir,**keyargs)

        self.geometry = "hi_lo"

    def TiltCorr(self,q,a,deg=False):
        """
        TiltCorr(q):
        Correct a q-space position by a certain tilt angle.

        required input arguments:
        q ................. list or numpy array with the tilted q-space position
        a ................. tilt angle

        optional keyword arguments:
        deg ............... True/False (default False) whether the input data is 
                            in degree or radiants

        return value:
        numpy array with the corrected q-space position.
        """


        #calculate the angular position of the q-space point
        [om,tth,delta] = self.Q2Ang(q)

        #calcualte the new direction of the peak
        q = self.Ang2Q(om-a,tth,delta)

        return q

    def Ang2Q(self,om,tth,delta,deg=True,dom=0.,dtth=0.,ddel=0.):
        """
        Ang2Q(om,tth,delta,deg=True,dom=0.,dtth=0.,ddel=0.):
        Convert angular positions into Q-space positions.
        The method can treate 2D and 3D data maps by setting delta in the 
        appropriate way.

        required input arguments:
        om ..................... omega angle
        tth .................... 2theta scattering angle
        delta .................. off-plane angle (apart the scattering plane)

        optional keyword arguments:
        dom .................... omega offset
        dtth ................... tth offset
        ddel ................... delta offset
        deg .................... True/Flase (default is True) determines whether 
                                 or not input angles are given in degree or
                                 radiants

        return value:
        [qx,qy,qz] ............. array of q-space values
        """
        
        deg2rad = numpy.pi/180.0;
        if deg:
            ltth = (tth-dtth)*deg2rad
            lom  = (om-dom)*deg2rad
            ldel = delta*deg2rad
        else:
            ltth = tth - dtth
            lom  = om - dom

        
        qx=2.0*self.k0*numpy.sin(ltth*0.5)*numpy.sin(lom-0.5*ltth)*numpy.sin(ldel)
        qy=2.0*self.k0*numpy.sin(ltth*0.5)*numpy.sin(lom-0.5*ltth)*numpy.cos(ldel)
        qz=2.0*self.k0*numpy.sin(0.5*ltth)*numpy.cos(lom-0.5*ltth)      

        return [qx,qy,qz];    

    def Q2Ang(self,Q,trans=False,geom="hi_lo",deg=True):
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
        deg ................... True/Flase (default True) determines if the
                                angles are returned in radiants or degrees

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
        tth = 2.*numpy.arcsin(qa/2./self.k0)

        #calculation of the delta angle
        delta = numpy.arctan(q[0]/q[1])
        if numpy.isnan(delta):
            delta =0 
        
        om1 = numpy.arcsin(q[1]/qa/numpy.cos(delta))+0.5*tth
        om2 = numpy.arcsin(q[0]/qa/numpy.sin(delta))+0.5*tth
        if numpy.isnan(om1):
            om = om2
        elif numpy.isnan(om2):
            om = om1
        else:
            om = om1

        #have to take now the scattering geometry into account
        
        rad2deg = 180./numpy.pi
        if deg:
            return [rad2deg*delta,rad2deg*om,rad2deg*tth]
        else:
            return [delta,om,tth,delta]



class GID(Experiment):
    pass

class GISAXS(Experiment):
    pass



    

