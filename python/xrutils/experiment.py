#module describing ex parameters

import numpy
import math
import materials
from numpy.linalg import norm


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
    #{{{1
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
            ldel = (delta-ddel)*deg2rad
        else:
            ltth = tth - dtth
            lom  = om - dom
            ldel = delta-ddel

        
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
            return [delta,om,tth]
    #}}}1


class GID(Experiment):
    pass

class GISAXS(Experiment):
    pass


class Powder(Experiment):
    #{{{1
    """
    Experimental class for powder diffraction
    This class is able to produce a powder spectrum for the given material
    """
    def __init__(self,mat,**keyargs):
        Experiment.__init__(self,[0,0,0],[0,0,0],**keyargs)
        if isinstance(mat,materials.Material):
            self.mat = mat
        else:
            raise TypeError,"mat must be an instance of class Material"

        self.digits = 5

    def PowderIntensity(self):
        """
        Calculates the powder intensity and positions up to an angle of 180 deg
        and stores the result in:
            data .... array with intensities
            ang ..... angular position of intensities
            qpos .... reciprocal space position of intensities
        """
        
        # calculate maximal Bragg indices
        hmax = int(numpy.ceil(norm(self.mat.lattice.a1)*self.k0/numpy.pi))
        hmin = -hmax
        kmax = int(numpy.ceil(norm(self.mat.lattice.a2)*self.k0/numpy.pi))
        kmin = -kmax
        lmax = int(numpy.ceil(norm(self.mat.lattice.a3)*self.k0/numpy.pi))
        lmin = -lmax
        
        qlist = []
        qabslist = []
        hkllist = []
        # calculate structure factor for each reflex
        for h in range(hmin,hmax+1):
            for k in range(kmin,kmax+1):
                for l in range(lmin,lmax+1):
                    q = self.mat.rlattice.GetPoint(h,k,l)
                    if norm(q)<2*self.k0:
                        qlist.append(q)
                        hkllist.append([h,k,l])
                        qabslist.append(numpy.round(norm(q),self.digits))
        
        qabs = numpy.array(qabslist,dtype=numpy.double)
        s = self.mat.lattice.StructureFactorForQ(self.energy,qlist)
        r = numpy.absolute(s)**2

        _tmp_data = numpy.zeros(r.size,dtype=[('q',numpy.double),('r',numpy.double),('hkl',list)])
        _tmp_data['q'] = qabs
        _tmp_data['r'] = r
        _tmp_data['hkl'] = hkllist
        # sort the list and compress equal entries
        _tmp_data.sort(order='q')

        self.qpos = [0]
        self.data = [0]
        self.hkl = [[0,0,0]]
        for r in _tmp_data:
            if r[0] == self.qpos[-1]:
                self.data[-1] += r[1]
            elif numpy.round(r[1],self.digits) != 0.:
                self.qpos.append(r[0])
                self.data.append(r[1])
                self.hkl.append(r[2])

        # cat first element to get rid of q = [0,0,0] divergence
        self.qpos = numpy.array(self.qpos[1:],dtype=numpy.double)
        self.ang = self.Q2Ang(self.qpos)  
        self.data = numpy.array(self.data[1:],dtype=numpy.double)
        self.hkl = self.hkl[1:]

        # correct data for polarization and lorentzfactor and unit cell volume
        # and also include Debye-Waller factor for later implementation
        # see L.S. Zevin : Quantitative X-Ray Diffractometry 
        # page 18ff
        polarization_factor = (1+numpy.cos(numpy.deg2rad(2*self.ang))**2)/2.
        lorentz_factor = 1./(numpy.sin(numpy.deg2rad(self.ang))**2*numpy.cos(numpy.deg2rad(self.ang)))
        B=0 # do not have B data yet: they need to be implemented in lattice base class and feeded by the material initialization also the debye waller factor needs to be included there and not here
        debye_waller_factor = numpy.exp(-2*B*numpy.sin(numpy.deg2rad(self.ang))**2/self._wl**2)
        unitcellvol = self.mat.lattice.UnitCellVolume()
        self.data = self.data * polarization_factor * lorentz_factor / unitcellvol**2

    def Convolute(self,stepwidth,width,min=0,max=90):
        """
        Convolutes the intensity positions with Gaussians with angular width 
        of "width". returns array of angular positions with corresponding intensity
            theta ... array with angular positions
            int ..... intensity at the positions ttheta
        """
        
        # define a gaussion which is needed for convolution
        def gauss(amp,x0,sigma,x):
            return amp*numpy.exp(-(x-x0)**2/(2*sigma**2))
        
        # convolute each peak with a gaussian and add them up
        theta = numpy.arange(min,max,stepwidth)
        intensity = numpy.zeros(theta.size,dtype=numpy.double)
        
        for i in range(self.ang.size):
            intensity += gauss(self.data[i],self.ang[i],width,theta)

        return theta,intensity

    def Ang2Q(self,th,deg=True):
        """
        Converts theta angles to reciprocal space positions 
        returns the absolute value of momentum transfer
        """
        if deg:
            lth = numpy.deg2rad(th)
        else:
            lth = th

        qpos = 2*self.k0*numpy.sin(lth)
        return qpos

    def Q2Ang(self,qpos,deg=True):
        """
        Converts reciprocal space values to theta angles
        """
        th = numpy.arcsin(qpos/(2*self.k0))

        if deg:
            th= numpy.rad2deg(th)

        return th
    
    def __str__(self):
        """
        Prints out available information about the material and reflections
        """
        ostr = "\nPowder diffraction object \n"
        ostr += "-------------------------\n"
        ostr += "Material: "+ self.mat.name + "\n"
        ostr += "Lattice:\n" + self.mat.lattice.__str__()
        if self.qpos != None:
            max = self.data.max()
            ostr += "\nReflections: \n"
            ostr += "--------------\n"
            ostr += "      h k l     |    tth    |    Int     |   Int (%)\n"
            for i in range(self.qpos.size):
                ostr += "%15s   %8.4f   %10.2f  %10.2f\n" % (self.hkl[i].__str__(), 2*self.ang[i],self.data[i], self.data[i]/max*100.)

        return ostr
    #}}}1        

    

