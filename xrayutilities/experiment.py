# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2009-2010 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2014 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2012 Tanja Etzelstorfer <tanja.etzelstorfer@jku.at>

"""
module helping with planning and analyzing experiments

various classes are provided for

* describing experiments
* calculating angular coordinates of Bragg reflections
* converting angular coordinates to Q-space and vice versa
* simulating (first order approx. -> just peak positions) powder diffraction patterns for materials
"""

import numbers
import numpy
from numpy.linalg import norm
import warnings
import re

# package internal imports
from . import math
from . import materials
from . import utilities
from . import cxrayutilities
from . import config
from .exception import InputError

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str

# regular expression to check goniometer circle syntax
circleSyntax = re.compile("[xyz][+-]")
circleSyntaxSample = re.compile("[xyzk][+-]")

class QConversion(object):
    """
    Class for the conversion of angular coordinates to momentum space for
    arbitrary goniometer geometries and X-ray energy.  Both angular scans
    (where some goniometer angles change during data acquisition) and energy
    scans (where the energy is varied during acquisition) as well as mixed
    cases can be treated.

    the class is configured with the initialization and does provide three
    distinct routines for conversion to momentum space for

    * point detector:     point(...) or __call__()
    * linear detector:    linear(...)
    * area detector:      area(...)

    linear() and area() can only be used after the init_linear()
    or init_area() routines were called
    """
    def __init__(self,sampleAxis,detectorAxis,r_i,**kwargs):
        """
        initialize QConversion object.
        This means the rotation axis of the sample and detector circles
        need to be given: starting with the outer most circle.

        Parameters
        ----------
        sampleAxis:     list or tuple of sample circles,
                        e.g. ['x+','z+'] would mean two sample circles whereas
                        the outer one turns righthanded around the x axis and the inner
                        one turns righthanded around z.

        detectorAxis:   list or tuple of detector circles
                        e.g. ['x-'] would mean a detector arm with a single motor turning
                        lefthanded around the x-axis.

        r_i:            vector giving the direction of the primary beam
                        (length is irrelevant)

        **kwargs:       optional keyword arguments
            wl:         wavelength of the x-rays in Angstroem
            en:         energy of the x-rays in electronvolt
            UB:         matrix for conversion from (hkl) coordinates to Q of sample
                        used to determine not Q but (hkl)
                        (default: identity matrix)
        """

        for k in kwargs.keys():
            if k not in ['wl','en','UB']:
                raise Exception("unknown keyword argument given: allowed are 'en': for x-ray energy, 'wl': x-ray wavelength, 'UB': orientation/orthonormalization matrix")
        
        #initialize some needed variables
        self._kappa_dir = numpy.array((numpy.nan,numpy.nan,numpy.nan))

        # set/check sample and detector axis geometry
        self._set_sampleAxis(sampleAxis)
        self._set_detectorAxis(detectorAxis)

        # r_i: primary beam direction
        if isinstance(r_i,(list,tuple,numpy.ndarray)):
            self.r_i = numpy.array(r_i,dtype=numpy.double)
            if self.r_i.size != 3:
                print("XU.QConversion: warning invalid primary beam direction given -> using [0,1,0]")
                self.r_i = numpy.array([0,1,0],dtype=numpy.double)
        else:
            raise TypeError("QConversion: invalid type of primary beam direction r_i, must be tuple, list or numpy.ndarray")

        # kwargs
        if "wl" in kwargs:
            self._set_wavelength(kwargs["wl"])
        else:
            self._set_wavelength(config.WAVELENGTH)

        if "en" in kwargs:
            self._set_energy(kwargs["en"])

        if 'UB' in kwargs:
            self._set_UB(kwargs['UB'])
        else:
            self._set_UB(numpy.identity(3))

        self._linear_init = False
        self._area_init = False
        self._area_detrotaxis_set = False


    def _set_energy(self,energy):
        self._en = utilities.energy(energy)
        self._wl = utilities.en2lam(self._en)

    def _set_wavelength(self,wl):
        self._wl = utilities.wavelength(wl)
        self._en = utilities.lam2en(self._wl)

    def _get_energy(self):
        return self._en

    def _get_wavelength(self):
        return self._wl

    def _set_sampleAxis(self,sampleAxis):
        """
        property handler for _sampleAxis
        checks if a syntactically correct list of sample circles is given

        Parameter
        ---------
         sampleAxis:     list or tuple of sample circles, e.g. ['x+','z+']
        """

        if isinstance(sampleAxis,(basestring,list,tuple)):
            if isinstance(sampleAxis,basestring):
                sAxis = list([sampleAxis])
            else:
                sAxis = list(sampleAxis)
            for circ in sAxis:
                if not isinstance(circ,basestring) or len(circ)!=2:
                    raise InputError("QConversion: incorrect sample circle type or syntax (%s)" %repr(circ))
                if not circleSyntaxSample.search(circ):
                    raise InputError("QConversion: incorrect sample circle syntax (%s)" %circ)
                if circ[0]=='k': # determine kappa rotation axis
                    # determine reference direction
                    if config.KAPPA_PLANE[0] == 'x':
                        self._kappa_dir = numpy.array((1.,0,0))
                        # turn reference direction
                        if config.KAPPA_PLANE[1] =='y':
                            self._kappa_dir = math.ZRotation(config.KAPPA_ANGLE)(self._kappa_dir)
                        elif config.KAPPA_PLANE[1] == 'z':
                            self._kappa_dir = math.YRotation(-config.KAPPA_ANGLE)(self._kappa_dir)
                        else:
                            raise TypeError("Qconverision init: invalid kappa_plane in config!")
                    elif config.KAPPA_PLANE[0] == 'y':
                        self._kappa_dir = numpy.array((0,1.,0))
                        # turn reference direction
                        if config.KAPPA_PLANE[1] =='z':
                            self._kappa_dir = math.XRotation(config.KAPPA_ANGLE)(self._kappa_dir)
                        elif config.KAPPA_PLANE[1] == 'x':
                            self._kappa_dir = math.ZRotation(-config.KAPPA_ANGLE)(self._kappa_dir)
                        else:
                            raise TypeError("Qconverision init: invalid kappa_plane in config!")
                    elif config.KAPPA_PLANE[0] == 'z':
                        self._kappa_dir = numpy.array((0,0,1.))
                        # turn reference direction
                        if config.KAPPA_PLANE[1] =='x':
                            self._kappa_dir = math.YRotation(config.KAPPA_ANGLE)(self._kappa_dir)
                        elif config.KAPPA_PLANE[1] == 'y':
                            self._kappa_dir = math.XRotation(-config.KAPPA_ANGLE)(self._kappa_dir)
                        else:
                            raise TypeError("Qconverision init: invalid kappa_plane in config!")
                    else:
                        raise TypeError("Qconverision init: invalid kappa_plane in config!")

                    # rotation sense
                    if circ[1]=='-':
                        self._kappa_dir = -self._kappa_dir

                    if config.VERBOSITY >= config.DEBUG:
                        print("XU.QConversion: kappa_dir: (%5.3f %5.3f %5.3f)" % tuple(self._kappa_dir))

        else:
            raise TypeError("Qconversion error: invalid type for sampleAxis, must be str, list, or tuple")
        self._sampleAxis = sAxis
        self._sampleAxis_str = ''
        for circ in self._sampleAxis:
            self._sampleAxis_str += circ

    def _get_sampleAxis(self):
        """
        property handler for _sampleAxis

        Returns
        -------
        list of sample axis following the syntax /[xyzk][+-]/
        """
        return self._sampleAxis

    def _set_detectorAxis(self,detectorAxis,detrot=False):
        """
        property handler for _detectorAxis_
        checks if a syntactically correct list of detector circles is given

        Parameter
        ---------
         detectorAxis:     list or tuple of detector circles, e.g. ['x+']
         detrot:           flag to tell that the detector rotation is going to be added
                           (used internally to avoid double adding of detector rotation axis)
        """
        if isinstance(detectorAxis,(basestring,list,tuple)):
            if isinstance(detectorAxis,basestring):
                dAxis = list([detectorAxis])
            else:
                dAxis = list(detectorAxis)
            for circ in dAxis:
                if not isinstance(circ,basestring) or len(circ)!=2:
                    raise InputError("QConversion: incorrect detector circle type or syntax (%s)" %repr(circ))
                if not circleSyntax.search(circ):
                    raise InputError("QConversion: incorrect detector circle syntax (%s)" %circ)
        else:
            raise TypeError("Qconversion error: invalid type for detectorAxis, must be str, list or tuple")
        self._detectorAxis = dAxis
        self._detectorAxis_str = ''
        for circ in self._detectorAxis:
            self._detectorAxis_str += circ
        if detrot:
            self._area_detrotaxis_set = True
        else:
            self._area_init = False

    def _get_detectorAxis(self):
        """
        property handler for _detectorAxis

        Returns
        -------
        list of detector axis following the syntax /[xyz][+-]/
        """
        return self._detectorAxis

    def _get_UB(self):
        return self._UB

    def _set_UB(self,UB):
        """
        set the orientation matrix used in the Qconversion
        needs to be (3,3) matrix
        """
        tmp = numpy.array(UB)
        if tmp.shape!=(3,3):
            raise InputError("QConversion: incorrect shape of UB matrix (shape: %s)" %str(tmp.shape))
        else:
            self._UB = tmp

    energy = property(_get_energy,_set_energy)
    wavelength = property(_get_wavelength,_set_wavelength)
    sampleAxis = property(_get_sampleAxis,_set_sampleAxis)
    detectorAxis = property(_get_detectorAxis,_set_detectorAxis)
    UB = property(_get_UB,_set_UB)

    def __str__(self):
        pstr =  'QConversion geometry \n'
        pstr += '---------------------------\n'
        pstr += 'sample geometry(%d): ' %len(self._sampleAxis) + self._sampleAxis_str + '\n'
        if self._sampleAxis_str.find('k') != -1:
            pstr += 'kappa rotation axis (%5.3f %5.3f %5.3f)\n' % tuple(self._kappa_dir)
        pstr += 'detector geometry(%d): ' %len(self._detectorAxis) + self._detectorAxis_str + '\n'
        pstr += 'primary beam direction: (%5.2f %5.2f %5.2f) \n' %(self.r_i[0],self.r_i[1],self.r_i[2])

        if self._linear_init:
            pstr += '\n linear detector initialized:\n'
            pstr += 'linear detector mount direction: ' + self._linear_detdir + '\n'
            pstr += 'number of channels/center channel: %d/%d\n' %(self._linear_Nch,self._linear_cch)
            pstr += 'distance to center of rotation/pixel width: %10.4g/%10.4g \n' %(self._linear_distance,self._linear_pixwidth)
            chpdeg = 2*self._linear_distance/self._linear_pixwidth*numpy.tan(numpy.radians(0.5))
            pstr += 'corresponds to channel per degree: %8.2f\n' %(chpdeg)
        if self._area_init:
            pstr += '\n area detector initialized:\n'
            pstr += 'area detector mount directions: %s/%s\n' %(self._area_detdir1,self._area_detdir2)
            pstr += 'number of channels/center channels: (%d,%d) / (%d,%d)\n' %(self._area_Nch1,self._area_Nch2,self._area_cch1,self._area_cch2)
            pstr += 'distance to center of rotation/pixel width: %10.4g/ (%10.4g,%10.4g) \n' %(self._area_distance,self._area_pwidth1,self._area_pwidth2)
            chpdeg1 = 2*self._area_distance/self._area_pwidth1*numpy.tan(numpy.radians(0.5))
            chpdeg2 = 2*self._area_distance/self._area_pwidth2*numpy.tan(numpy.radians(0.5))
            pstr += 'corresponds to channel per degree: (%8.2f,%8.2f)\n' %(chpdeg1,chpdeg2)

        return pstr

    def _checkInput(self,*args):
        """
        helper function to check that the arguments given to the QConversion
        routines have the correct shape. It determines the number of points in
        the input from the longest array/list given and checks that only inputs
        combatible with this length are given

        Parameters
        ----------
         *args:     arguments from the QConversion routine (sample and detector angles)
        
        Returns
        -------
         Npoints:   integer to tell the number of points given
        """
        np = 1
        for a in args:
            #optain size of input
            if isinstance(a,numpy.ndarray):
                anp = a.size
            elif isinstance(a,(list,tuple)):
                anp = len(a)
            elif isinstance(a,numbers.Number):
                anp = 1
            else:
                raise TypeError('QConversion: Input argument #%d has an invalid type.'%args.index(a))
            # check if the input field is valid
            if anp>1 and np==1:
                np=anp
            elif anp>1 and np!=anp:
                raise InputError('QConversion: Several input-arrays arguments with different shape are an invalid input!')

        return np

    def _reshapeInput(self,npoints,delta,*args):
        """
        helper function to reshape the input of arguments to (len(args),npoints)
        The input arguments must be either scalars or are of length npoints.

        Parameters
        ----------
         npoints:   length of the input arrays
         delta:     value to substract from the input arguments as array with len(args)
         *args:     input arrays and scalars

        Returns
        -------
         inarr:     numpy.ndarray of shape (len(args),npoints) with the input arguments
         retshape:  shape of return values 
        """

        inarr = numpy.empty((len(args),npoints),dtype=numpy.double)
        retshape = (npoints,) #default value

        for i in range(len(args)):
            arg = args[i]
            if not isinstance(arg,(numbers.Number,list,tuple,numpy.ndarray)):
                raise TypeError("QConversion: invalid type for one of the sample coordinates, must be scalar, list or array")
            elif isinstance(arg,numbers.Number):
                arg = numpy.ones(npoints,dtype=numpy.double)*arg
            elif isinstance(arg,(list,tuple)):
                arg = numpy.array(arg,dtype=numpy.double)
            else: # determine return value shape
                retshape = arg.shape
            arg = arg - delta[i]
            inarr[i,:] = numpy.ravel(arg)

        return inarr,retshape

    def __call__(self,*args,**kwargs):
        """
        wrapper function for point(...)
        """
        return self.point(*args,**kwargs)

    def point(self,*args,**kwargs):
        """
        angular to momentum space conversion for a point detector
        located in direction of self.r_i when detector angles are zero

        Parameters
        ----------
        *args:           sample and detector angles as numpy array, lists
                         or Scalars in total
                         len(self.sampleAxis)+len(detectorAxis) must be given,
                         always starting with the outer most circle.  all
                         arguments must have the same shape or length but can
                         be mixed with Scalars (i.e. if an angle is always the
                         same it can be given only once instead of an array)

             sAngles:    sample circle angles, number of arguments must
                         correspond to len(self.sampleAxis)
             dAngles:    detector circle angles, number of arguments must
                         correspond to len(self.detectorAxis)

         **kwargs:       optional keyword arguments
             delta:      giving delta angles to correct the given ones for
                         misalignment delta must be an numpy array or list
                         of len(*args)
                         used angles are than *args - delta
             UB:         matrix for conversion from (hkl) coordinates to Q of sample
                         used to determine not Q but (hkl)
                         (default: self.UB)
             wl:         x-ray wavelength in angstroem (default: self._wl)
             en:         x-ray energy in eV (default is converted self._wl)
                         both wavelength and energy can also be an array
                         which enables the QConversion for energy scans.
                         Note that the en keyword overrules the wl keyword!
             deg:        flag to tell if angles are passed as degree
                         (default: True)
             sampledis:  sample displacement vector in relative units of the detector 
                         distance (default: (0,0,0))

        Returns
        -------
        reciprocal space positions as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input
        """

        for k in kwargs.keys():
            if k not in ['wl','en','deg','UB','delta','sampledis']:
                raise Exception("unknown keyword argument given: allowed are 'delta': angle offsets, 'wl/en': x-ray wavelength/energy, 'UB': orientation/orthonormalization matrix, 'deg': flag to tell if angles are in degrees, 'sampledis': sample displacement vector")
        
        Ns = len(self.sampleAxis)
        Nd = len(self.detectorAxis)
        if self._area_detrotaxis_set:
            Nd-=1 # do not consider detector rotation for point detector
        Ncirc = Ns + Nd

        # kwargs
        if 'wl' in kwargs:
            wl = utilities.wavelength(kwargs['wl'])
        else:
            wl = self._wl
        if 'en' in kwargs:
            wl = utilities.lam2en(utilities.energy(kwargs['en']))

        if 'deg' in kwargs:
            deg = kwargs['deg']
        else:
            deg = True

        if 'delta' in kwargs:
            delta = numpy.array(kwargs['delta'],dtype=numpy.double)
            if delta.size != Ncirc:
                raise InputError("QConversion: keyword argument delta does not have an appropriate shape")
        else:
            delta = numpy.zeros(Ncirc)

        if 'UB' in kwargs:
            UB = numpy.array(kwargs['UB'])
        else:
            UB = self.UB
        
        if 'sampledis' in kwargs:
            sd = numpy.array(kwargs['sampledis'])
        else:
            sd = numpy.zeros(3) 

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments given, \
                             number of arguments should be %d" %(len(args),Ncirc))
        
        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles,retshape = self._reshapeInput(Npoints,delta[:Ns],*args[:Ns])
        dAngles = self._reshapeInput(Npoints,delta[Ns:],*args[Ns:])[0]
        wl = numpy.ravel(self._reshapeInput(Npoints,(0,),wl)[0])
        
        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        if deg:
            sAngles = numpy.radians(sAngles)
            dAngles = numpy.radians(dAngles)

        sAxis=self._sampleAxis_str
        dAxis=self._detectorAxis_str

        if self._area_detrotaxis_set:
            dAxis=self._detectorAxis_str[:-2] # do not consider detector rotation for point detector
        else:
            dAxis=self._detectorAxis_str

        if config.VERBOSITY >= config.DEBUG:
            print("XU.QConversion: Ns, Nd: %d %d" % (Ns,Nd))
            print("XU.QConversion: sAngles / dAngles %s / %s" %(str(sAngles),str(dAngles)))

        if numpy.any(sd):
            qpos = cxrayutilities.ang2q_conversion_sd(sAngles, dAngles, self.r_i,sAxis, dAxis, self._kappa_dir,
                                               UB, sd, wl, config.NTHREADS)
        else:
            qpos = cxrayutilities.ang2q_conversion(sAngles, dAngles, self.r_i,sAxis, dAxis, self._kappa_dir,
                                               UB, wl, config.NTHREADS)

        if Npoints==1:
            return ( qpos[0,0], qpos[0,1], qpos[0,2] )
        else:
            return numpy.reshape(qpos[:,0],retshape),\
                   numpy.reshape(qpos[:,1],retshape),\
                   numpy.reshape(qpos[:,2],retshape)

    def init_linear(self,detectorDir,cch,Nchannel,distance=None,pixelwidth=None,chpdeg=None,tilt=0,**kwargs):
        """
        initialization routine for linear detectors
        detector direction as well as distance and pixel size or
        channels per degree must be given.

        Parameters
        ----------
         detectorDir:     direction of the detector (along the pixel array); e.g. 'z+'
         cch:             center channel, in direction of self.r_i at zero detectorAngles
         Nchannel:        total number of detector channels
         distance:        distance of center channel from center of rotation
         pixelwidth:      width of one pixel (same unit as distance)
         chpdeg:          channels per degree (only absolute value is relevant) sign
                          determined through detectorDir

                          !! Either distance and pixelwidth or chpdeg must be given !!
         tilt:            tilt of the detector axis from the detectorDir (in degree)

            Note: the channel numbers run from 0 .. Nchannel-1
        
        **kwargs:        optional keyword arguments
          Nav:           number of channels to average to reduce data size (default: 1)
          roi:           region of interest for the detector pixels; e.g. [100,900]
        """

        for k in kwargs.keys():
            if k not in ['Nav','roi']:
                raise Exception("unknown keyword argument given: allowed are 'Nav': number of channels for block-average, 'roi': region of interest")
        
        # detectorDir
        if not isinstance(detectorDir,basestring) or len(detectorDir)!=2:
            raise InputError("QConversion: incorrect detector direction type or syntax (%s)" %repr(detectorDir))
        if not circleSyntax.search(detectorDir):
            raise InputError("QConversion: incorrect detector direction syntax (%s)" %detectorDir)
        self._linear_detdir = detectorDir

        self._linear_Nch = int(Nchannel)
        self._linear_cch = float(cch)
        self._linear_tilt = numpy.radians(tilt)

        if distance!=None and pixelwidth!=None:
            self._linear_distance = 1.0
            self._linear_pixwidth = float(pixelwidth)/float(distance)
        elif chpdeg!=None:
            self._linear_distance = 1.0
            self._linear_pixwidth = 2*self._linear_distance/numpy.abs(float(chpdeg))*numpy.tan(numpy.radians(0.5))
        else:
            # not all needed values were given
            raise InputError("QConversion: not all mandatory arguments were given -> read API doc, need distance and pixelwidth or chpdeg")


        # kwargs
        if 'roi' in kwargs:
            self._linear_roi = kwargs['roi']
        else:
            self._linear_roi = [0,self._linear_Nch]
        if 'Nav' in kwargs:
            self._linear_nav = kwargs['Nav']
        else:
            self._linear_nav = 1

        self._linear_init = True

    def linear(self,*args,**kwargs):
        """
        angular to momentum space conversion for a linear detector
        the cch of the detector must be in direction of self.r_i when
        detector angles are zero

        the detector geometry must be initialized by the init_linear(...) routine

        Parameters
        ----------
        *args:          sample and detector angles as numpy array, lists or
                        Scalars in total len(self.sampleAxis)+len(detectorAxis)
                        must be given, always starting with the outer most
                        circle.  all arguments must have the same shape or
                        length but can be mixed with Scalars (i.e. if an angle
                        is always the same it can be given only once instead of
                        an array)

            sAngles:    sample circle angles, number of arguments must correspond to
                        len(self.sampleAxis)
            dAngles:    detector circle angles, number of arguments must correspond to
                        len(self.detectorAxis)

        **kwargs:       possible keyword arguments
            delta:      giving delta angles to correct the given ones for misalignment
                        delta must be an numpy array or list of len(*args)
                        used angles are than *args - delta
            UB:         matrix for conversion from (hkl) coordinates to Q of sample
                        used to determine not Q but (hkl)
                        (default: self.UB)
            Nav:        number of channels to average to reduce data size (default: self._linear_nav)
            roi:        region of interest for the detector pixels; e.g. [100,900] (default: self._linear_roi)
            wl:         x-ray wavelength in angstroem (default: self._wl)
            en:         x-ray energy in eV (default is converted self._wl)
                        both wavelength and energy can also be an array
                        which enables the QConversion for energy scans.
                        Note that the en keyword overrules the wl keyword!
            deg:        flag to tell if angles are passed as degree (default: True)
            sampledis:  sample displacement vector in same units as the detector distance (default: (0,0,0))

        Returns
        -------
        reciprocal space position of all detector pixels in a numpy.ndarray of shape
        ( (*)*(self._linear_roi[1]-self._linear_roi[0]+1) , 3 )
        """

        if not self._linear_init:
            raise Exception("QConversion: linear detector not initialized -> call Ang2Q.init_linear(...)")
        
        for k in kwargs.keys():
            if k not in ['wl','en','deg','UB','delta','Nav','roi','sampledis']:
                raise Exception("unknown keyword argument given: allowed are 'delta': angle offsets, 'wl/en': x-ray wavelength/energy, 'UB': orientation/orthonormalization matrix, 'deg': flag to tell if angles are in degrees, 'Nav': number of channels for block-averaging, 'roi': region of interest, 'sampledis': sample displacement vector")

        Ns = len(self.sampleAxis)
        Nd = len(self.detectorAxis)
        Ncirc = Ns + Nd

        # kwargs
        if 'wl' in kwargs:
            wl = utilities.wavelength(kwargs['wl'])
        else:
            wl = self._wl
        if 'en' in kwargs:
            wl = utilities.lam2en(utilities.energy(kwargs['en']))

        if 'deg' in kwargs:
            deg = kwargs['deg']
        else:
            deg = True

        if 'Nav' in kwargs:
            nav = kwargs['Nav']
        else:
            nav = self._linear_nav

        if 'roi' in kwargs:
            oroi = kwargs['roi']
        else:
            oroi = self._linear_roi

        if 'delta' in kwargs:
            delta = numpy.array(kwargs['delta'],dtype=numpy.double)
            if delta.size != Ncirc:
                raise InputError("QConversion: keyword argument delta does not have an appropriate shape")
        else:
            delta = numpy.zeros(Ncirc)

        if 'UB' in kwargs:
            UB = numpy.array(kwargs['UB'])
        else:
            UB = self.UB

        if 'sampledis' in kwargs:
            sd = numpy.array(kwargs['sampledis'])
        else:
            sd = numpy.zeros(3) 
        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments given, \
                             number of arguments should be %d" %(len(args),Ncirc))

        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles,retshape = self._reshapeInput(Npoints,delta[:Ns],*args[:Ns])
        dAngles = self._reshapeInput(Npoints,delta[Ns:],*args[Ns:])[0]
        wl = numpy.ravel(self._reshapeInput(Npoints,(0,),wl)[0])
        
        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        if deg:
            sAngles = numpy.radians(sAngles)
            dAngles = numpy.radians(dAngles)

        # initialize psd geometry to for C subprogram (include Nav and roi possibility)
        cch = self._linear_cch/float(nav)
        pwidth = self._linear_pixwidth*nav
        #roi = numpy.ceil(numpy.array(roi)/float(nav)).astype(numpy.int32)
        roi = numpy.array(oroi)
        roi[0] = numpy.floor(oroi[0]/float(nav))
        roi[1] = numpy.ceil((oroi[1]-oroi[0])/float(nav)) + roi[0]
        roi = roi.astype(numpy.int32)

        sAxis=self._sampleAxis_str
        dAxis=self._detectorAxis_str

        if numpy.any(sd):
            qpos = cxrayutilities.ang2q_conversion_linear_sd(sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir,
                     cch, pwidth, roi, self._linear_detdir, self._linear_tilt, UB, sd, wl, config.NTHREADS)
        else:
            qpos = cxrayutilities.ang2q_conversion_linear(sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir,
                     cch, pwidth, roi, self._linear_detdir, self._linear_tilt, UB, wl, config.NTHREADS)


        #reshape output
        if Npoints==1:
            qpos.shape = (Npoints*(roi[1]-roi[0]),3)
            return qpos[:,0],qpos[:,1],qpos[:,2]
        else:
            qpos.shape = (Npoints,(roi[1]-roi[0]),3)
            return qpos[:,:,0],qpos[:,:,1],qpos[:,:,2]

    def init_area(self,detectorDir1,detectorDir2,cch1,cch2,Nch1,Nch2,distance=None,
                  pwidth1=None,pwidth2=None,chpdeg1=None,chpdeg2=None,detrot=0, tiltazimuth=0, tilt=0, **kwargs):
        """
        initialization routine for area detectors
        detector direction as well as distance and pixel size or
        channels per degree must be given. Two separate pixel sizes and
        channels per degree for the two orthogonal directions can be given

        Parameters
        ----------
         detectorDir1:    direction of the detector (along the pixel direction 1);
                          e.g. 'z+' means higher pixel numbers at larger z positions
         detectorDir2:    direction of the detector (along the pixel direction 2); e.g. 'x+'
         cch1,2:          center pixel, in direction of self.r_i at zero detectorAngles
         Nch1:            number of detector pixels along direction 1
         Nch2:            number of detector pixels along direction 2
         distance:        distance of center pixel from center of rotation
         pwidth1,2:       width of one pixel (same unit as distance)
         chpdeg1,2:       channels per degree (only absolute value is relevant) sign
                          determined through detectorDir1,2
         detrot:          angle of the detector rotation around primary beam direction (used to correct misalignments)
         tiltazimuth:     direction of the tilt vector in the detector plane (in degree)
         tilt:            tilt of the detector plane around an axis normal to the direction
                          given by the tiltazimuth

           Note: Either distance and pwidth1,2 or chpdeg1,2 must be given !!

           Note: the channel numbers run from 0 .. NchX-1
       
        **kwargs:         optional keyword arguments
          Nav:            number of channels to average to reduce data size (default: [1,1])
          roi:            region of interest for the detector pixels; e.g. [100,900,200,800]
        """

        for k in kwargs.keys():
            if k not in ['Nav','roi']:
                raise Exception("unknown keyword argument given: allowed are 'Nav': number of channels for block-average, 'roi': region of interest")
        
        # detectorDir
        if not isinstance(detectorDir1,basestring) or len(detectorDir1)!=2:
            raise InputError("QConversion: incorrect detector direction1 type or syntax (%s)" %repr(detectorDir1))
        if not circleSyntax.search(detectorDir1):
            raise InputError("QConversion: incorrect detector direction1 syntax (%s)" %detectorDir1)
        self._area_detdir1 = detectorDir1
        if not isinstance(detectorDir2,basestring) or len(detectorDir2)!=2:
            raise InputError("QConversion: incorrect detector direction2 type or syntax (%s)" %repr(detectorDir2))
        if not circleSyntax.search(detectorDir2):
            raise InputError("QConversion: incorrect detector direction2 syntax (%s)" %detectorDir2)
        self._area_detdir2 = detectorDir2

        # other none keyword arguments
        self._area_Nch1 = int(Nch1)
        self._area_Nch2 = int(Nch2)
        self._area_cch1 = int(cch1)
        self._area_cch2 = int(cch2)

        # if detector rotation is present add new motor to consider it in conversion
        self._area_detrot = numpy.radians(detrot)
        if self._area_detrot !=0.:
            if self._area_detrotaxis_set:
                self._set_detectorAxis(self._get_detectorAxis()[:-1] + [math.getSyntax(self.r_i)],detrot=True)
            else:
                self._set_detectorAxis(self._get_detectorAxis() + [math.getSyntax(self.r_i)],detrot=True)

        self._area_tiltazimuth = numpy.radians(tiltazimuth)
        self._area_tilt = numpy.radians(tilt)

        # mandatory keyword arguments
        if distance!=None and pwidth1!=None and pwidth2!=None:
            self._area_distance = float(distance)
            self._area_pwidth1 = float(pwidth1)
            self._area_pwidth2 = float(pwidth2)
        elif chpdeg1!=None and chpdeg2!=None:
            self._area_distance = 1.0
            self._area_pwidth1 = 2*self._area_distance/numpy.abs(float(chpdeg1))*numpy.tan(numpy.radians(0.5))
            self._area_pwidth2 = 2*self._area_distance/numpy.abs(float(chpdeg2))*numpy.tan(numpy.radians(0.5))
        else:
            # not all needed values were given
            raise InputError("Qconversion error: not all mandatory arguments were given -> read API doc")

        # kwargs
        if 'roi' in kwargs:
            self._area_roi = kwargs['roi']
        else:
            self._area_roi = [0,self._area_Nch1,0,self._area_Nch2]
        if 'Nav' in kwargs:
            self._area_nav = kwargs['Nav']
        else:
            self._area_nav = [1,1]

        self._area_init = True

    def area(self,*args,**kwargs):
        """
        angular to momentum space conversion for a area detector
        the center pixel defined by the init_area routine must be
        in direction of self.r_i when detector angles are zero

        the detector geometry must be initialized by the init_area(...) routine

        Parameters
        ----------
        *args:          sample and detector angles as numpy array, lists or
                        Scalars in total len(self.sampleAxis)+len(detectorAxis)
                        must be given, always starting with the outer most
                        circle.  all arguments must have the same shape or
                        length but can be mixed with Scalars (i.e. if an angle
                        is always the same it can be given only once instead of
                        an array)

            sAngles:    sample circle angles, number of arguments must correspond to
                        len(self.sampleAxis)
            dAngles:    detector circle angles, number of arguments must correspond to
                        len(self.detectorAxis)

        **kwargs:       possible keyword arguments
            delta:      giving delta angles to correct the given ones for misalignment
                        delta must be an numpy array or list of len(*args)
                        used angles are than *args - delta
            UB:         matrix for conversion from (hkl) coordinates to Q of sample
                        used to determine not Q but (hkl)
                        (default: self.UB)
            roi:        region of interest for the detector pixels; e.g. [100,900,200,800]
                        (default: self._area_roi)
            Nav:        number of channels to average to reduce data size e.g. [2,2]
                        (default: self._area_nav)
            wl:         x-ray wavelength in angstroem (default: self._wl)
            en:         x-ray energy in eV (default is converted self._wl)
                        both wavelength and energy can also be an array
                        which enables the QConversion for energy scans.
                        Note that the en keyword overrules the wl keyword!
            deg:        flag to tell if angles are passed as degree (default: True)
            sampledis:  sample displacement vector in same units as the detector distance (default: (0,0,0))

        Returns
        -------
        reciprocal space position of all detector pixels in a numpy.ndarray of shape
        ( (*)*(self._area_roi[1]-self._area_roi[0]+1)*(self._area_roi[3]-self._area_roi[2]+1) , 3 )
        were detectorDir1 is the fastest varing
        """

        if not self._area_init:
            raise Exception("QConversion: area detector not initialized -> call Ang2Q.init_area(...)")
        
        for k in kwargs.keys():
            if k not in ['wl','en','deg','UB','delta','Nav','roi','sampledis']:
                raise Exception("unknown keyword argument given: allowed are 'delta': angle offsets, 'wl/en': x-ray wavelength/energy, 'UB': orientation/orthonormalization matrix, 'deg': flag to tell if angles are in degrees, 'Nav': number of channels for block-averaging, 'roi': region of interest, 'sampledis': sample displacement vector")

        Ns = len(self.sampleAxis)
        Nd = len(self.detectorAxis)
        if self._area_detrotaxis_set:
            Nd = Nd-1
        Ncirc = Ns + Nd

        # kwargs
        if 'wl' in kwargs:
            wl = utilities.wavelength(kwargs['wl'])
        else:
            wl = self._wl
        if 'en' in kwargs:
            wl = utilities.lam2en(utilities.energy(kwargs['en']))

        if 'deg' in kwargs:
            deg = kwargs['deg']
        else:
            deg = True

        if 'roi' in kwargs:
            oroi = kwargs['roi']
        else:
            oroi = self._area_roi

        if 'Nav' in kwargs:
            nav = kwargs['Nav']
        else:
            nav = self._area_nav

        if 'delta' in kwargs:
            delta = numpy.array(kwargs['delta'],dtype=numpy.double)
            if delta.size != Ncirc:
                raise InputError("QConversion: keyword argument delta does not have an appropriate shape")
        else:
            delta = numpy.zeros(Ncirc)

        if 'UB' in kwargs:
            UB = numpy.array(kwargs['UB'])
        else:
            UB = self.UB

        if 'sampledis' in kwargs:
            sd = numpy.array(kwargs['sampledis'])
        else:
            sd = numpy.zeros(3) 
        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Ncirc:
            raise InputError("QConversion: wrong amount (%d) of arguments given, \
                             number of arguments should be %d" %(len(args),Ncirc))

        # determine the number of points
        a = args + (wl,)
        Npoints = self._checkInput(*a)

        # reshape/recast input arguments for sample and detector angles
        sAngles,retshape = self._reshapeInput(Npoints,delta[:Ns],*args[:Ns])
        wl = numpy.ravel(self._reshapeInput(Npoints,(0,),wl)[0])
        
        if self._area_detrotaxis_set:
            Nd = Nd + 1
            if deg:
                a = args[Ns:] + (numpy.degrees(self._area_detrot),)
                dAngles = self._reshapeInput(Npoints,numpy.append(delta[Ns:],0),*a)[0]
            else:
                a = args[Ns:] + (self._area_detrot,)
                dAngles = self._reshapeInput(Npoints,numpy.append(delta[Ns:],0),*a)[0]
        else:
            dAngles = self._reshapeInput(Npoints,delta[Ns:],*args[Ns:])[0]

        sAngles = sAngles.transpose()
        dAngles = dAngles.transpose()

        if deg:
            sAngles = numpy.radians(sAngles)
            dAngles = numpy.radians(dAngles)

        # initialize ccd geometry to for C subroutine (include Nav and roi possibility)
        cch1 = self._area_cch1/float(nav[0])
        cch2 = self._area_cch2/float(nav[1])
        pwidth1 = self._area_pwidth1*nav[0]/self._area_distance
        pwidth2 = self._area_pwidth2*nav[1]/self._area_distance
        roi = numpy.array(oroi)
        roi[0] = numpy.floor(oroi[0]/float(nav[0]))
        roi[1] = numpy.ceil((oroi[1]-oroi[0])/float(nav[0])) + roi[0]
        roi[2] = numpy.floor(oroi[2]/float(nav[1]))
        roi[3] = numpy.ceil((oroi[3]-oroi[2])/float(nav[1])) + roi[2]
        roi = roi.astype(numpy.int32)
        if config.VERBOSITY >= config.DEBUG:
            print("QConversion.area: roi, number of points per frame: %s, %d" %(str(roi),(roi[1]-roi[0])*(roi[3]-roi[2])))
            print("QConversion.area: cch1,cch2: %5.2f %5.2f" %(cch1,cch2))

        sAxis=self._sampleAxis_str
        dAxis=self._detectorAxis_str

        if numpy.any(sd):
            qpos = cxrayutilities.ang2q_conversion_area_sd(sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir, 
                     cch1, cch2, pwidth1, pwidth2, roi, self._area_detdir1, self._area_detdir2,
                     self._area_tiltazimuth, self._area_tilt, UB, sd, wl, config.NTHREADS)
        else:
            qpos = cxrayutilities.ang2q_conversion_area(sAngles, dAngles, self.r_i, sAxis, dAxis, self._kappa_dir, 
                     cch1, cch2, pwidth1, pwidth2, roi, self._area_detdir1, self._area_detdir2,
                     self._area_tiltazimuth, self._area_tilt, UB, wl, config.NTHREADS)

        #reshape output
        if Npoints==1:
            qpos.shape = ((roi[1]-roi[0]),(roi[3]-roi[2]),3)
            return qpos[:,:,0],qpos[:,:,1],qpos[:,:,2]
        else:
            qpos.shape = (Npoints,(roi[1]-roi[0]),(roi[3]-roi[2]),3)
            return qpos[:,:,:,0],qpos[:,:,:,1],qpos[:,:,:,2]


class Experiment(object):
    """
    base class for describing experiments
    users should use the derived classes: HXRD, GID, Powder
    """
    def __init__(self,ipdir,ndir,**keyargs):
        """
        initialization of an Experiment class needs the sample orientation
        given by the samples surface normal and an second not colinear direction
        specifying the inplane reference direction.

        Parameters
        ----------
         ipdir:      inplane reference direction (ipdir points into the primary beam
                     direction at zero angles)
         ndir:       surface normal of your sample (ndir points in a direction perpendicular
                     to the primary beam and the innermost detector rotation axis)

        keyargs:     optional keyword arguments
          qconv:     QConversion object to use for the Ang2Q conversion 
          wl:        wavelength of the x-rays in Angstroem (default: 1.5406A)
          en:        energy of the x-rays in eV (default: 8048eV == 1.5406A )
                     the en keyword overrules the wl keyword

        Note:
         The qconv argument does not change the Q2Ang function's behavior. See Q2AngFit
         function in case you want to calculate for arbitrary goniometers which some restrictions.
        """

        for k in keyargs.keys():
            if k not in ['qconv','wl','en']:
                raise Exception("unknown keyword argument given: allowed are 'en': for x-ray energy, 'wl': x-ray wavelength, 'qconv': reciprocal space conversion.")

        if isinstance(ipdir,(list,tuple)):
            self.idir = math.VecUnit(numpy.array(ipdir,dtype=numpy.double))
        elif isinstance(ipdir,numpy.ndarray):
            self.idir = math.VecUnit(ipdir)
        else:
            raise TypeError("Inplane direction must be list or numpy array")

        if isinstance(ndir,(list,tuple)):
            self.ndir = math.VecUnit(numpy.array(ndir,dtype=numpy.double))
        elif isinstance(ndir,numpy.ndarray):
            self.ndir = math.VecUnit(ndir)
        else:
            raise TypeError("normal direction must be list or numpy array")

        #test the given direction to be not parallel and warn if not perpendicular
        if(norm(numpy.cross(self.idir,self.ndir))<config.EPSILON):
            raise InputError("given inplane direction is parallel to normal direction, they must be linear independent!")
        if(numpy.abs(numpy.dot(self.idir,self.ndir))>config.EPSILON):
            self.idir = numpy.cross(numpy.cross(self.ndir,self.idir),self.ndir)
            self.idir = self.idir/norm(self.idir)
            warnings.warn("Experiment: given inplane direction is not perpendicular to normal direction\n -> Experiment class uses the following direction with the same azimuth:\n %s" %(' '.join(map(str,numpy.round(self.idir,3)))))

        # initialize Ang2Q conversion
        if "qconv" in keyargs:
            self._A2QConversion = keyargs["qconv"]
        else:
            self._A2QConversion = QConversion('x+','x+',[0,1,0]) # 1S+1D goniometer
        self.Ang2Q = self._A2QConversion

        #set the coordinate transform for the azimuth used in the experiment
        self.scatplane = math.VecUnit(numpy.cross(self.idir,self.ndir))
        self._set_transform( self.scatplane, self.idir, self.ndir)

        #calculate the energy from the wavelength
        if "wl" in keyargs:
            self._set_wavelength(keyargs["wl"])
        else:
            self._set_wavelength(config.WAVELENGTH)

        if "en" in keyargs:
            self._set_energy(keyargs["en"])

    def __str__(self):

        ostr = "scattering plane normal: (%f %f %f)\n" %(self.scatplane[0],
                                                 self.scatplane[1],
                                                 self.scatplane[2])
        ostr += "inplane azimuth: (%f %f %f)\n" %(self.idir[0],
                                                 self.idir[1],
                                                 self.idir[2])
        ostr += "second refercence direction: (%f %f %f)\n" %(self.ndir[0],
                                                 self.ndir[1],
                                                 self.ndir[2])
        ostr += "energy: %f (eV)\n" %self._en
        ostr += "wavelength: %f (Anstrom)\n" %(self._wl)
        ostr += self._A2QConversion.__str__()

        return ostr

    def _set_transform(self,v1,v2,v3):
        """
        set new transformation of the coordinate system to use in the experimental class
        """
        self._t1 = math.CoordinateTransform(v1,v2,v3) # turn idir to Y and ndir to Z

        yi = self._A2QConversion.r_i
        idc = self._A2QConversion.detectorAxis[-1]
        xi = math.getVector(idc)
        if numpy.linalg.norm(numpy.cross(xi,yi)) < config.EPSILON:
            # this is the case when a detector rotation around the primary beam dir. is installed
            idc = self._A2QConversion.detectorAxis[-2]
            xi = math.getVector(idc)
        zi = math.VecUnit(numpy.cross(xi,yi))
        # turn r_i to Y and Z defined by detector rotation plane
        self._t2 = math.CoordinateTransform(xi,yi,zi)

        self._transform = math.Transform(numpy.dot(self._t2.imatrix,self._t1.matrix))

    def _set_energy(self,energy):
        self._en = utilities.energy(energy)
        self._wl = utilities.en2lam(self._en)
        self.k0 = numpy.pi*2./self._wl
        self._A2QConversion.wavelength = self._wl

    def _set_wavelength(self,wl):
        self._wl = utilities.wavelength(wl)
        self._en = utilities.lam2en(self._wl)
        self.k0 = numpy.pi*2./self._wl
        self._A2QConversion.wavelength = self._wl

    def _get_energy(self):
        return self._en

    def _get_wavelength(self):
        return self._wl

    energy = property(_get_energy,_set_energy)
    wavelength = property(_get_wavelength,_set_wavelength)

    def _set_inplane_direction(self,dir):
        if isinstance(dir,list):
            self.idir = numpy.array(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.idir = dir
        else:
            raise TypeError("Inplane direction must be list or numpy array")

        v1 = numpy.cross(self.ndir,self.idir)
        self._set_transform(v1,self.idir,self.ndir)

    def _get_inplane_direction(self):
        return self.idir

    def _set_normal_direction(self,dir):
        if isinstance(dir,list):
            self.ndir = numpy.array(dir,dtype=numpy.double)
        elif isinstance(dir,numpy.ndarray):
            self.ndir = dir
        else:
            raise TypeError("Surface normal must be list or numpy array")

        v1 = numpy.cross(self.ndir,self.idir)
        self._set_transform(v1,self.idir,self.ndir)

    def _get_normal_direction(self):
        return self.ndir

    def Q2Ang(self,qvec):
        pass

    def Ang2HKL(self,*args,**kwargs):
        """
        angular to (h,k,l) space conversion.
        It will set the UB argument to Ang2Q and pass all other parameters unchanged.
        See Ang2Q for description of the rest of the arguments.

        Parameters
        ----------

        **kwargs:   optional keyword arguments
            B:      reciprocal space conversion matrix of a Material.
                    you can specify the matrix B (default identiy matrix)
                    shape needs to be (3,3)
            mat:    Material object to use to obtain a B matrix
                    (e.g. xu.materials.Si)
                    can be used as alternative to the B keyword argument
                    B is favored in case both are given
            U:      orientation matrix U can be given
                    if none is given the orientation defined in the Experiment class
                    is used.
            dettype:detector type: one of ('point', 'linear', 'area')
                    decides which routine of Ang2Q to call
                    default 'point'
            delta:  giving delta angles to correct the given ones for misalignment
                    delta must be an numpy array or list of length 2.
                    used angles are than om,tt - delta
            wl:     x-ray wavelength in angstroem (default: self._wl)
            en:     x-ray energy in eV (default: converted self._wl)
            deg:    flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        H K L coordinates as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input (*args)

        """

        for k in kwargs.keys():
            if k not in ['U','B','mat','dettype','delta','wl','en','deg']:
                raise Exception("unknown keyword argument given: allowed are 'B': orthonormalization matrix, 'U': orientation matrix, 'mat': material object, 'dettype': string with detector type, and 'delta,wl,en,deg' from Ang2Q")
        
        if "B" in kwargs:
            B = numpy.array(kwargs['B'])
            kwargs.pop("B")
        elif "mat" in kwargs:
            mat = kwargs['mat']
            B = mat.B
            kwargs.pop("mat")
        else:
            B = numpy.identity(3)

        if "U" in kwargs:
            U = numpy.array(kwargs['U'])
            kwargs.pop("U")
        else:
            U = self._transform.matrix

        kwargs['UB'] = numpy.dot(U,B)

        if "dettype" in kwargs:
            typ = kwargs['dettype']
            if typ not in ('point', 'linear', 'area'):
                raise InputError("wrong dettype given: needs to be one of 'point', 'linear', 'area'")
            kwargs.pop("dettype")
        else:
            typ = 'point'

        if typ=='linear':
            return self.Ang2Q.linear(*args,**kwargs)
        elif typ=='area':
            return self.Ang2Q.area(*args,**kwargs)
        else:
            return self.Ang2Q(*args,**kwargs)

    def Transform(self,v):
        """
        transforms a vector, matrix or tensor of rank 4 (e.g. elasticity tensor)
        to the coordinate frame of the Experiment class. This is for example 
        necessary before any Q2Ang-conversion can be performed.

        Parameters
        ----------
         v:     object to transform, list or numpy array of shape
                    (n,) (n,n), (n,n,n,n) where n is the rank of the
                    transformation matrix

        Returns
        -------
         transformed object of the same shape as v
        """
        return self._transform(v)

    def TiltAngle(self,q,deg=True):
        """
        TiltAngle(q,deg=True):
        Return the angle between a q-space position and the surface normal.

        Parameters
        ----------
         q:          list or numpy array with the reciprocal space position

        optional keyword arguments:
         deg:        True/False whether the return value should be in degree or radians
                     (default: True)
        """

        if isinstance(q,list):
            qt = numpy.array(q,dtype=numpy.double)
        elif isinstance(q,numpy.ndarray):
            qt = q
        else:
            raise TypeError("q-space position must be list or numpy array")

        return math.VecAngle(self.ndir,qt,deg)

class HXRD(Experiment):
    """
    class describing high angle x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as helps with analyzing measured data

    the class describes a two circle (omega,twotheta) goniometer to
    help with coplanar x-ray diffraction experiments. Nevertheless 3D data
    can be treated with the use of linear and area detectors.
    see help self.Ang2Q
    """
    def __init__(self,idir,ndir,geometry='hi_lo',**keyargs):
        """
        initialization routine for the HXRD Experiment class

        Parameters
        ----------
        same as for the Experiment base class -> please look at the docstring of
        Experiment.__init__ for more details
        +
        keyargs         additional optional keyword argument
          geometry:     determines the scattering geometry:
                        "hi_lo" (default) high incidence-low exit
                        "lo_hi" low incidence - high exit
                        "real" general geometry - q-coordinates determine
                               high or low incidence
        """
        Experiment.__init__(self,idir,ndir,**keyargs)

        if geometry in ["hi_lo","lo_hi","real"]:
            self.geometry = geometry
        else:
            raise InputError("HXRD: invalid value for the geometry argument given")

        # initialize Ang2Q conversion
        if "qconv" not in keyargs:
            self._A2QConversion = QConversion('x+','x+',[0,1,0],wl=self._wl) # 1S+1D goniometer
            self.Ang2Q = self._A2QConversion

        if config.VERBOSITY >= config.DEBUG:
            print("XU.HXRD.__init__: \nEnergy: %s \nGeometry: %s \n%s---" %(self._en,self.geometry,str(self.Ang2Q)))

    def Ang2Q(self,om,tt,**kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help HXRD.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
         om,tt:      sample and detector angles as numpy array, lists or Scalars
                     must be given. all arguments must have the same shape or
                     length

        **kwargs:    optional keyword arguments
             delta:  giving delta angles to correct the given ones for misalignment
                     delta must be an numpy array or list of length 2.
                     used angles are than om,tt - delta
             UB:     matrix for conversion from (hkl) coordinates to Q of sample
                     used to determine not Q but (hkl)
                     (default: identity matrix)
             wl:     x-ray wavelength in angstroem (default: self._wl)
             deg:    flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        reciprocal space positions as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input

        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine
        pass

    def Q2Ang(self,*Q,**keyargs):
        """
        Convert a reciprocal space vector Q to COPLANAR scattering angles.
        The keyword argument trans determines whether Q should be transformed
        to the experimental coordinate frame or not. The coplanar scattering angles
        correspond to a goniometer with sample rotations ['x+','y+','z-']
        and detector rotation 'x+' and primary beam along y. This is a standard 
        four circle diffractometer.

        Parameters
        ----------
         Q:          a list, tuple or numpy array of shape (3) with
                     q-space vector components
                     or 3 separate lists with qx,qy,qz

        optional keyword arguments:
         trans:      True/False apply coordinate transformation on Q (default True)
         deg:        True/Flase (default True) determines if the
                     angles are returned in radians or degrees
         geometry:   determines the scattering geometry:

                     - "hi_lo" high incidence and low exit
                     - "lo_hi" low incidence and high exit
                     - "real" general geometry with angles determined by q-coordinates (azimuth); this and upper geometries return [omega,0,phi,twotheta]
 				     - "realTilt" general geometry with angles determined by q-coordinates (tilt); returns [omega,chi,phi,twotheta]

                     default: self.geometry

         refrac:     boolean to determine if refraction is taken into account
                     default: False
                     if True then also a material must be given
         mat:        Material object; needed to obtain its optical properties for
                     refraction correction, otherwise not used
         full_output:boolean to determine if additional output is given to determine
                     scattering angles more accurately in case refraction is set to True
                     default: False
         fi,fd:      if refraction correction is applied one can optionally specify
                     the facet through which the beam enters (fi) and exits (fd)
                     fi, fd must be the surface normal vectors (not transformed &
                     not necessarily normalized). If omitted the normal direction
                     of the experiment is used.

        Returns
        -------
        a numpy array of shape (4) with four scattering angles which are
        [omega,chi,phi,twotheta]

         omega:      incidence angle with respect to surface
         chi:        sample tilt for the case of non-coplanar geometry
         phi:        sample azimuth with respect to inplane reference direction
         twotheta:   scattering angle/detector angle

        if full_output:
        a numpy array of shape (6) with five angles which are
        [omega,chi,phi,twotheta,psi_i,psi_d]

         psi_i: offset of the incidence beam from the scattering plane due to refraction
         pdi_d: offset ot the diffracted beam from the scattering plane due to refraction
        """

        for k in keyargs.keys():
            if k not in ['trans','deg','geometry','refrac','mat','fi','fd','full_output']:
                raise Exception("unknown keyword argument given: see documentation for details")
        
        # collect the q-space input
        if len(Q)<3:
            Q = Q[0]
            if len(Q)<3:
                raise InputError("need 3 q-space vector components")

        if isinstance(Q,(list,tuple)):
            q = numpy.array(Q,dtype=numpy.double)
        elif isinstance(Q,numpy.ndarray):
            q = Q
        else:
            raise TypeError("Q vector must be a list, tuple or numpy array")

        # reshape input to have the same q array for all possible
        # types of different input
        if len(q.shape) != 2:
            q = q.reshape(3,1)

        # parse keyword arguments
        if 'geometry' in keyargs:
            if keyargs['geometry'] in ["hi_lo","lo_hi","real", "realTilt"]:
                geom = keyargs['geometry']
            else:
                raise InputError("HXRD: invalid value for the geometry argument given\n valid entries are: hi_lo,lo_hi,real,realTilt")
        else:
            geom = self.geometry

        if 'trans' in keyargs:
            trans = keyargs['trans']
        else:
            trans = True

        if 'deg' in keyargs:
            deg = keyargs['deg']
        else:
            deg = True

        if 'mat' in keyargs: # material for optical properties
            mat = keyargs['mat']
        else:
            mat = None

        if 'refrac' in keyargs:
            refrac = keyargs['refrac']
            if refrac: # check if material is available
                if mat==None: raise InputError("keyword argument 'mat' must be set when 'refrac' is set to True!")
        else:
            refrac = False

        if 'full_output' in keyargs:
            foutp = keyargs['full_output']
        else:
            foutp = False

        if 'fi' in keyargs: # incidence facet
            fi = keyargs['fi']
        else:
            fi = self.ndir
        fi = math.VecUnit(self.Transform(fi))

        if 'fd' in keyargs: # exit facet
            fd = keyargs['fd']
        else:
            fd = self.ndir
        fd = math.VecUnit(self.Transform(fd))

        # set parameters for the calculation
        z = self.Transform(self.ndir) # z
        y = self.Transform(self.idir) # y
        x = self.Transform(self.scatplane) # x
        if refrac:
            n = numpy.real(mat.idx_refraction(self.energy)) # index of refraction
            k = self.k0*n
        else: k = self.k0

        # start calculation for each given Q-point
        if foutp:
            angle = numpy.zeros((6,q.shape[1]))
        else:
            angle= numpy.zeros((4,q.shape[1]))
        for i in range(q.shape[1]):
            qvec = q[:,i]

            if trans:
                qvec = self.Transform(qvec)

            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.HXRD.Q2Ang: qvec= %s" %repr(qvec))

            qa = math.VecNorm(qvec)
            tth = 2.*numpy.arcsin(qa/2./k)

            #calculation of the sample azimuth phi (scattering plane spanned by qvec[1] and qvec[2] directions)

            chi = -numpy.arctan2(math.VecDot(x,qvec),math.VecDot(z,qvec))
            if numpy.abs(chi - numpy.pi/2.) < config.EPSILON:
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.HXRD: Given peak is perpendicular to ndir-reference direction (might be inplane/unreachable)")

            if geom == 'hi_lo':
                om = tth/2. + math.VecAngle(z,qvec) # +: high incidence geometry
                phi = -numpy.arctan2(math.VecDot(x,qvec),math.VecDot(y,qvec))
            elif geom == 'lo_hi':
                om = tth/2. - math.VecAngle(z,qvec) # -: low incidence geometry
                phi = -numpy.arctan2(-1*math.VecDot(x,qvec),-1*math.VecDot(y,qvec))
            elif geom == 'real':
                phi = -numpy.arctan2(math.VecDot(x,qvec),math.VecDot(y,qvec))
                if numpy.abs(phi)<=numpy.pi/2.:
                    sign = +1.
                    phi = -numpy.arctan2(math.VecDot(x,qvec),math.VecDot(y,qvec))
                else:
                    sign = -1.
                    phi = -numpy.arctan2(-1*math.VecDot(x,qvec),-1*math.VecDot(y,qvec))
                om = tth/2 + sign * math.VecAngle(z,qvec)
            elif geom == 'realTilt':
                phi = 0.
                om = tth/2 + numpy.arctan2(math.VecDot(y,qvec),numpy.sqrt(math.VecDot(z,qvec)**2+math.VecDot(x,qvec)**2))

            # refraction correction at incidence and exit facet
            psi_i = 0.
            psi_d = 0. # needed if refrac is false and full_output is True
            if refrac:
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.HXRD.Q2Ang: considering refraction correction")

                beta = tth - om

                ki = k * numpy.array([0.,numpy.cos(om),-numpy.sin(om)],dtype=numpy.double)
                kd = k * numpy.array([0.,numpy.cos(beta),numpy.sin(beta)],dtype=numpy.double)

                # refraction at incidence facet
                cosbi = numpy.abs(numpy.dot(fi,ki)/norm(ki))
                cosb0 = numpy.sqrt(1-n**2*(1-cosbi**2))

                ki0 = self.k0 * ( n*ki/norm(ki) - numpy.sign(numpy.dot(fi,ki)) * (n*cosbi-cosb0)*fi )

                # refraction at exit facet
                cosbd = numpy.abs(numpy.dot(fd,kd)/norm(kd))
                cosb0 = numpy.sqrt(1-n**2*(1-cosbd**2))

                kd0 = self.k0 * ( n*kd/norm(kd) - numpy.sign(numpy.dot(fd,kd)) * (n*cosbd-cosb0)*fd )

                # observed goniometer angles

                om = math.VecAngle(y,ki0)
                tth = math.VecAngle(ki0,kd0)
                psi_i = numpy.arcsin(ki0[0]/self.k0)
                psi_d = numpy.arcsin(kd0[0]/self.k0)

            if geom == 'realTilt':
                angle[0,i] = om
                angle[1,i] = chi
                angle[3,i] = tth
            else:
                angle[0,i] = om
                angle[2,i] = phi
                angle[3,i] = tth
            if foutp:
                angle[4,i] = psi_i
                angle[5,i] = psi_d

        if q.shape[1]==1:
            angle = angle.flatten()
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.HXRD.Q2Ang: om,chi,phi,tth,[psi_i,psi_d] = %s" %repr(angle))

        if deg:
            return numpy.degrees(angle)
        else:
            return angle

class NonCOP(Experiment):
    """
    class describing high angle x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as helps with analyzing measured data for NON-COPLANAR measurements,
    where the tilt is used to align asymmetric peaks, like in the case of a polefigure
    measurement.

    the class describes a four circle (omega,twotheta) goniometer to
    help with x-ray diffraction experiments. Linear and area detectors can be treated as
    described in "help self.Ang2Q"
    """
    def __init__(self,idir,ndir,**keyargs):
        """
        initialization routine for the NonCOP Experiment class

        Parameters
        ----------
        same as for the Experiment base class

        """
        Experiment.__init__(self,idir,ndir,**keyargs)

        # initialize Ang2Q conversion
        if "qconv" not in keyargs:
            self._A2QConversion = QConversion(['x+','y+','z-'],'x+',[0,1,0],wl=self._wl) # 3S+1D goniometer (standard four-circle goniometer, omega,chi,phi,theta)
            self.Ang2Q = self._A2QConversion

    def Ang2Q(self,om,chi,phi,tt,**kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help NonCOP.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
         om,chi,phi,tt: sample and detector angles as numpy array, lists or Scalars
                        must be given. all arguments must have the same shape or
                        length

        **kwargs:   optional keyword arguments
            delta:  giving delta angles to correct the given ones for misalignment
                    delta must be an numpy array or list of length 4.
                    used angles are than om,chi,phi,tt - delta
            UB:     matrix for conversion from (hkl) coordinates to Q of sample
                    used to determine not Q but (hkl)
                    (default: identity matrix)
            wl:     x-ray wavelength in angstroem (default: self._wl)
            deg:    flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        reciprocal space positions as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input

        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine
        pass

    def Q2Ang(self,*Q,**keyargs):
        """
        Convert a reciprocal space vector Q to NON-COPLANAR scattering angles.
        The keyword argument trans determines whether Q should be transformed
        to the experimental coordinate frame or not.

        Parameters
        ----------
         Q:          a list, tuple or numpy array of shape (3) with
                     q-space vector components
                     or 3 separate lists with qx,qy,qz

        optional keyword arguments:
         trans:      True/False apply coordinate transformation on Q (default True)
         deg:        True/Flase (default True) determines if the
                     angles are returned in radians or degree

        Returns
        -------
        a numpy array of shape (4) with four scattering angles which are
        [omega,chi,phi,twotheta]

         omega:      sample rocking angle
         chi:        sample tilt
         phi:        sample azimuth
         twotheta:   scattering angle (detector)
        """

        for k in keyargs.keys():
            if k not in ['trans','deg']:
                raise Exception("unknown keyword argument given: allowed are 'trans': coordinate transformation flag, 'deg': degree-flag")
        
        # collect the q-space input
        if len(Q)<3:
            Q = Q[0]
            if len(Q)<3:
                raise InputError("need 3 q-space vector components")

        if isinstance(Q,(list,tuple)):
            q = numpy.array(Q,dtype=numpy.double)
        elif isinstance(Q,numpy.ndarray):
            q = Q
        else:
            raise TypeError("Q vector must be a list, tuple or numpy array")

        # reshape input to have the same q array for all possible
        # types of different input
        if len(q.shape) != 2:
            q = q.reshape(3,1)

        if 'trans' in keyargs:
            trans = keyargs['trans']
        else:
            trans = True

        if 'deg' in keyargs:
            deg = keyargs['deg']
        else:
            deg = True

        angle = numpy.zeros((4,q.shape[1]))
        # set parameters for the calculation
        z = self.Transform(self.ndir) # z
        y = self.Transform(self.idir) # y
        x = self.Transform(self.scatplane) # x

        for i in range(q.shape[1]):
            qvec = q[:,i]

            if trans:
                qvec = self.Transform(qvec)

            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.HXRD.Q2Ang: qvec= %s" %repr(qvec))

            qa = math.VecNorm(qvec)
            tth = 2.*numpy.arcsin(qa/2./self.k0)
            om = tth/2.

            #calculation of the sample azimuth
            phi = -1*numpy.arctan2(math.VecDot(x,qvec),math.VecDot(y,qvec)) - numpy.pi/2. # the sign depends on the phi movement direction

            chi = (math.VecAngle(z,qvec))

            angle[0,i] = om
            angle[1,i] = chi
            angle[2,i] = phi
            angle[3,i] = tth

        if q.shape[1]==1:
            angle = angle.flatten()
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.HXRD.Q2Ang: [om,chi,phi,tth] = %s" %repr(angle))

        if deg:
            return numpy.degrees(angle)
        else:
            return angle

class GID(Experiment):
    """
    class describing grazing incidence x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as it helps with analyzing measured data

    the class describes a four circle (alpha_i,azimuth,twotheta,beta)
    goniometer to help with GID experiments at the ROTATING ANODE.
    3D data can be treated with the use of linear and area detectors.
    see help self.Ang2Q
    """
    def __init__(self,idir,ndir,**keyargs):
        """
        initialization routine for the GID Experiment class

        idir defines the inplane reference direction (idir points into the PB
           direction at zero angles)
        ndir defines the surface normal of your sample (ndir points along the
           innermost sample rotation axis)

        Parameters
        ----------
        same as for the Experiment base class
        """
        Experiment.__init__(self,idir,ndir,**keyargs)

        # initialize Ang2Q conversion
        if "qconv" not in keyargs:
            self._A2QConversion = QConversion(['z-','x+'],['x+','z-'],[0,1,0],wl=self._wl) # 2S+2D goniometer
            self.Ang2Q = self._A2QConversion

    def _set_transform(self,v1,v2,v3):
        """
        set new transformation of the coordinate system to use in the experimental class
        """
        self._t1 = math.CoordinateTransform(v1,v2,v3) # turn idir to Y and ndir to Z

        yi = self._A2QConversion.r_i
        isc = self._A2QConversion.sampleAxis[-1]
        zi = numpy.abs(math.getVector(isc))
        if numpy.all(yi==zi):
            zi = numpy.roll(zi,1)
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.Experiment: Warning, sample orientation convention failed. Using (%.3f %.3f %.3f) as internal z-axis"%(zi[0],zi[1],zi[2]))
        xi = math.VecUnit(numpy.cross(yi,zi))
        # turn r_i to Y and Z define by detector rotation plane
        self._t2 = math.CoordinateTransform(xi,yi,zi)

        self._transform = math.Transform(numpy.dot(self._t2.imatrix,self._t1.matrix))

    def Q2Ang(self,Q,trans=True,deg=True,**kwargs):
        """
        calculate the GID angles needed in the experiment
        the inplane reference direction defines the direction were
        the reference direction is parallel to the primary beam
        (i.e. lattice planes perpendicular to the beam)

        Parameters
        ----------
         Q:          a list or numpy array of shape (3) with
                     q-space vector components

        optional keyword arguments:
         trans:      True/False apply coordinate transformation on Q
         deg:        True/Flase (default True) determines if the
                     angles are returned in radians or degrees

        Returns
        -------
        a numpy array of shape (4) with the four GID scattering angles which are
        [alpha_i,azimuth,twotheta,beta]

         alpha_i:    incidence angle to surface (at the moment always 0)
         azimuth:    sample rotation with respect to the inplane reference direction
         twotheta:   scattering angle
         beta:       exit angle from surface (at the moment always 0)
        """

        for k in kwargs.keys():
            if k not in ['trans','deg']:
                raise Exception("unknown keyword argument given: allowed are 'trans': coordinate transformation flag, 'deg': degree-flag")
        
        if isinstance(Q,list):
            q = numpy.array(Q,dtype=numpy.double)
        elif isinstance(Q,numpy.ndarray):
            q = Q
        else:
            raise TypeError("Q vector must be a list or numpy array")

        if trans:
            q = self.Transform(q)

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.GID.Q2Ang: q = %s" %repr(q))

        # set parameters for the calculation
        z = self.Transform(self.ndir) # z
        y = self.Transform(self.idir) # y
        x = self.Transform(self.scatplane) # x

        # check if reflection is inplane
        if numpy.abs(math.VecDot(q,z)) >= 0.001:
            raise InputError("Reflection not reachable in GID geometry (Q: %s)" %str(q))

        # calculate angle to inplane reference direction
        aref = numpy.arctan2(math.VecDot(x,q),math.VecDot(y,q))

        # calculate scattering angle
        qa = math.VecNorm(q)
        tth = 2.*numpy.arcsin(qa/2./self.k0)
        azimuth = numpy.pi/2 + aref + tth/2.

        if deg:
            ang = [0,numpy.degrees(azimuth),numpy.degrees(tth),0]
        else:
            ang = [0,azimuth,tth,0]

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.GID.Q2Ang: [ai,azimuth,tth,beta] = %s \n difference to inplane reference which is %5.2f" %(str(ang),aref) )

        return ang

    def Ang2Q(self,ai,phi,tt,beta,**kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help GID.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
         ai,phi,tt,beta: sample and detector angles as numpy array, lists or Scalars
                        must be given. all arguments must have the same shape or
                        length

        **kwargs:   optional keyword arguments
            delta:  giving delta angles to correct the given ones for misalignment
                    delta must be an numpy array or list of length 4.
                    used angles are than ai,phi,tt,beta - delta
            UB:     matrix for conversion from (hkl) coordinates to Q of sample
                    used to determine not Q but (hkl)
                    (default: identity matrix)
            wl:     x-ray wavelength in angstroem (default: self._wl)
            deg:    flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        reciprocal space positions as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine
        pass

class GISAXS(Experiment):
    """
    class describing grazing incidence x-ray diffraction experiments
    the class helps with calculating the angles of Bragg reflections
    as well as it helps with analyzing measured data

    the class describes a three circle (alpha_i,twotheta,beta)
    goniometer to help with GISAXS experiments at the ROTATING ANODE.
    3D data can be treated with the use of linear and area detectors.
    see help self.Ang2Q
    """
    def __init__(self,idir,ndir,**keyargs):
        """
        initialization routine for the GISAXS Experiment class

        idir defines the inplane reference direction (idir points into the PB
        direction at zero angles)

        Parameters
        ----------
        same as for the Experiment base class

        """
        Experiment.__init__(self,idir,ndir,**keyargs)

        if "qconv" not in keyargs:
            # initialize Ang2Q conversion
            self._A2QConversion = QConversion(['x+'],['x+','z-'],[0,1,0],wl=self._wl) # 1S+2D goniometer
            self.Ang2Q = self._A2QConversion

    def Q2Ang(self,Q,trans=True,deg=True,**kwargs):
        pass

    def Ang2Q(self,ai,tt,beta,**kwargs):
        """
        angular to momentum space conversion for a point detector. Also see
        help GISAXS.Ang2Q for procedures which treat line and area detectors

        Parameters
        ----------
         ai,tt,beta: sample and detector angles as numpy array, lists or Scalars
                        must be given. all arguments must have the same shape or
                        length

        **kwargs:   optional keyword arguments
            delta:  giving delta angles to correct the given ones for misalignment
                    delta must be an numpy array or list of length 3.
                    used angles are than ai,tt,beta - delta
            UB:     matrix for conversion from (hkl) coordinates to Q of sample
                    used to determine not Q but (hkl)
                    (default: identity matrix)
            wl:     x-ray wavelength in angstroem (default: self._wl)
            deg:    flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        reciprocal space positions as numpy.ndarray with shape ( * , 3 )
        where * corresponds to the number of points given in the input
        """
        # dummy function to have some documentation string available
        # the real function is generated dynamically in the __init__ routine
        pass

class Powder(Experiment):
    """
    Experimental class for powder diffraction
    This class is able to simulate a powder spectrum for the given material
    """
    def __init__(self,mat,**keyargs):
        """
        the class is initialized with xrayutilities.materials.Material instance

        Parameters
        ----------
         mat:        xrayutilities.material.Material instance
                     giving the material for the experimental class
         keyargs:    optional keyword arguments
                     same as for the Experiment base class
        """
        Experiment.__init__(self,[0,1,0],[0,0,1],**keyargs)
        if isinstance(mat,materials.Material):
            self.mat = mat
        else:
            raise TypeError("mat must be an instance of class xrayutilities.materials.Material")

        self.digits = 5 # number of significant digits, needed to identify equal floats
        self.qpos = None

    def PowderIntensity(self,tt_cutoff=180):
        """
        Calculates the powder intensity and positions up to an angle of tt_cutoff (deg)
        and stores the result in:

            data .... array with intensities
            ang ..... angular position of intensities
            qpos .... reciprocal space position of intensities
        """

        # calculate maximal Bragg indices
        hmax = int(numpy.ceil(norm(self.mat.lattice.a1)*self.k0/numpy.pi*numpy.sin(numpy.radians(tt_cutoff/2.))))
        hmin = -hmax
        kmax = int(numpy.ceil(norm(self.mat.lattice.a2)*self.k0/numpy.pi*numpy.sin(numpy.radians(tt_cutoff/2.))))
        kmin = -kmax
        lmax = int(numpy.ceil(norm(self.mat.lattice.a3)*self.k0/numpy.pi*numpy.sin(numpy.radians(tt_cutoff/2.))))
        lmin = -lmax

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.Powder.PowderIntensity: tt_cutoff; (hmax,kmax,lmax): %6.2f (%d,%d,%d)" %(tt_cutoff,hmax,kmax,lmax))

        qlist = []
        qabslist = []
        hkllist = []
        qmax = numpy.sqrt(2)*self.k0*numpy.sqrt(1-numpy.cos(numpy.radians(tt_cutoff)))
        # calculate structure factor for each reflex
        for h in range(hmin,hmax+1):
            for k in range(kmin,kmax+1):
                for l in range(lmin,lmax+1):
                    q = self.mat.Q(h,k,l)
                    if norm(q)<qmax:
                        qlist.append(q)
                        hkllist.append([h,k,l])
                        qabslist.append(numpy.round(norm(q),self.digits))

        qabs = numpy.array(qabslist,dtype=numpy.double)
        s = self.mat.StructureFactorForQ(qlist,self.energy)
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
        # see L.S. Zevin : Quantitative X-Ray Diffractometry
        # page 18ff
        polarization_factor = (1+numpy.cos(numpy.radians(2*self.ang))**2)/2.
        lorentz_factor = 1./(numpy.sin(numpy.radians(self.ang))**2*numpy.cos(numpy.radians(self.ang)))
        unitcellvol = self.mat.lattice.UnitCellVolume()
        self.data = self.data * polarization_factor * lorentz_factor / unitcellvol**2

    def Convolute(self,stepwidth,width,min=0,max=None):
        """
        Convolutes the intensity positions with Gaussians with width in momentum space
        of "width". returns array of angular positions with corresponding intensity

            theta ... array with angular positions
            int ..... intensity at the positions ttheta
        """

        if not max: max= 2*self.k0

        # convolute each peak with a gaussian and add them up
        qcoord = numpy.arange(min,max,stepwidth)
        theta = self.Q2Ang(qcoord)
        intensity = numpy.zeros(theta.size,dtype=numpy.double)

        for i in range(self.ang.size):
            intensity += math.Gauss1d(qcoord,self.qpos[i],width,self.data[i],0)

        return theta,intensity

    def _Ang2Q(self,th,deg=True):
        """
        Converts theta angles to reciprocal space positions
        returns the absolute value of momentum transfer
        """
        if deg:
            lth = numpy.radians(th)
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
            th= numpy.degrees(th)

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
            ostr += "      h k l     |    tth    |    |Q|    |    Int     |   Int (%)\n"
            ostr += "   ---------------------------------------------------------------\n"
            for i in range(self.qpos.size):
                ostr += "%15s   %8.4f   %8.3f   %10.2f  %10.2f\n" % (self.hkl[i].__str__(), 2*self.ang[i],self.qpos[i],self.data[i], self.data[i]/max*100.)

        return ostr

