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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrutils_example_plot_3D_ESRF_ID01.py

import numpy
import xrutils as xu
import os

default_en=10330.0 # x-ray energy in eV
default_cch = [348.9,158.6] # vertical(del)/horizontal(nu) z-y+ (although in reality slightly tilted! (0.6deg when last determined (Oct2012))
# cch describes the "zero" position of the detector. this means at
# "detector arm angles"=0 the primary beam is hitting the detector at some particular position. these two values specify this pixel position
default_chpdeg = [280.8, 280.8] # channel per degree for the detector
# chpdeg specify how many pixels the beam position on the detector changes
# for 1 degree movement. basically this determines the detector distance
# and needs to be determined every time the distance is changed
default_nav = [2,2] # reduce data: number of pixels to average in each detector direction
default_roi = [0,516,0,516] # region of interest on the detector

def filtfact(att):
    """
    function to determine the absorper correction factor 
    in case filters where used
    """
    
    attnum = numpy.array(att,dtype=numpy.int)
    ret = numpy.ones(attnum.shape)
    fact = (1,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan) # filter factors to be determined by reference measurements at the energy you use
    if ret.size > 1:
        ret[:] = numpy.nan
        ret[0 == attnum] = 1.0000
        ret[2 == attnum] = 3.6769
        return ret
    else:
        return fact[int(att)]
        
        
xid01_normalizer = xu.IntensityNormalizer('CCD',time='Seconds',mon='Opt2',absfun = lambda d: filtfact(d['Filter'])) # av_mon=1.e8
# define intensity normalizer class to normalize for count time and monitor changes: to have comparable absolute intensities set the keyword argument av_mon to a fixed value, otherwise different scans can not be compared!

def hotpixelkill(ccd):
    """
    function to remove hot pixels from CCD frames
    ADD REMOVE VALUES IF NEEDED!
    """  
    ccd[44,159] = 0
    ccd[45,159] = 0
    ccd[43,160] = 0
    ccd[46,160] = 0
    ccd[44,161] = 0
    ccd[45,161] = 0
    ccd[304,95] = 0
    ccd[414,283] = 0
    return ccd

def rawmap(h5file,scannr,ccdfiletmp,roi=default_roi,angdelta=[0,0,0,0,0],en=default_en,cch=default_cch,chdeg=default_chpdeg):
    """
    read ccd frames and and convert them in reciprocal space
    angular coordinates are taken from the spec file
    """

    [mu,eta,phi,nu,delta],sdata = xu.io.geth5_scan(h5file,scannr,'Mu','Eta','Phi','Nu','Delta')
    ccdn = sdata['ccd_n']
    
    qconv = xu.experiment.QConversion(['z+','y-','z-'],['z+','y-'],[1,0,0]) # 3S+2D goniometer (simplified ID01 goniometer, sample mu,phi detector nu,del
# convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)

    hxrd = xu.HXRD([1,0,0],[0,0,1],en=en,qconv=qconv) # define experimental class for angle conversion
hxrd.Ang2Q.init_area('z-','y+',cch1=cch[0],cch2=cch[1],Nch1=516,Nch2=516, chpdeg1=chpdeg[0],chpdeg2=chpdeg[1],Nav=nav,roi=roi) # initialize area detector properties
    
    for idx in range(len(ccdn)):
        i = ccdn[idx]
        # read ccd image from EDF file
        e = xu.io.EDFFile(ccdfiletmp%i)
        ccdraw = e.data
        ccd = hotpixelkill(ccdraw)
      
        # normalize ccd-data (optional)
        #d = {'CCD': ccd, 'Opt2': sdata['Opt2'][idx], 'Filter': sdata['Filter'][idx], 'Seconds': sdata['Seconds'][idx]} # create data object for normalization
        #ccd = xid01_normalizer(d)
      
        #here a darkfield correction would be done
        # reduce data size
        CCD = xu.blockAverage2D(ccd, nav[0],nav[1], roi=roi) 
        
        if i==ccdn[0]:
            intensity = numpy.zeros( (len(ccdn),) + CCD.shape )

        intensity[i-ccdn[0],:,:] = CCD
    
    # transform scan angles to reciprocal space coordinates for all detector pixels
    qx,qy,qz = hxrd.Ang2Q.area(mu,eta,phi,nu,delta,delta=angdelta)
    
    return qx,qy,qz,intensity

def gridmap(h5file,scannr,ccdfiletmp,nx,ny,nz,**kwargs):
    """
    read ccd frames and grid them in reciprocal space
    angular coordinates are taken from the spec file
    
    *kwargs are passed to the rawmap function
    """
   
    qx,qy,qz,intensity = rawmap(h5file,scannr,ccdfiletmp,*kwargs)

    # convert data to rectangular grid in reciprocal space
    gridder = xu.Gridder3D(nx,ny,nz)
    gridder(qx,qy,qz,intensity)
    
    return gridder.xaxis,gridder.yaxis,gridder.zaxis,gridder.gdata,gridder


