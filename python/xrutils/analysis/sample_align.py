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
# Copyright (C) 2011 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
functions to help with experimental alignment during experiments, especially for
experiments with linear detectors
"""

import numpy
import scipy
import scipy.stats
import scipy.optimize as optimize
from scipy.odr import odrpack as odr
from scipy.odr import models

from .. import config
from .. import math
from .line_cuts import fwhm_exp

try:
    from matplotlib import pyplot as plt
except RuntimeError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analysis.sample_align: warning; plotting functionality not available")


#################################################
## channel per degree calculation
#################################################
def psd_chdeg(angles,channels,stdev=None,usetilt=False,plot=True):
    """
    function to determine the channels per degree using a linear
    fit of the function nchannel = center_ch+chdeg*tan(angles)
    or the equivalent including a detector tilt

    Parameters
    ----------
     angles:    detector angles for which the position of the beam was
                measured
     channels:  detector channels where the beam was found
     keyword arguments:
      stdev     standard deviation of the beam position
      plot:     flag to specify if a visualization of the fit should be done
      usetilt   whether to use model considering a detector tilt (deviation angle of the pixel direction from orthogonal to the primary beam) (default: False)

    Returns:
     (chdeg,centerch[,tilt])
        chdeg:    channel per degree
        centerch: center channel of the detector
        tilt: tilt of the detector from perpendicular to the beam
    
    Note:
    distance is given by: channel_width*channelperdegree/tan(radians(1))
    """

    if stdev == None:
        stdevu = numpy.ones(len(channels))
    else:
        stdevu = stdev

    # define detector model and other functions needed for the tilt
    def straight_tilt(p, x):
        """
        model for straight-linear detectors including tilt 
        
        Parameters
        ----------
         p ... [D/w_pix*pi/180 ~= channel/degree, center_channel, detector_tilt] 
         x ... independent variable of the model: detector angle (degree)
        """
        rad = numpy.radians(x)
        r = numpy.degrees(p[0])*numpy.sin(rad)/numpy.cos(rad-numpy.radians(p[2])) + p[1]
        return r
            
    def straight_tilt_der_x(p, x):
        """
        derivative of straight-linear detector model with respect to the angle
        for parameter description see straigt_tilt
        """
        rad = numpy.radians(x)
        p2 = numpy.radians(p[2])
        r = numpy.degrees(p[0])*(numpy.cos(rad)/numpy.cos(rad-p2) + numpy.sin(rad)/numpy.cos(rad-p2)**2*numpy.sin(rad-p2))
        return r
        
    def straight_tilt_der_p(p, x):
        """
        derivative of straight-linear detector model with respect to the paramters
        for parameter description see straigt_tilt
        """
        rad = numpy.radians(x)
        p2 = numpy.radians(p[2])
        r = numpy.concatenate([180./numpy.pi*numpy.sin(rad)/numpy.cos(rad-p2),\
            numpy.ones(x.shape,dtype=numpy.float),\
            -numpy.degrees(p[0])*numpy.sin(rad)/numpy.cos(rad-p2)**2*numpy.sin(rad-p2)])
        r.shape = (3,) + x.shape
        return r

    # fit linear
    model = models.unilinear
    data = odr.RealData(angles,channels,sy=stdevu)
    my_odr = odr.ODR(data,model)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fitlin = my_odr.run()

    # fit linear with tangens angle
    model = models.unilinear
    data = odr.RealData(numpy.degrees(numpy.tan(numpy.radians(angles))),channels,sy=stdevu)
    my_odr = odr.ODR(data,model)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fittan = my_odr.run()

    if usetilt:
        # fit tilted straight detector model
        model = odr.Model(straight_tilt, fjacd=straight_tilt_der_x, fjacb=straight_tilt_der_p)
        data = odr.RealData(angles,channels,sy=stdevu)
        my_odr = odr.ODR(data,model,beta0=[fittan.beta[0],fittan.beta[1],0])
        # fit type 2 for least squares
        my_odr.set_job(fit_type=2)
        fittilt = my_odr.run()        

    try: plt.__name__
    except NameError:
            print("XU.analyis.psd_chdeg: Warning: plot functionality not available")
            plot = False

    if plot:
        plt.figure()
        # first plot to show linear model
        plt.subplot(211)
        if stdev == None:
            plt.plot(angles,channels,'kx',label='data')
        else:
            plt.errorbar(angles,channels,fmt='kx',yerr=stdevu,label='data')
        angr = angles.max()-angles.min()
        angp = numpy.linspace(angles.min()-angr*0.1,angles.max()+angr*.1,1000)
        plt.plot(angp,models._unilin(fittan.beta,numpy.degrees(numpy.tan(numpy.radians(angp)))),'r-',label='tan')
        plt.plot(angp,models._unilin(fitlin.beta,angp),'k-',label='')
        if usetilt:
            plt.plot(angp,straight_tilt(fittilt.beta,angp),'b-',label='w/tilt')

        plt.ylabel("PSD channel")
        
        # lower plot to show deviations from linear model
        plt.subplot(212)
        if stdev == None:
            plt.plot(angles,channels - models._unilin(fitlin.beta,angles),'kx',label='data')
        else:
            plt.errorbar(angles,channels - models._unilin(fitlin.beta,angles),fmt='kx',yerr=stdevu,label='data')
        plt.plot(angp,models._unilin(fittan.beta,numpy.degrees(numpy.tan(numpy.radians(angp)))) - models._unilin(fitlin.beta,angp),'r-',label='tan')
        if usetilt:
            plt.plot(angp,straight_tilt(fittilt.beta,angp) - models._unilin(fitlin.beta,angp),'b-',label='w/tilt')
        plt.xlabel("detector angle")
        plt.ylabel("PSD channel - linear trend")
        plt.hlines(0,angp.min(),angp.max())
        plt.legend(numpoints=1)

        if usetilt:
            plt.suptitle("center_ch: %8.2f; ch/deg: %8.2f; tilt: %5.2fdeg"%(fittilt.beta[0],fittilt.beta[1],fittilt.beta[2]))
        else:
            plt.suptitle("center_ch: %8.2f; ch/deg: %8.2f"%(fittan.beta[0],fittan.beta[1]))
    
    if usetilt: 
        fit = fittilt
    else: 
        fit = fittan

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.psd_chdeg: channel per degree / center channel: %8.2f / %8.2f" % (fit.beta[0],fit.beta[1]))

    return fit.beta

#################################################
## channel per degree calculation from scan with
## linear detector (determined maximum by Gauss fit)
#################################################
def linear_detector_calib(angle,mca_spectra,**keyargs):
    """
    function to calibrate the detector distance/channel per degrees
    for a straigt linear detector mounted on a detector arm
    
    parameters
    ----------
     angle ........ array of angles in degree of measured detector spectra
     mca_spectra .. corresponding detector spectra 
                    (shape: (len(angle),Nchannels) 
     **keyargs passed to psd_chdeg function used for the modelling
     
    returns
    -------
     channelperdegree,centerchannel[,tilt]

     distance is given by: channel_width*channelperdegree/tan(radians(1))
    """

    # max intensity per spectrum
    mca_int = mca_spectra.sum(axis=1)
    mca_avg = numpy.average(mca_int)
    mca_rowmax = numpy.max(mca_int)
    mca_std = numpy.std(mca_int)

    # determine positions
    pos = []
    posstd = []
    ang = []
    nignored = 0
    for i in range(len(mca_spectra)):
        #print(i)
        row = mca_spectra[i,:]
        row_int = row.sum()
        #print(row_int)
        if (numpy.abs(row_int-mca_avg) > 3*mca_std) or (row_int-mca_rowmax*0.7 < 0):
            if config.VERBOSITY >= config.DEBUG:
                print("XU.analysis.det_dist: spectrum #%d out of intensity range -> ignored" %i)
            nignored += 1
            continue
            
        maxp = numpy.argmax(row)
        fwhm = fwhm_exp(numpy.arange(row.size),row)
        N = int(7*numpy.ceil(fwhm))//2*2

        # fit beam position
        # determine maximal usable length of array around peak position
        Nuse = min(maxp+N//2,len(row)-1) - max(maxp-N//2,0)
        #print("%d %d %d"%(N,max(maxp-N//2,0),min(maxp+N//2,len(row)-1)))
        param, perr, itlim = math.gauss_fit(numpy.arange(Nuse),row[max(maxp-N//2,0):min(maxp+N//2,len(row)-1)])
        param[0] += max(maxp-N//2,0)
        pos.append(param[0])
        posstd.append(perr[0])
        ang.append(angle[i])
        
    ang = numpy.array(ang)
    pos = numpy.array(pos)
    posstd = numpy.array(posstd)

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.det_distance: used/total spectra: %d/%d" %(mca_spectra.shape[0]-nignored,mca_spectra.shape[0]))
    return psd_chdeg(ang, pos, stdev=posstd, **keyargs)

#################################################
## equivalent to PSD_refl_align MATLAB script
## from J. Stangl
#################################################
def psd_refl_align(primarybeam,angles,channels,plot=True):
    """
    function which calculates the angle at which the sample
    is parallel to the beam from various angles and detector channels
    from the reflected beam. The function can be used during the half
    beam alignment with a linear detector.

    Parameters
    ----------
    primarybeam :   primary beam channel number
    angles :        list or numpy.array with angles
    channels :      list or numpy.array with corresponding detector channels
    plot:           flag to specify if a visualization of the fit is wanted
                    default: True

    Returns
    -------
    omega : angle at which the sample is parallel to the beam

    Example
    -------
    >>> psd_refl_align(500,[0,0.1,0.2,0.3],[550,600,640,700])

    """

    (a_s,b_s,r,tt,stderr)=scipy.stats.linregress(channels,angles)

    zeropos = scipy.polyval(numpy.array([a_s,b_s]),primarybeam)

    try: plt.__name__
    except NameError:
            print("XU.analyis.psd_chdeg: Warning: plot functionality not available")
            plot = False

    if plot:
        xmin = min(min(channels),primarybeam)
        xmax = max(max(channels),primarybeam)
        ymin = min(min(angles),zeropos)
        ymax = max(max(angles),zeropos)
        # open new figure for the plot
        plt.figure()
        plt.plot(channels,angles,'kx',ms=8.,mew=2.)
        plt.plot([xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1],scipy.polyval(numpy.array([a_s,b_s]),[xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1]),'g-',linewidth=1.5)
        ax = plt.gca()
        plt.grid()
        ax.set_xlim(xmin-(xmax-xmin)*0.15,xmax+(xmax-xmin)*0.15)
        ax.set_ylim(ymin-(ymax-ymin)*0.15,ymax+(ymax-ymin)*0.15)
        plt.vlines(primarybeam,ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1,linewidth=1.5)
        plt.xlabel("PSD Channel")
        plt.ylabel("sample angle")

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.psd_refl_align: sample is parallel to beam at goniometer angle %8.4f (R=%6.4f)" % (zeropos,r))
    return zeropos

#################################################
#  miscut calculation from alignment in 2 and
#  more azimuths
#################################################
def miscut_calc(phi,aomega,zeros=None,plot=True,omega0=None):
    """
    function to calculate the miscut direction and miscut angle of a sample
    by fitting a sinusoidal function to the variation of the aligned
    omega values of more than two reflections.
    The function can also be used to fit reflectivity alignment values
    in various azimuths.

    Parameters
    ----------
     phi:       azimuths in which the reflection was aligned (deg)
     aomega:    aligned omega values (deg)
     zeros:     (optional) angles at which surface is parallel to
                the beam (deg). For the analysis the angles
                (aomega-zeros) are used.
     plot:      flag to specify if a visualization of the fit is wanted.
                default: True
     omega0:    if specified the nominal value of the reflection is not
                included as fit parameter, but is fixed to the specified
                value. This value is MANDATORY if ONLY TWO AZIMUTHs are
                given.

    Returns
    -------
    [omega0,phi0,miscut]

    list with fitted values for
     omega0:    the omega value of the reflection should be close to
                the nominal one
     phi0:      the azimuth in which the primary beam looks upstairs
     miscut:    amplitude of the sinusoidal variation == miscut angle

    """

    if zeros != None:
        om = (numpy.array(aomega)-numpy.array(zeros))
    else:
        om = numpy.array(aomega)

    a = numpy.array(phi)

    if omega0==None:
        # first guess for the parameters
        p0 = (om.mean(),a[om.argmax()],om.max()-om.min()) # omega0,phi0,miscut
        fitfunc = lambda p,phi: numpy.abs(p[2])*numpy.cos(numpy.radians(phi-(p[1]%360.))) + p[0]
    else:
        # first guess for the parameters
        p0 = (a[om.argmax()],om.max()-om.min()) # # omega0,phi0,miscut
        fitfunc = lambda p,phi: numpy.abs(p[1])*numpy.cos(numpy.radians(phi-(p[0]%360.))) + omega0
    errfunc = lambda p,phi,om: fitfunc(p,phi) - om

    p1, success = optimize.leastsq(errfunc, p0, args=(a,om),maxfev=10000)
    if config.VERBOSITY >= config.INFO_ALL:
        print("xu.analysis.misfit_calc: leastsq optimization return value: %d" %success)

    try: plt.__name__
    except NameError:
            print("XU.analyis.psd_chdeg: Warning: plot functionality not available")
            plot = False

    if plot:
        plt.figure()
        plt.plot(a,om,'kx',mew=2,ms=8)
        plt.plot(numpy.linspace(a.min()-45,a.min()+360-45,num=1000),fitfunc(p1,numpy.linspace(a.min()-45,a.min()+360-45,num=1000)),'g-',linewidth=1.5)
        plt.grid()
        plt.xlabel("azimuth")
        plt.ylabel("aligned sample angle")

    if omega0==None:
        ret = [p1[0],p1[1]%360.,numpy.abs(p1[2])]
    else:
        ret = [omega0]+[p1[0]%360.,numpy.abs(p1[1])]

    if config.VERBOSITY >= config.INFO_LOW:
        print("xu.analysis.misfit_calc: \n \
                \t fitted reflection angle: %8.3f \n \
                \t looking upstairs at phi: %8.2f \n \
                \t mixcut angle: %8.3f \n" % (ret[0],ret[1],ret[2]))

    return ret

#################################################
#  correct substrate Bragg peak position in 
#  reciprocal space maps
#################################################
def fit_bragg_peak(om,tt,psd,omalign,ttalign,exphxrd,frange=(0.03,0.03),plot=True):
    """
    helper function to determine the Bragg peak position in a reciprocal 
    space map used to obtain the position needed for correction of the data.
    the determination is done by fitting a two dimensional Gaussian 
    (xrutils.math.Gauss2d)
    
    PLEASE ALWAYS CHECK THE RESULT CAREFULLY!

    Parameter
    ---------
     om,tt: angular coordinates of the measurement (numpy.ndarray)
            either with size of psd or of psd.shape[0]
     psd:   intensity values needed for fitting
     omalign: aligned omega value, used as first guess in the fit
     ttalign: aligned two theta values used as first guess in the fit
              these values are also used to set the range for the fit:
              the peak should be within +/-frange\AA^{-1} of those values
     exphxrd: experiment class used for the conversion between angular and 
              reciprocal space.
     frange:  data range used for the fit in both directions
              (see above for details default:(0.03,0.03) unit: \AA^{-1})
     plot:  if True (default) function will plot the result of the fit in comparison
            with the measurement.
    
    Returns
    -------
    omfit,ttfit,params,covariance: fitted angular values, and the fit
            parameters (of the Gaussian) as well as their errors
    """ 
    if om.size != psd.size:
        [qx,qy,qz] = exphxrd.Ang2Q.linear(om,tt)
    else:
        [qx,qy,qz] = exphxrd.Ang2Q(om,tt)
    [qxsub,qysub,qzsub] = exphxrd.Ang2Q(omalign,ttalign)
    params = [qysub[0],qzsub[0],0.001,0.001,psd.max(),0,0.]
    params,covariance = math.fit_peak2d(qy.flatten(),qz.flatten(),psd.flatten(),params,[qysub[0]-frange[0],qysub[0]+frange[0],qzsub[0]-frange[1],qzsub[0]+frange[1]],math.Gauss2d,maxfev=10000)
    # correct params
    params[6] = params[6]%(numpy.pi)
    if params[5]<0 : params[5] = 0

    [omfit,dummy,dummy,ttfit] = exphxrd.Q2Ang((0,params[0],params[1]),trans=False,geometry="real")
    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.fit_bragg_peak:fitted substrate angles: \n\tom =%8.4f \n\ttt =%8.4f" %(omfit,ttfit))
    
    if plot:
        plt.figure(); plt.clf()
        from .. import gridder
        from .. import utilities
        gridder = gridder.Gridder2D(200,200)
        gridder(qy,qz,psd)
        # calculate intensity which should be plotted
        INT = utilities.maplog(gridder.gdata.transpose(),4,0)
        QXm = gridder.xmatrix
        QZm = gridder.ymatrix
        cl = plt.contour(gridder.xaxis,gridder.yaxis,utilities.maplog(math.Gauss2d(QXm,QZm,*params),4,0).transpose(),8,colors='k',linestyles='solid')
        cf = plt.contourf(gridder.xaxis, gridder.yaxis, INT,35)
        cf.collections[0].set_label('data')
        cl.collections[0].set_label('fit')
        #plt.legend() # for some reason not working?
        plt.colorbar(extend='min')
        plt.title("plot shows only coarse data! fit used raw data!")
    
    return omfit,ttfit,params,covariance


