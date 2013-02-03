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

from .. import config
from .. import math

try:
    from matplotlib import pyplot as plt
except RuntimeError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analysis.sample_align: warning; plotting functionality not available")


#################################################
## channel per degree calculation
#################################################
def psd_chdeg(angles,channels,plot=True):
    """
    function to determine the channels per degree using a linear
    fit.

    Parameters
    ----------
     angles:    detector angles for which the position of the beam was
                measured
     channels:  detector channels where the beam was found
     plot:      flag to specify if a visualization of the fit should be done

    Returns:
     (chdeg,centerch)
        chdeg:    channel per degree
        centerch: center channel of the detector
    """

    (a_s,b_s,r,tt,stderr)=scipy.stats.linregress(angles,channels)

    if config.VERBOSITY >= config.DEBUG:
        print ("XU.analysis.psd_chdeg: %8.4f %8.4f %6.4f %6.4f %6.4f" %(a_s,b_s,r,tt,stderr))
    centerch = scipy.polyval(numpy.array([a_s,b_s]),0.0)
    chdeg = a_s

    try: plt.__name__
    except NameError:
            print("XU.analyis.psd_chdeg: Warning: plot functionality not available")
            plot = False

    if plot:
        ymin = min(min(channels),centerch)
        ymax = max(max(channels),centerch)
        xmin = min(min(angles),0.0)
        xmax = max(max(angles),0.0)
        # open new figure for the plot
        plt.figure()
        plt.plot(angles,channels,'kx',ms=8.,mew=2.)
        plt.plot([xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1],scipy.polyval(numpy.array([a_s,b_s]),[xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1]),'g-',linewidth=1.5)
        ax = plt.gca()
        plt.grid()
        ax.set_xlim(xmin-(xmax-xmin)*0.15,xmax+(xmax-xmin)*0.15)
        ax.set_ylim(ymin-(ymax-ymin)*0.15,ymax+(ymax-ymin)*0.15)
        plt.vlines(0.0,ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1,linewidth=1.5)
        plt.xlabel("detector angle")
        plt.ylabel("PSD channel")

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.psd_chdeg: channel per degree / center channel: %8.4f / %8.4f (R=%6.4f)" % (chdeg,centerch,r))
    return (chdeg,centerch)


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


