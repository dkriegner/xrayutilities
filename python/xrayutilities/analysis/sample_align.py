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

import re
import numpy
import scipy
import scipy.stats
import scipy.optimize as optimize
import time
from scipy.odr import odrpack as odr
from scipy.odr import models
from scipy.ndimage.measurements import center_of_mass

from .. import config
from .. import math
from .. import utilities
from .line_cuts import fwhm_exp
from ..exception import InputError
from .. import libxrayutils

try:
    from matplotlib import pyplot as plt
except RuntimeError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analysis.sample_align: warning; plotting functionality not available")

# regular expression to check goniometer circle syntax
circleSyntax = re.compile("[xyz][+-]")

#################################################
## channel per degree calculation
#################################################
def psd_chdeg(angles,channels,stdev=None,usetilt=False,plot=True,datap="kx",modelline="r--",modeltilt="b-",fignum=None,mlabel="fit",mtiltlabel="fit w/tilt",dlabel="data"):
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
     (L/pixelwidth*pi/180 ,centerch[,tilt]):

    L/pixelwidth*pi/180 = channel/degree for large detector distance
    with L sample detector disctance, and
    pixelwidth the width of one detector channel

    centerch: center channel of the detector
    tilt: tilt of the detector from perpendicular to the beam

    Note:
     distance of the detector is given by: channelwidth*channelperdegree/tan(radians(1))
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
         p ... [L/w_pix*pi/180 ~= channel/degree, center_channel, detector_tilt]
               with L sample detector disctance
               w_pix the width of one detector channel
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
        markersize = 6.0
        markeredgewidth = 1.5
        linewidth = 2.0
        if fignum==None:
            plt.figure()
        else:    
            plt.figure(fignum)
        # first plot to show linear model
        ax1 = plt.subplot(211)
        if stdev == None: 
            plt.plot(angles,channels,datap, ms=markersize, mew=markeredgewidth ,label=dlabel)
        else:
            plt.errorbar(angles,channels,fmt=datap,yerr=stdevu, ms=markersize, mew=markeredgewidth ,label=dlabel,ecolor='0.5')
        angr = angles.max()-angles.min()
        angp = numpy.linspace(angles.min()-angr*0.1,angles.max()+angr*.1,1000)
        plt.plot(angp,models._unilin(fittan.beta,numpy.degrees(numpy.tan(numpy.radians(angp)))),modelline,label=mlabel,lw=linewidth)
        plt.plot(angp,models._unilin(fitlin.beta,angp),'k-',label='')
        plt.grid(True)
        if usetilt:
            plt.plot(angp,straight_tilt(fittilt.beta,angp),modeltilt, label=mtiltlabel,lw=linewidth)
        leg = plt.legend(numpoints=1)
        leg.get_frame().set_alpha(0.8)

        plt.ylabel("channel number")

        # lower plot to show deviations from linear model
        ax2 = plt.subplot(212,sharex=ax1)
        if stdev == None:
            plt.plot(angles,channels - models._unilin(fitlin.beta,angles),datap, ms=markersize, mew=markeredgewidth ,label=dlabel)
        else:
            plt.errorbar(angles,channels - models._unilin(fitlin.beta,angles),fmt=datap,yerr=stdevu, ms=markersize, mew=markeredgewidth ,label=dlabel,ecolor='0.5')
        plt.plot(angp,models._unilin(fittan.beta,numpy.degrees(numpy.tan(numpy.radians(angp)))) - models._unilin(fitlin.beta,angp),modelline,label=mlabel,lw=linewidth)
        if usetilt:
            plt.plot(angp,straight_tilt(fittilt.beta,angp) - models._unilin(fitlin.beta,angp),modeltilt,label=mtiltlabel,lw=linewidth)
        plt.xlabel("detector angle (deg)")
        plt.ylabel("ch. num. - linear trend")
        plt.grid(True)
        plt.hlines(0,angp.min(),angp.max())

        if usetilt:
            plt.suptitle("L/w*pi/180: %8.2f; center channel: %8.2f; tilt: %5.2fdeg"%(fittilt.beta[0],fittilt.beta[1],fittilt.beta[2]))
        else:
            plt.suptitle("L/w*pi/180: %8.2f; center channel: %8.2f"%(fittan.beta[0],fittan.beta[1]))

    if usetilt:
        fit = fittilt
    else:
        fit = fittan

    if config.VERBOSITY >= config.INFO_LOW:
        if usetilt:
            print("XU.analysis.psd_chdeg: L/w*pi/180 ~= channel per degree / center channel / tilt: %8.2f / %8.2f / %6.3fdeg" % (fit.beta[0],fit.beta[1],fit.beta[2]))
            print("XU.analysis.psd_chdeg:     errors of channel per degree / center channel / tilt: %8.3f / %8.3f / %6.3fdeg" % (fit.sd_beta[0],fit.sd_beta[1],fit.sd_beta[2]))
        else:
            print("XU.analysis.psd_chdeg: L/w*pi/180 ~= channel per degree / center channel: %8.2f / %8.2f" % (fit.beta[0],fit.beta[1]))
            print("XU.analysis.psd_chdeg:     errors of channel per degree / center channel: %8.3f / %8.3f" % (fit.sd_beta[0],fit.sd_beta[1]))


    return fit.beta

#################################################
## channel per degree calculation from scan with
## linear detector (determined maximum by Gauss fit)
#################################################
def linear_detector_calib(angle,mca_spectra,**keyargs):
    """
    function to calibrate the detector distance/channel per degrees
    for a straight linear detector mounted on a detector arm

    parameters
    ----------
     angle ........ array of angles in degree of measured detector spectra
     mca_spectra .. corresponding detector spectra
                    (shape: (len(angle),Nchannels)

    **keyargs passed to psd_chdeg function used for the modelling additional options:
     r_i .......... primary beam direction as vector [xyz][+-]; default: 'y+'
     detaxis ...... detector arm rotation axis [xyz][+-] e.g. 'x+'; default: 'x+'

    returns
    -------
     L/pixelwidth*pi/180 ~= channel/degree, center_channel[, detector_tilt]

    The function also prints out how a linear detector can be initialized using the results
    obtained from this calibration. Carefully check the results

    Note:
     distance of the detector is given by: channel_width*channelperdegree/tan(radians(1))
    """

    if "detaxis" in keyargs:
        detrotaxis = keyargs["detaxis"]
    else: # use default
        detrotaxis = 'x+'
    if "r_i" in keyargs:
        r_i = keyargs["r_i"]
    else: # use default
        r_i = 'y+'

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
                print("XU.analysis.linear_detector_calib: spectrum #%d out of intensity range -> ignored" %i)
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

    detparam = psd_chdeg(ang, pos, stdev=posstd, **keyargs)
    if numpy.sign(detparam[0]) > 0:
        sign = '-'
    else:
        sign = '+'

    detaxis='  '
    detd = numpy.cross(math.getVector(detrotaxis),math.getVector(r_i))
    argm = numpy.abs(detd).argmax()

    def flipsign(char,val):
        if numpy.sign(val)<0:
            if char=='+':
                return '-'
            else:
                return '+'
        else:
            return char

    if argm == 0:
        detaxis = 'x'+flipsign(sign,detd[argm])
    elif argm == 1:
        detaxis = 'y'+flipsign(sign,detd[argm])
    elif argm == 2:
        detaxis = 'z'+flipsign(sign,detd[argm])

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.linear_detector_calib:\n\tused/total spectra: %d/%d" %(mca_spectra.shape[0]-nignored,mca_spectra.shape[0]))
        print("\tdetector rotation axis (given by user): %s" %detrotaxis)
        if len(detparam)==3:
            tilt = detparam[2]
        else:
            tilt = 0
        print("\tdetector initialization with: init_linear('%s',%.1f,%d,chpdeg=%.1f,tilt=%.2f)" %(detaxis,detparam[1],mca_spectra.shape[1],numpy.abs(detparam[0]),tilt))

    return detparam

######################################################
## detector parameter calculation from scan with
## area detector (determine maximum by center of mass)
######################################################
def area_detector_calib(angle1,angle2,ccdimages,detaxis,r_i,plot=True,cut_off = 0.7,start = (0,0,0,0), fix = (False,False,False,False), fig=None,wl=None):
    """
    function to calibrate the detector parameters of an area detector
    it determines the detector tilt possible rotations and offsets in the
    detector arm angles

    parameters
    ----------
     angle1 ..... outer detector arm angle
     angle2 ..... inner detector arm angle
     ccdimages .. images of the ccd taken at the angles given above
     detaxis .... detector arm rotation axis
                  default: ['z+','y-']
     r_i ........ primary beam direction [xyz][+-]
                  default 'x+'

    keyword_arguments:
        plot .... flag to determine if results and intermediate results should be plotted
                  default: True
        cut_off . cut off intensity to decide if image is used for the determination or not
                  default: 0.7 = 70%
        start ... sequence of start values of the fit for parameters,
                  which can not be estimated automatically
                  these are:
                  tiltazimuth,tilt,detector_rotation,outerangle_offset. By default (0,0,0,0) is used.
        fix ..... fix parameters of start (default: (False,False,False,False))
        fig ..... matplotlib figure used for plotting the error
                  default: None (creates own figure)
        wl ...... wavelength of the experiment in Angstrom (default: config.WAVELENGTH)
                  value does not matter here and does only affect the scaling of the error
    """

    debug=False
    plotlog=False
    if plot:
        try: plt.__name__
        except NameError:
            print("XU.analyis.area_detector_calib: Warning: plot functionality not available")
            plot = False

    if wl==None:
        wl = config.WAVELENGTH
    else:
        wl = utilities.wavelength(wl)

    t0 = time.time()
    Npoints = len(angle1)
    if debug:
        print("number of given images: %d"%Npoints)

    # determine center of mass position from detector images
    # also use only images with an intensity larger than 70% of the average intensity
    n1 = numpy.zeros(0,dtype=numpy.double)
    n2 = n1
    ang1 = n1
    ang2 = n1

    avg = 0
    for i in range(Npoints):
        avg += numpy.sum(ccdimages[i])
    avg /= float(Npoints)
    (N1,N2) =  ccdimages[0].shape

    if debug:
        print("average intensity per image: %.1f" %avg)

    for i in range(Npoints):
        img = ccdimages[i]
        if numpy.sum(img) > cut_off*avg:
            [cen1,cen2] = center_of_mass(img)
            n1 = numpy.append(n1,cen1)
            n2 = numpy.append(n2,cen2)
            ang1 = numpy.append(ang1,angle1[i])
            ang2 = numpy.append(ang2,angle2[i])
            #if debug:
            #    print("%8.3f %8.3f \t%.2f %.2f"%(angle1[i],angle2[i],cen1,cen2))
    Nused = len(ang1)

    if debug:
        print("Nused / Npoints: %d / %d" %(Nused,Npoints))

    # guess initial parameters
    # center channel and detector pixel direction and pixel size
    (s1,i1,r1,dummy,dummy)=scipy.stats.linregress(ang1-start[3],n1)
    (s2,i2,r2,dummy,dummy)=scipy.stats.linregress(ang1-start[3],n2)
    (s3,i3,r3,dummy,dummy)=scipy.stats.linregress(ang2,n1)
    (s4,i4,r4,dummy,dummy)=scipy.stats.linregress(ang2,n2)
    if debug:
        print("%.2f %.2f %.2f %.2f"%(s1,s2,s3,s4))
        print("%.2f %.2f %.2f %.2f"%(r1,r2,r3,r4))
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(ang1-start[3],n1,'bx',label='channel 1')
            plt.plot(ang1-start[3],n2,'rx',label='channel 2')
            plt.legend()
            plt.xlabel('angle 1')
            plt.subplot(212)
            plt.plot(ang2,n1,'bx',label='channel 1')
            plt.plot(ang2,n2,'rx',label='channel 2')
            plt.legend()
            plt.xlabel('angle 2')

    # determine detector directions
    s = ord('x') + ord('y') + ord('z')
    c1 = ord(detaxis[0][0]) + ord(r_i[0])
    c2 = ord(detaxis[1][0]) + ord(r_i[0])
    sign1 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[0]),math.getVector(r_i))))
    sign2 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[1]),math.getVector(r_i))))
    if r1**2>r2**2:
        detdir1 = chr(s-c1)
        detdir2 = chr(s-c2)
        if numpy.sign(s1) > 0:
            if sign1 > 0:
                detdir1 += '-'
            else:
                detdir1 += '+'
        else:
            if sign1 > 0:
                detdir1 += '+'
            else:
                detdir1 += '-'

        if numpy.sign(s4) > 0:
            if sign2 > 0:
                detdir2 += '-'
            else:
                detdir2 += '+'
        else:
            if sign2 > 0:
                detdir2 += '+'
            else:
                detdir2 += '-'
    else:
        detdir1 = chr(s-c2)
        detdir2 = chr(s-c1)
        if numpy.sign(s3) > 0:
            if sign2 > 0:
                detdir1 += '-'
            else:
                detdir1 += '+'
        else:
            if sign2 > 0:
                detdir1 += '+'
            else:
                detdir1 += '-'

        if numpy.sign(s2) > 0:
            if sign1 > 0:
                detdir2 += '-'
            else:
                detdir2 += '+'
        else:
            if sign1 > 0:
                detdir2 += '+'
            else:
                detdir2 += '-'

    epslist = []
    paramlist = []
    epsmin = 1.
    fitmin = None

    print("tiltaz   tilt   detrot   offset:  error (relative) (fittime)")
    print("------------------------------------------------------------")
    #find optimal detector rotation (however keep other parameters free)
    detrot = start[2]
    if not fix[2]:
        for detrotstart in numpy.linspace(start[2]-1,start[2]+1,20):
            start = start[:2] + (detrotstart,) + (start[3],)
            eps,param,fit = _area_detector_calib_fit(ang1,ang2,n1,n2,detaxis,r_i,detdir1, detdir2,start = start, fix = fix, full_output=True,wl = wl)
            epslist.append(eps)
            paramlist.append(param)
            if epslist[-1]<epsmin:
                epsmin = epslist[-1]
                parammin = param
                fitmin = fit
                detrot = param[6]
            if debug:
                print("single fit")
                print(param)

    Ntiltaz = 1 if fix[0] else 5
    Ntilt = 1 if fix[1] else 6
    Noffset = 1 if fix[3] else 100
    if fix[3]:
        Ntilt = Ntilt*5 if not fix[1] else Ntilt
        Ntiltaz = Ntiltaz*5 if not fix[0] else Ntiltaz

    startparam = start[:2] + (detrot,) + (start[3],)

    for tiltazimuth in numpy.linspace(startparam[0] if fix[0] else 0,360,Ntiltaz,endpoint=False):
        for tilt in numpy.linspace(startparam[1] if fix[1] else 0,4,Ntilt):
            for offset in numpy.linspace(startparam[3] if fix[3] else -3+startparam[3],3+startparam[3],Noffset):
                t1 = time.time()
                start = (tiltazimuth,tilt,detrot,offset)
                eps,param,fit = _area_detector_calib_fit(ang1,ang2,n1,n2,detaxis,r_i,detdir1, detdir2,start = start, fix = fix, full_output=True,wl = wl)
                epslist.append(eps)
                paramlist.append(param)
                t2 = time.time()
                print("%6.1f %6.2f %8.3f %8.3f: %10.4e (%4.2f) (%5.2fsec)" %(start+(epslist[-1],epslist[-1]/epsmin,t2-t1)))

                if epslist[-1]<epsmin:
                    print("************************")
                    print("new global minimum found")
                    epsmin = epslist[-1]
                    parammin = param
                    fitmin = fit
                    print("new best parameters: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" %parammin)
                    print("************************\n")

    (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset) = parammin

    if plot:
        if fig:
            plt.figure(fig.number)
        else:
            plt.figure("CCD Calib fit")
        nparams = numpy.array(paramlist)
        neps = numpy.array(epslist)
        labels = ('cch1 (1)','cch2 (1)',r'pwidth1 ($\mu$m@1m)','pwidth2 ($\mu$m@1m)','tiltazimuth (deg)','tilt (deg)','detrot (deg)','outerangle offset (deg)')
        xscale = (1., 1., 1.e6, 1.e6, 1., 1., 1., 1.)
        #plt.suptitle("best fit: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" %(cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset))
        for p in range(8):
            plt.subplot(3,3,p+1)
            if plotlog:
                plt.semilogy(nparams[:,p]*xscale[p],neps,'k.')
            else:
                plt.scatter(nparams[:,p]*xscale[p],neps,c=nparams[:,-1],
                            s=10,marker='o',cmap=plt.cm.gnuplot,edgecolor='none')
            plt.xlabel(labels[p])

        for p in range(8):
            plt.subplot(3,3,p+1)
            if plotlog:
                plt.semilogy(parammin[p]*xscale[p],epsmin,'ko',ms=8,mew=2.5,mec='k',mfc='w')
            else:
                plt.plot(parammin[p]*xscale[p],epsmin,'ko',ms=8,mew=2.5,mec='k',mfc='w')
                plt.ylim(epsmin*0.7,epsmin*2.)
            plt.locator_params(nbins=4,axis='x')
        plt.tight_layout()

    if config.VERBOSITY >= config.INFO_LOW:
        print("total time needed for fit: %.2fsec" %(time.time()-t0))
        print("fitted parameters: epsilon: %10.4e (%d,%s) " %(epsmin,fitmin.info,repr(fitmin.stopreason)))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" %(cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset))

    if config.VERBOSITY > 0:
        print("please check the resulting data (consider setting plot=True)")
        print("detector rotation axis / primary beam direction (given by user): %s / %s"%(repr(detaxis),r_i))
        print("detector pixel directions / distance: %s %s / %g" %(detdir1,detdir2,1.))
        print("\tdetector initialization with: init_area('%s','%s',cch1=%.2f,cch2=%.2f,Nch1=%d,Nch2=%d, pwidth1=%.4e,pwidth2=%.4e,distance=1.,detrot=%.3f,tiltazimuth=%.1f,tilt=%.3f)" %(detdir1,detdir2,cch1,cch2,N1,N2,pwidth1,pwidth2,detrot,tiltazimuth,tilt))
        print("AND ALWAYS USE an (additional) OFFSET of %.4fdeg in the OUTER DETECTOR ANGLE!" %(outerangle_offset))

    return (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset),eps


def _area_detector_calib_fit(ang1,ang2,n1,n2, detaxis, r_i, detdir1, detdir2, start = (0,0,0,0), fix = (False,False,False,False),full_output=False,wl=1.):
    """
    INTERNAL FUNCTION
    function to calibrate the detector parameters of an area detector
    it determines the detector tilt possible rotations and offsets in the
    detector arm angles

    parameters
    ----------
     angle1 ..... outer detector arm angle
     angle2 ..... inner detector arm angle
     ccdimages .. images of the ccd taken at the angles given above
     detaxis .... detector arm rotation axis
                  default: ['z+','y-']
     detdir1,2 .. detector pixel directions of first and second pixel coordinates; e.g. 'y+'
     r_i ........ primary beam direction [xyz][+-]
                  default 'x+'

    keyword_arguments:
        start ... sequence of start values of the fit for parameters,
                  which can not be estimated automatically
                  these are:
                  tiltazimuth,tilt,detector_rotation,outerangle_offset.
                  By default: (0,0,0,0)
        fix ..... fix parameters of start (default: (False,False,False,False))
        full_output   flag to tell if to return fit object with final parameters and detector directions
        wl ...... wavelength of the experiment in Angstrom (default: 1)
                  value does not matter here and does only affect the scaling of the error

    returns
    -------
        eps   final epsilon of the fit

    if full_output:
        eps,param,fit
    """

    debug=False

    def areapixel(params,detectorDir1,detectorDir2,r_i,detectorAxis,*args,**kwargs):
        """
        angular to momentum space conversion for pixels of an area detector
        the center pixel is in direction of self.r_i when detector angles are zero

        the detector geometry must be given to the routine

        Parameters
        ----------
        *args:          detector angles and channel numbers
                dAngles as numpy array, lists or Scalars
                        in total len(detectorAxis) must be given starting with
                        the outer most circle all arguments must have the same shape or length
                channel numbers n1 and n2 where the primary beam hits the detector
                        same length as the detector values

        params:         parameters of the detector calibration model
                        (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot)
                cch1,2: center pixel, in direction of self.r_i at zero
                        detectorAngles
                pwidth1,2: width of one pixel (same unit as distance)
                tiltazimuth: direction of the tilt vector in the detector plane (in degree)
                tilt: tilt of the detector plane around an axis normal to the direction
                      given by the tiltazimuth
                detrot: detector rotation around the primary beam direction as given by r_i

        detectorDir1:   direction of the detector (along the pixel direction 1);
                        e.g. 'z+' means higher pixel numbers at larger z positions
        detectorDir2:   direction of the detector (along the pixel direction 2); e.g. 'x+'

        r_i:            primary beam direction e.g. 'x+'
        detectorAxis:   list or tuple of detector circles
                        e.g. ['z+','y-'] would mean a detector arm with a two rotations

        **kwargs:       possible keyword arguments
            delta:      giving delta angles to correct the given ones for misalignment
                        delta must be an numpy array or list of len(*dAngles)
                        used angles are than *args - delta
            wl:         x-ray wavelength in angstroem (default: 1 (since it does not matter here))
            distance:   distance of center pixel from center of rotation (default: 1. (since it does not matter here))
            deg:        flag to tell if angles are passed as degree (default: True)

        Returns
        -------
        reciprocal space position of detector pixels n1,n2 in a numpy.ndarray of shape
        ( len(args) , 3 )
        """

        # check detector circle argument
        if isinstance(detectorAxis,(str,list,tuple)):
            if isinstance(detectorAxis,str):
                dAxis = list([detectorAxis])
            else:
                dAxis = list(detectorAxis)
            for circ in dAxis:
                if not isinstance(circ,str) or len(circ)!=2:
                    raise InputError("QConversionPixel: incorrect detector circle type or syntax (%s)" %repr(circ))
                if not circleSyntax.search(circ):
                    raise InputError("QConversionPixel: incorrect detector circle syntax (%s)" %circ)
        else:
            raise TypeError("Qconversion error: invalid type for detectorAxis, must be str, list or tuple")
        # add detector rotation around primary beam
        dAxis += [r_i]
        _detectorAxis_str = ''
        for circ in dAxis:
            _detectorAxis_str += circ

        Nd = len(dAxis)
        Nargs = Nd + 2 -1

        # check detectorDir
        if not isinstance(detectorDir1,str) or len(detectorDir1)!=2:
            raise InputError("QConversionPixel: incorrect detector direction1 type or syntax (%s)" %repr(detectorDir1))
        if not circleSyntax.search(detectorDir1):
            raise InputError("QConversionPixel: incorrect detector direction1 syntax (%s)" %detectorDir1)
        _area_detdir1 = detectorDir1
        if not isinstance(detectorDir2,str) or len(detectorDir2)!=2:
            raise InputError("QConversionPixel: incorrect detector direction2 type or syntax (%s)" %repr(detectorDir2))
        if not circleSyntax.search(detectorDir2):
            raise InputError("QConversionPixel: incorrect detector direction2 syntax (%s)" %detectorDir2)
        _area_detdir2 = detectorDir2

        # parse parameter arguments
        _area_cch1 = float(params[0])
        _area_cch2 = float(params[1])
        _area_pwidth1 = float(params[2])
        _area_pwidth2 = float(params[3])
        _area_tiltazimuth = numpy.radians(params[4])
        _area_tilt = numpy.radians(params[5])
        _area_rot = float(params[6])
        _area_ri = math.getVector(r_i)

        # kwargs
        if "distance" in kwargs:
            _area_distance = float(kwargs["distance"])
        else:
            _area_distance = 1.

        if 'wl' in kwargs:
            wl = utilities.wavelength(kwargs['wl'])
        else:
            wl = 1.

        if 'deg' in kwargs:
            deg = kwargs['deg']
        else:
            deg = True

        if 'delta' in kwargs:
            delta = numpy.array(kwargs['delta'],dtype=numpy.double)
            if delta.size != Nd-1:
                raise InputError("QConversionPixel: keyword argument delta does not have an appropriate shape")
        else:
            delta = numpy.zeros(Nd)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Nargs:
            raise InputError("QConversionPixel: wrong amount (%d) of arguments given, \
                             number of arguments should be %d" %(len(args),Nargs))

        try: Npoints = len(args[0])
        except (TypeError,IndexError): Npoints = 1

        dAngles = numpy.array((),dtype=numpy.double)
        for i in range(Nd-1):
            arg = args[i]
            if not isinstance(arg,(numpy.ScalarType,list,numpy.ndarray)):
                raise TypeError("QConversionPixel: invalid type for one of the detector coordinates, must be scalar, list or array")
            elif isinstance(arg,numpy.ScalarType):
                arg = numpy.array([arg],dtype=numpy.double)
            elif isinstance(arg,list):
                arg = numpy.array(arg,dtype=numpy.double)
            arg = arg - delta[i]
            dAngles = numpy.concatenate((dAngles,arg))
        # add detector rotation around primary beam
        dAngles = numpy.concatenate((dAngles,numpy.ones(arg.shape,dtype=numpy.double)*_area_rot))

        # read channel numbers
        n1 = numpy.array((),dtype=numpy.double)
        n2 = numpy.array((),dtype=numpy.double)

        arg = args[Nd-1]
        if not isinstance(arg,(numpy.ScalarType,list,numpy.ndarray)):
            raise TypeError("QConversionPixel: invalid type for one of the detector coordinates, must be scalar, list or array")
        elif isinstance(arg,numpy.ScalarType):
            arg = numpy.array([arg],dtype=numpy.double)
        elif isinstance(arg,list):
            arg = numpy.array(arg,dtype=numpy.double)
        n1 = arg

        arg = args[Nd]
        if not isinstance(arg,(numpy.ScalarType,list,numpy.ndarray)):
            raise TypeError("QConversionPixel: invalid type for one of the detector coordinates, must be scalar, list or array")
        elif isinstance(arg,numpy.ScalarType):
            arg = numpy.array([arg],dtype=numpy.double)
        elif isinstance(arg,list):
            arg = numpy.array(arg,dtype=numpy.double)
        n2 = arg

        # flatten arrays with angles for passing to C routine
        if Npoints > 1:
            dAngles.shape = (Nd,Npoints)
            dAngles = numpy.ravel(dAngles.transpose())
            n1 = numpy.ravel(n1)
            n2 = numpy.ravel(n2)

        if deg:
            dAngles = numpy.radians(dAngles)

        # check that arrays have correct type and memory alignment for passing to C routine
        dAngles = numpy.require(dAngles,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        n1 = numpy.require(n1,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        n2 = numpy.require(n2,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])

        # initialize return value (qposition) array
        qpos = numpy.empty(Npoints*3,dtype=numpy.double,order='C')
        qpos = numpy.require(qpos,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])

        libxrayutils.cang2q_areapixel( dAngles, qpos, n1, n2 ,_area_ri, Nd, Npoints, _detectorAxis_str,
                     _area_cch1, _area_cch2, _area_pwidth1/_area_distance, _area_pwidth2/_area_distance,
                     _area_detdir1, _area_detdir2, _area_tiltazimuth, _area_tilt, wl)

        #reshape output
        qpos.shape = (Npoints,3)
        return qpos[:,0],qpos[:,1],qpos[:,2]

    def afunc(param,x,detectorDir1,detectorDir2,r_i,detectorAxis,wl):
        """
        function for fitting the detector parameters
        basically this is a wrapper for the areapixel function

        parameters
        ----------
         param           fit parameters
                         (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)
         x               independent variables (contains angle1, angle2, n1, n2)
                         shape is (4,Npoints)

         detectorDir1:   direction of the detector (along the pixel direction 1);
                         e.g. 'z+' means higher pixel numbers at larger z positions
         detectorDir2:   direction of the detector (along the pixel direction 2); e.g. 'x+'

         r_i:            primary beam direction e.g. 'x+'
         detectorAxis:   list or tuple of detector circles
                         e.g. ['z+','y-'] would mean a detector arm with a two rotations
         wl:             wavelength of the experiment in Angstroem

        Returns
        -------
         reciprocal space position of detector pixels n1,n2 in a numpy.ndarray of shape
         ( 3, x.shape[1] )
        """

        angle1 = x[0,:]
        angle2 = x[1,:]
        n1 = x[2,:]
        n2 = x[3,:]

        # use only positive tilt
        param[5] = numpy.abs(param[5])

        (qx,qy,qz) = areapixel(param[:-1],detectorDir1,detectorDir2,r_i,detectorAxis,angle1,angle2,n1,n2,delta=[param[-1],0.],distance=1.,wl=wl)

        return qx**2+qy**2+qz**2

    Npoints = len(ang1)

    # guess initial parameters
    # center channel and detector pixel direction and pixel size
    (s1,i1,r1,dummy,dummy)=scipy.stats.linregress(ang1-start[3],n1)
    (s2,i2,r2,dummy,dummy)=scipy.stats.linregress(ang1-start[3],n2)
    (s3,i3,r3,dummy,dummy)=scipy.stats.linregress(ang2,n1)
    (s4,i4,r4,dummy,dummy)=scipy.stats.linregress(ang2,n2)

    s = ord('x') + ord('y') + ord('z')
    c1 = ord(detaxis[0][0]) + ord(r_i[0])
    c2 = ord(detaxis[1][0]) + ord(r_i[0])
    sign1 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[0]),math.getVector(r_i))))
    sign2 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[1]),math.getVector(r_i))))
    if r1**2>r2**2:
        cch1 = i1
        cch2 = i4
        pwidth1 = 2/numpy.abs(float(s1))*numpy.tan(numpy.radians(0.5))
        pwidth2 = 2/numpy.abs(float(s4))*numpy.tan(numpy.radians(0.5))
    else:
        cch1 = i3
        cch2 = i2
        pwidth1 = 2/numpy.abs(float(s3))*numpy.tan(numpy.radians(0.5))
        pwidth2 = 2/numpy.abs(float(s2))*numpy.tan(numpy.radians(0.5))

    tilt = numpy.abs(start[1])
    tiltazimuth = start[0]
    detrot = start[2]
    outerangle_offset = start[3]
    # parameters for the fitting
    param = (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)
    if debug:
        print("initial parameters: ")
        print("primary beam / detector pixel directions / distance: %s / %s %s / %e" %(r_i,detdir1,detdir2,1.))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" %param)


    # set data
    x = numpy.empty((4,Npoints),dtype=numpy.double)
    x[0,:] = ang1
    x[1,:] = ang2
    x[2,:] = n1
    x[3,:] = n2
    data = odr.Data(x,y=1)
    # define model for fitting
    model = odr.Model(afunc, extra_args=(detdir1,detdir2,r_i,detaxis,wl), implicit=True)
    # check if parameters need to be fixed
    ifixb = ()
    for i in range(len(fix)):
        ifixb += (int(not fix[i]),)

    my_odr = odr.ODR(data,model,beta0=param,ifixb=(1,1,1,1)+ifixb , ifixx =(0,0,0,0) ,stpb=(0.4,0.4,pwidth1/50.,pwidth2/50.,2,0.125,0.01,0.01), sclb=(1/numpy.abs(cch1),1/numpy.abs(cch2),1/pwidth1,1/pwidth2,1/90.,1/0.2,1/0.2,1/0.2) ,maxit=1000,ndigit=12, sstol=1e-11, partol=1e-11)
    if debug:
        my_odr.set_iprint(final=1)
        my_odr.set_iprint(iter=2)

    fit = my_odr.run()

    (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset) = fit.beta
    # fix things in parameters
    tiltazimuth = tiltazimuth%360.
    tilt = numpy.abs(tilt)

    final_q = afunc([cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset],x,detdir1,detdir2,r_i,detaxis,wl)
    final_error = numpy.mean(final_q)

    if False: # inactive code path
        if fig:
            plt.figure(fig.number)
        else:
            plt.figure("CCD Calib fit")
        plt.grid(True)
        plt.xlabel("Image number")
        plt.ylabel(r"|$\Delta$Q|")
        errp1, = plt.semilogy(afunc(my_odr.beta0,x,detdir1,detdir2,r_i,detaxis,wl),'x-',label='initial param')
        errp2, = plt.semilogy(afunc(fit.beta,x,detdir1,detdir2,r_i,detaxis,wl),'x-',label='param: %.1f %.1f %5.2g %5.2g %.1f %.2f %.3f %.3f'%(cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset))

    if debug:
        print("fitted parameters: (%d,%s) " %(fit.info,repr(fit.stopreason)))
        print("primary beam / detector pixel directions / distance: %s / %s %s / %g" %(r_i,detdir1,detdir2,1.))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" %(cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset))

    if full_output:
        return final_error,(cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset),fit
    else:
        return final_error

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
    (xrayutilities.math.Gauss2d)

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

