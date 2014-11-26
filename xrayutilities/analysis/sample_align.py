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
# Copyright (C) 2011,2013 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
functions to help with experimental alignment during experiments, especially
for experiments with linear and area detectors
"""

import re
import numpy
import numbers
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
from .. import cxrayutilities

try:
    from matplotlib import pyplot as plt
except ImportError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analysis.sample_align: warning; plotting functionality "
              "not available")

# regular expression to check goniometer circle syntax
circleSyntax = re.compile("[xyz][+-]")

#################################################
# channel per degree calculation
#################################################


def psd_chdeg(angles, channels, stdev=None, usetilt=True, plot=True,
              datap="kx", modelline="r--", modeltilt="b-", fignum=None,
              mlabel="fit", mtiltlabel="fit w/tilt", dlabel="data",
              figtitle=True):
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
     usetilt   whether to use model considering a detector tilt, i.e. deviation
               angle of the pixel direction from orthogonal to the primary beam
               (default: True)
     datap:    plot format of data points
     modelline:  plot format of modelline
     modeltilt:  plot format of modeltilt
     fignum:     figure number to use for the plot
     mlabel:     label of the model w/o tilt to be used in the plot
     mtiltlabel:  label of the model with tilt to be used in the plot
     dlabel:    label of the data line to be used in the plot
     figtitle:  boolean to tell if the figure title should show the fit
                parameters

    Returns
    -------
     (pixelwidth,centerch,tilt)

    pixelwidth:  the width of one detector channel @ 1m distance, which is
                 negative in case the hit channel number decreases upon an
                 increase of the detector angle.
    centerch: center channel of the detector
    tilt: tilt of the detector from perpendicular to the beam
          (will be zero in case of usetilt=False)

    Note:
     L/pixelwidth*pi/180 = channel/degree for large detector distance with the
     sample detector disctance L
    """

    if stdev is None:
        stdevu = numpy.ones(len(channels))
    else:
        stdevu = stdev

    # define detector model and other functions needed for the tilt
    def straight_tilt(p, x):
        """
        model for straight-linear detectors including tilt

        Parameters
        ----------
         p ... [L/w_pix*pi/180 ~= channel/degree, center_channel,
                detector_tilt] with L sample detector disctance, and
               w_pix the width of one detector channel
         x ... independent variable of the model: detector angle (degree)
        """
        rad = numpy.radians(x)
        r = numpy.degrees(p[0]) * numpy.sin(rad) / \
            numpy.cos(rad - numpy.radians(p[2])) + p[1]
        return r

    def straight_tilt_der_x(p, x):
        """
        derivative of straight-linear detector model with respect to the angle
        for parameter description see straigt_tilt
        """
        rad = numpy.radians(x)
        p2 = numpy.radians(p[2])
        r = numpy.degrees(p[0]) * \
            (numpy.cos(rad) / numpy.cos(rad - p2) +
             numpy.sin(rad) / numpy.cos(rad - p2) ** 2 * numpy.sin(rad - p2))
        return r

    def straight_tilt_der_p(p, x):
        """
        derivative of straight-linear detector model with respect to the
        paramters for parameter description see straigt_tilt
        """
        rad = numpy.radians(x)
        p2 = numpy.radians(p[2])
        r = numpy.concatenate([180. / numpy.pi * numpy.sin(rad) /
                               numpy.cos(rad - p2),
                               numpy.ones(x.shape, dtype=numpy.float),
                               - numpy.degrees(p[0]) * numpy.sin(rad) /
                               numpy.cos(rad - p2) ** 2 *
                               numpy.sin(rad - p2)])
        r.shape = (3,) + x.shape
        return r

    # fit linear
    model = models.unilinear
    data = odr.RealData(angles, channels, sy=stdevu)
    my_odr = odr.ODR(data, model)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fitlin = my_odr.run()

    # fit linear with tangens angle
    model = models.unilinear
    data = odr.RealData(numpy.degrees(numpy.tan(numpy.radians(angles))),
                        channels, sy=stdevu)
    my_odr = odr.ODR(data, model)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fittan = my_odr.run()

    if usetilt:
        # fit tilted straight detector model
        model = odr.Model(straight_tilt, fjacd=straight_tilt_der_x,
                          fjacb=straight_tilt_der_p)
        data = odr.RealData(angles, channels, sy=stdevu)
        my_odr = odr.ODR(data, model, beta0=[fittan.beta[0],
                                             fittan.beta[1], 0])
        # fit type 2 for least squares
        my_odr.set_job(fit_type=2)
        fittilt = my_odr.run()

    try:
        plt.__name__
    except NameError:
        print("XU.analyis.psd_chdeg: Warning: plot functionality not "
              "available")
        plot = False

    if plot:
        markersize = 6.0
        markeredgewidth = 1.5
        linewidth = 2.0
        if fignum is None:
            plt.figure()
        else:
            plt.figure(fignum)
        # first plot to show linear model
        ax1 = plt.subplot(211)
        angr = angles.max() - angles.min()
        angp = numpy.linspace(angles.min() - angr * 0.1,
                              angles.max() + angr * .1, 1000)
        if modelline:
            plt.plot(angp, models._unilin(fittan.beta,
                     numpy.degrees(numpy.tan(numpy.radians(angp)))),
                     modelline, label=mlabel, lw=linewidth)
        plt.plot(angp, models._unilin(fitlin.beta, angp), 'k-', label='')
        if usetilt:
            plt.plot(angp, straight_tilt(fittilt.beta, angp),
                     modeltilt, label=mtiltlabel, lw=linewidth)
        if stdev is None:
            plt.plot(angles, channels, datap, ms=markersize,
                     mew=markeredgewidth, mec=datap[0],
                     mfc='none', label=dlabel)
        else:
            plt.errorbar(angles, channels, fmt=datap, yerr=stdevu,
                         ms=markersize, mew=markeredgewidth, mec=datap[0],
                         mfc='none', label=dlabel, ecolor='0.5')
        plt.grid(True)
        leg = plt.legend(numpoints=1)
        leg.get_frame().set_alpha(0.8)

        plt.ylabel("channel number")

        # lower plot to show deviations from linear model
        ax2 = plt.subplot(212, sharex=ax1)
        if modelline:
            plt.plot(angp, models._unilin(fittan.beta,
                     numpy.degrees(numpy.tan(numpy.radians(angp))))
                     - models._unilin(fitlin.beta, angp),
                     modelline, label=mlabel, lw=linewidth)
        if usetilt:
            plt.plot(angp, straight_tilt(fittilt.beta, angp)
                     - models._unilin(fitlin.beta, angp),
                     modeltilt, label=mtiltlabel, lw=linewidth)
        if stdev is None:
            plt.plot(angles, channels - models._unilin(fitlin.beta, angles),
                     datap, ms=markersize, mew=markeredgewidth, mec=datap[0],
                     mfc='none', label=dlabel)
        else:
            plt.errorbar(angles,
                         channels - models._unilin(fitlin.beta, angles),
                         fmt=datap, yerr=stdevu, ms=markersize,
                         mew=markeredgewidth, mec=datap[0], mfc='none',
                         label=dlabel, ecolor='0.5')
        plt.xlabel("detector angle (deg)")
        plt.ylabel("ch. num. - linear trend")
        plt.grid(True)
        plt.hlines(0, angp.min(), angp.max())

        if figtitle:
            if usetilt:
                plt.suptitle(
                    "L/w*pi/180: %8.2f; center channel: %8.2f; tilt: %5.2fdeg"
                    % (fittilt.beta[0], fittilt.beta[1], fittilt.beta[2]))
            else:
                plt.suptitle("L/w*pi/180: %8.2f; center channel: %8.2f"
                             % (fittan.beta[0], fittan.beta[1]))

    if usetilt:
        fit = fittilt
    else:
        fit = fittan

    if config.VERBOSITY >= config.INFO_LOW:
        if usetilt:
            print("XU.analysis.psd_chdeg:  channelwidth@1m / center channel /"
                  " tilt: %8.4e / %8.2f / %6.3fdeg"
                  % (numpy.abs(1 / numpy.degrees(fit.beta[0])),
                     fit.beta[1], fit.beta[2]))
            print("XU.analysis.psd_chdeg:  error of channelwidth / "
                  "center channel / tilt: %8.4e / %8.3f / %6.3fdeg"
                  % (numpy.radians(fit.sd_beta[0] / fit.beta[0] ** 2),
                     fit.sd_beta[1], fit.sd_beta[2]))
        else:
            print("XU.analysis.psd_chdeg:  channelwidth@1m / center channel: "
                  "%8.4e / %8.2f"
                  % (1 / numpy.degrees(fit.beta[0]), fit.beta[1]))
            print("XU.analysis.psd_chdeg:  error of channelwidth / "
                  "center channel: %8.4e / %8.3f"
                  % (numpy.radians(fit.sd_beta[0] / fit.beta[0] ** 2),
                     fit.sd_beta[1]))

    if usetilt:
        return (1. / numpy.degrees(fit.beta[0]), fit.beta[1], fit.beta[2])
    else:
        return (1. / numpy.degrees(fit.beta[0]), fit.beta[1], 0.)


#################################################
# channel per degree calculation from scan with
# linear detector (determined maximum by peak_fit)
#################################################
def linear_detector_calib(angle, mca_spectra, **keyargs):
    """
    function to calibrate the detector distance/channel per degrees
    for a straight linear detector mounted on a detector arm

    Parameters
    ----------
     angle:         array of angles in degree of measured detector spectra
     mca_spectra :  corresponding detector spectra
                    (shape: (len(angle), Nchannels)

    keyword arguments:
     r_i:           primary beam direction as vector [xyz][+-]; default: 'y+'
     detaxis:       detector arm rotation axis [xyz][+-]; default: 'x+'

    other options are passed to psd_chdeg function, options include:
     plot:     flag to specify if a visualization of the fit should be done
     usetilt:  whether to use model considering a detector tilt, i.e.
               deviation angle of the pixel direction from orthogonal to the
               primary beam) (default: True)

    Note:  see help of psd_chdeg for more options

    Returns
    -------
     pixelwidth (at one meter distance), center_channel[, detector_tilt]

    Note:  L/pixelwidth*pi/180 ~= channel/degree, with the sample detector
           distance L

    pixelwidth is negative in case the hit channel number decreases upon an
    increase of the detector angle The function also prints out how a linear
    detector can be initialized using the results obtained from this
    calibration. Carefully check the results
    """

    if "detaxis" in keyargs:
        detrotaxis = keyargs["detaxis"]
        keyargs.pop("detaxis")
    else:  # use default
        detrotaxis = 'x+'
    if "r_i" in keyargs:
        r_i = keyargs["r_i"]
        keyargs.pop("r_i")
    else:  # use default
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
        # print(i)
        row = mca_spectra[i, :]
        row_int = row.sum()
        # print(row_int)
        if ((numpy.abs(row_int - mca_avg) > 3 * mca_std) or
                (row_int - mca_rowmax * 0.7 < 0)):
            if config.VERBOSITY >= config.DEBUG:
                print("XU.analysis.linear_detector_calib: spectrum #%d "
                      "out of intensity range -> ignored" % i)
            nignored += 1
            continue

        maxp = numpy.argmax(row)
        fwhm = fwhm_exp(numpy.arange(row.size), row)
        N = int(7 * numpy.ceil(fwhm)) // 2 * 2

        # fit beam position
        # determine maximal usable length of array around peak position
        Nuse = min(maxp + N // 2, len(row) - 1) - max(maxp - N // 2, 0)
        # print("%d %d %d"%(N,max(maxp-N//2,0),min(maxp+N//2,len(row)-1)))
        param, perr, itlim = math.peak_fit(
            numpy.arange(Nuse),
            row[max(maxp - N // 2, 0):min(maxp + N // 2, len(row) - 1)],
            peaktype='PseudoVoigt')
        if param[0] > 0 and param[0] < Nuse and perr[0] < Nuse/2.: 
            param[0] += max(maxp - N // 2, 0)
            pos.append(param[0])
            posstd.append(perr[0])
            ang.append(angle[i])

    ang = numpy.array(ang)
    pos = numpy.array(pos)
    posstd = numpy.array(posstd)
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analysis.linear_detector_calib: using %d out of %d given "
              "spectra." % (len(ang), len(angle)))
    if config.VERBOSITY >= config.DEBUG:
        print("XU.analysis.linear_detector_calib: determined peak positions:")
        print(zip(pos,posstd))

    detparam = psd_chdeg(ang, pos, stdev=posstd, **keyargs)
    if numpy.sign(detparam[0]) > 0:
        sign = '-'
    else:
        sign = '+'

    detaxis = '  '
    detd = numpy.cross(math.getVector(detrotaxis), math.getVector(r_i))
    argm = numpy.abs(detd).argmax()

    def flipsign(char, val):
        if numpy.sign(val) < 0:
            if char == '+':
                return '-'
            else:
                return '+'
        else:
            return char

    if argm == 0:
        detaxis = 'x' + flipsign(sign, detd[argm])
    elif argm == 1:
        detaxis = 'y' + flipsign(sign, detd[argm])
    elif argm == 2:
        detaxis = 'z' + flipsign(sign, detd[argm])

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.linear_detector_calib:\n\tused/total spectra: %d/%d"
              % (mca_spectra.shape[0] - nignored, mca_spectra.shape[0]))
        print("\tdetector rotation axis (given by user/default input): %s"
              % detrotaxis)
        if len(detparam) == 3:
            tilt = detparam[2]
        else:
            tilt = 0
        print("\tdetector initialization with: init_linear('%s',%.2f,%d"
              ",pixelwidth=%.4e,distance=1.,tilt=%.2f)"
              % (detaxis, numpy.abs(detparam[1]), mca_spectra.shape[1],
                 numpy.abs(detparam[0]), tilt))

    return detparam

######################################################
# detector parameter calculation from scan with
# area detector (determine maximum by center of mass)
######################################################


def area_detector_calib(angle1, angle2, ccdimages, detaxis, r_i, plot=True,
                        cut_off=0.7, start=(0, 0, 0, 0),
                        fix=(False, False, False, False),
                        fig=None, wl=None, plotlog=False, debug=False):
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
        plot .... flag to determine if results and intermediate results should
                  be plotted; default: True
        cut_off . cut off intensity to decide if image is used for the
                  determination or not; default: 0.7 = 70%
        start ... sequence of start values of the fit for parameters, which can
                  not be estimated automatically. these are:
                  tiltazimuth,tilt,detector_rotation,outerangle_offset. By
                  default (0,0,0,0) is used.
        fix ..... fix parameters of start (default: (False,False,False,False))
        fig ..... matplotlib figure used for plotting the error
                  default: None (creates own figure)
        wl ...... wavelength of the experiment in Angstrom (default:
                  config.WAVELENGTH) value does not really matter here but does
                  affect the scaling of the error
        plotlog . flag to specify if the created error plot should be on
                  log-scale
        debug ... flag to specify that you want to see verbose output and
                  saving of images to show if the CEN determination works

    """

    if plot:
        try:
            plt.__name__
        except NameError:
            print("XU.analyis.area_detector_calib: Warning: plot functionality"
                  " not available")
            plot = False

    if wl is None:
        wl = config.WAVELENGTH
    else:
        wl = utilities.wavelength(wl)

    t0 = time.time()
    Npoints = len(angle1)
    if debug:
        print("number of given images: %d" % Npoints)

    # determine center of mass position from detector images
    # also use only images with an intensity larger than 70% of the average
    # intensity
    n1 = numpy.zeros(0, dtype=numpy.double)
    n2 = n1
    ang1 = n1
    ang2 = n1

    avg = 0
    for i in range(Npoints):
        avg += numpy.sum(ccdimages[i])
    avg /= float(Npoints)
    (N1, N2) = ccdimages[0].shape

    if debug:
        print("average intensity per image: %.1f" % avg)

    for i in range(Npoints):
        img = ccdimages[i]
        if numpy.sum(img) > cut_off * avg:
            [cen1, cen2] = center_of_mass(img)
            if debug:
                plt.figure("_ccd")
                plt.imshow(utilities.maplog(img), origin='low')
                plt.plot(cen2, cen1, "wo", mfc='none')
                plt.axis([cen2 - 25, cen2 + 25, cen1 - 25, cen1 + 25])
                plt.savefig("xu_calib_ccd_img%d.png" % i)
                plt.close("_ccd")

            n1 = numpy.append(n1, cen1)
            n2 = numpy.append(n2, cen2)
            ang1 = numpy.append(ang1, angle1[i])
            ang2 = numpy.append(ang2, angle2[i])
            # if debug:
            #     print("%8.3f %8.3f \t%.2f %.2f" % (angle1[i], angle2[i],
            #           cen1, cen2))
    Nused = len(ang1)

    if debug:
        print("Nused / Npoints: %d / %d" % (Nused, Npoints))

    # determine detector directions
    detdir1, detdir2 = _determine_detdir(
        ang1 - start[3], ang2, n1, n2, detaxis, r_i)

    epslist = []
    paramlist = []
    epsmin = numpy.inf
    fitmin = None

    print("tiltaz   tilt   detrot   offset:  error (relative) (fittime)")
    print("------------------------------------------------------------")
    # find optimal detector rotation (however keep other parameters free)
    detrot = start[2]
    if not fix[2]:
        for detrotstart in numpy.linspace(start[2] - 1, start[2] + 1, 20):
            start = start[:2] + (detrotstart,) + (start[3],)
            eps, param, fit = _area_detector_calib_fit(
                ang1, ang2, n1, n2, detaxis, r_i, detdir1, detdir2,
                start=start, fix=fix, full_output=True, wl=wl)
            epslist.append(eps)
            paramlist.append(param)
            if epslist[-1] < epsmin:
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
        Ntilt = Ntilt * 5 if not fix[1] else Ntilt
        Ntiltaz = Ntiltaz * 5 if not fix[0] else Ntiltaz

    startparam = start[:2] + (detrot,) + (start[3],)

    for tiltazimuth in numpy.linspace(startparam[0] if fix[0] else 0,
                                      360, Ntiltaz, endpoint=False):
        for tilt in numpy.linspace(startparam[1] if fix[1] else 0, 4, Ntilt):
            for offset in numpy.linspace(
                    startparam[3] if fix[3] else - 3 + startparam[3],
                    3 + startparam[3], Noffset):
                t1 = time.time()
                start = (tiltazimuth, tilt, detrot, offset)
                eps, param, fit = _area_detector_calib_fit(
                    ang1, ang2, n1, n2, detaxis, r_i, detdir1, detdir2,
                    start=start, fix=fix, full_output=True, wl=wl)
                epslist.append(eps)
                paramlist.append(param)
                t2 = time.time()
                print("%6.1f %6.2f %8.3f %8.3f: %10.4e (%4.2f) (%5.2fsec)" %
                      (start + (epslist[-1], epslist[-1] / epsmin, t2 - t1)))

                if epslist[-1] < epsmin:
                    print("************************")
                    print("new global minimum found")
                    epsmin = epslist[-1]
                    parammin = param
                    fitmin = fit
                    print("new best parameters: %.2f %.2f %10.4e %10.4e "
                          "%.1f %.2f %.3f %.3f" % parammin)
                    print("************************\n")

    (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
        outerangle_offset) = parammin

    if plot:
        if fig:
            plt.figure(fig.number)
        else:
            plt.figure("CCD Calib fit")
        nparams = numpy.array(paramlist)
        neps = numpy.array(epslist)
        labels = (
            'cch1 (1)',
            'cch2 (1)',
            r'pwidth1 ($\mu$m@1m)',
            'pwidth2 ($\mu$m@1m)',
            'tiltazimuth (deg)',
            'tilt (deg)',
            'detrot (deg)',
            'outerangle offset (deg)')
        xscale = (1., 1., 1.e6, 1.e6, 1., 1., 1., 1.)
        # plt.suptitle("best fit: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f"
        #              % (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
        #                 detrot, outerangle_offset))
        for p in range(8):
            plt.subplot(3, 3, p + 1)
            if plotlog:
                plt.semilogy(nparams[:, p] * xscale[p], neps, 'k.')
            else:
                plt.scatter(nparams[:, p] * xscale[p], neps, c=nparams[:, -1],
                            s=10, marker='o', cmap=plt.cm.gnuplot,
                            edgecolor='none')
            plt.xlabel(labels[p])

        for p in range(8):
            plt.subplot(3, 3, p + 1)
            if plotlog:
                plt.semilogy(parammin[p] * xscale[p], epsmin, 'ko', ms=8,
                             mew=2.5, mec='k', mfc='w')
            else:
                plt.plot(parammin[p] * xscale[p], epsmin, 'ko', ms=8, mew=2.5,
                         mec='k', mfc='w')
                plt.ylim(epsmin * 0.7, epsmin * 2.)
            plt.locator_params(nbins=4, axis='x')
        plt.tight_layout()

    if config.VERBOSITY >= config.INFO_LOW:
        print("total time needed for fit: %.2fsec" % (time.time() - t0))
        print("fitted parameters: epsilon: %10.4e (%d,%s) "
              % (epsmin, fitmin.info, repr(fitmin.stopreason)))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,"
              "outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f"
              % (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
                 outerangle_offset))

    if config.VERBOSITY > 0:
        print("please check the resulting data (consider setting plot=True)")
        print("detector rotation axis / primary beam direction "
              "(given by user): %s / %s" % (repr(detaxis), r_i))
        print("detector pixel directions / distance: %s %s / %g"
              % (detdir1, detdir2, 1.))
        print("\tdetector initialization with: init_area('%s', '%s', "
              "cch1=%.2f, cch2=%.2f, Nch1=%d, Nch2=%d, pwidth1=%.4e, "
              "pwidth2=%.4e, distance=1., detrot=%.3f, tiltazimuth=%.1f, "
              "tilt=%.3f)" % (detdir1, detdir2, cch1, cch2, N1, N2, pwidth1,
                              pwidth2, detrot, tiltazimuth, tilt))
        print("AND ALWAYS USE an (additional) OFFSET of %.4fdeg in the "
              "OUTER DETECTOR ANGLE!" % (outerangle_offset))

    return (cch1, cch2, pwidth1, pwidth2, tiltazimuth,
            tilt, detrot, outerangle_offset), eps


def _determine_detdir(ang1, ang2, n1, n2, detaxis, r_i):
    """
    determines detector pixel direction from correlation analysis of linear
    fits to the observed pixel numbers of the primary beam.
    """
    debug = False
    # center channel and detector pixel direction and pixel size
    (s1, i1, r1, dummy, dummy) = scipy.stats.linregress(ang1, n1)
    (s2, i2, r2, dummy, dummy) = scipy.stats.linregress(ang1, n2)
    (s3, i3, r3, dummy, dummy) = scipy.stats.linregress(ang2, n1)
    (s4, i4, r4, dummy, dummy) = scipy.stats.linregress(ang2, n2)
    if debug:
        print("%.2f %.2f %.2f %.2f" % (s1, s2, s3, s4))
        print("%.2f %.2f %.2f %.2f" % (r1, r2, r3, r4))
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(ang1, n1, 'bx', label='channel 1')
            plt.plot(ang1, n2, 'rx', label='channel 2')
            plt.legend()
            plt.xlabel('angle 1')
            plt.subplot(212)
            plt.plot(ang2, n1, 'bx', label='channel 1')
            plt.plot(ang2, n2, 'rx', label='channel 2')
            plt.legend()
            plt.xlabel('angle 2')

    # determine detector directions
    s = ord('x') + ord('y') + ord('z')
    c1 = ord(detaxis[0][0]) + ord(r_i[0])
    c2 = ord(detaxis[1][0]) + ord(r_i[0])
    sign1 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[0]),
                                             math.getVector(r_i))))
    sign2 = numpy.sign(numpy.sum(numpy.cross(math.getVector(detaxis[1]),
                                             math.getVector(r_i))))
    if r1 ** 2 > r2 ** 2:
        detdir1 = chr(s - c1)
        detdir2 = chr(s - c2)
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
        detdir1 = chr(s - c2)
        detdir2 = chr(s - c1)
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

    return detdir1, detdir2


def _area_detector_calib_fit(ang1, ang2, n1, n2, detaxis, r_i, detdir1,
                             detdir2, start=(0, 0, 0, 0),
                             fix = (False, False, False, False),
                             full_output=False, wl=1., debug=False):
    """
    INTERNAL FUNCTION
    function to calibrate the detector parameters of an area detector
    it determines the detector tilt possible rotations and offsets in the
    detector arm angles

    parameters
    ----------
     angle1 ..... outer detector arm angle
     angle2 ..... inner detector arm angle
     n1,n2 ...... pixel number at which the primary beam was observed
     detaxis .... detector arm rotation axis
                  default: ['z+','y-']
     detdir1,2 .. detector pixel directions of first and second pixel
                  coordinates; e.g. 'y+'
     r_i ........ primary beam direction [xyz][+-]; default 'x+'

    keyword_arguments:
        start ... sequence of start values of the fit for parameters,
                  which can not be estimated automatically
                  these are:
                  tiltazimuth,tilt,detector_rotation,outerangle_offset.
                  By default: (0,0,0,0)
        fix ..... fix parameters of start (default: (False,False,False,False))
        full_output   flag to tell if to return fit object with final
                      parameters and detector directions
        wl ...... wavelength of the experiment in Angstrom (default: 1)
                  value does not matter here and does only affect the scaling
                  of the error
        debug ... flag to tell if you want to see debug output of the script
                 (switch this to true only if you can handle it :))

    returns
    -------
        eps   final epsilon of the fit

    if full_output:
        eps,param,fit
    """

    def areapixel(params, detectorDir1, detectorDir2, r_i, detectorAxis,
                  *args, **kwargs):
        """
        angular to momentum space conversion for pixels of an area detector the
        center pixel is in direction of self.r_i when detector angles are zero

        the detector geometry must be given to the routine

        Parameters
        ----------
        *args:          detector angles and channel numbers
                dAngles as numpy array, lists or Scalars in total
                        len(detectorAxis) must be given starting with the outer
                        most circle all arguments must have the same shape or
                        length
                channel numbers n1 and n2 where the primary beam hits the
                        detector same length as the detector values

        params:         parameters of the detector calibration model
                        (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot)
                cch1,2: center pixel, in direction of self.r_i at zero
                        detectorAngles
                pwidth1,2: width of one pixel (same unit as distance)
                tiltazimuth: direction of the tilt vector in the detector plane
                             (in degree)
                tilt: tilt of the detector plane around an axis normal to the
                      direction given by the tiltazimuth
                detrot: detector rotation around the primary beam direction as
                        given by r_i

        detectorDir1:   direction of the detector (along the pixel
                        direction 1); e.g. 'z+' means higher pixel numbers at
                        larger z positions
        detectorDir2:   direction of the detector (along the pixel
                        direction 2); e.g. 'x+'

        r_i:            primary beam direction e.g. 'x+'
        detectorAxis:   list or tuple of detector circles
                        e.g. ['z+','y-'] would mean a detector arm with a two
                        rotations

        **kwargs:       possible keyword arguments
            delta:      giving delta angles to correct the given ones for
                        misalignment delta must be an numpy array or list of
                        len(*dAngles) used angles are than *args - delta
            wl:         x-ray wavelength in angstroem (default: 1 (since it
                        does not matter here))
            distance:   distance of center pixel from center of rotation
                        (default: 1. (since it does not matter here))
            deg:        flag to tell if angles are passed as degree
                        (default: True)

        Returns
        -------
        reciprocal space position of detector pixels n1,n2 in a numpy.ndarray
        of shape ( len(args) , 3 )
        """

        # check detector circle argument
        if isinstance(detectorAxis, (str, list, tuple)):
            if isinstance(detectorAxis, str):
                dAxis = list([detectorAxis])
            else:
                dAxis = list(detectorAxis)
            for circ in dAxis:
                if not isinstance(circ, str) or len(circ) != 2:
                    raise InputError("QConversionPixel: incorrect detector "
                                     "circle type or syntax (%s)" % repr(circ))
                if not circleSyntax.search(circ):
                    raise InputError("QConversionPixel: incorrect detector "
                                     "circle syntax (%s)" % circ)
        else:
            raise TypeError("Qconversion error: invalid type for detectorAxis,"
                            " must be str, list or tuple")
        # add detector rotation around primary beam
        dAxis += [r_i]
        _detectorAxis_str = ''
        for circ in dAxis:
            _detectorAxis_str += circ

        Nd = len(dAxis)
        Nargs = Nd + 2 - 1

        # check detectorDir
        if not isinstance(detectorDir1, str) or len(detectorDir1) != 2:
            raise InputError("QConversionPixel: incorrect detector direction1 "
                             "type or syntax (%s)" % repr(detectorDir1))
        if not circleSyntax.search(detectorDir1):
            raise InputError("QConversionPixel: incorrect detector direction1 "
                             "syntax (%s)" % detectorDir1)
        _area_detdir1 = detectorDir1
        if not isinstance(detectorDir2, str) or len(detectorDir2) != 2:
            raise InputError("QConversionPixel: incorrect detector direction2 "
                             "type or syntax (%s)" % repr(detectorDir2))
        if not circleSyntax.search(detectorDir2):
            raise InputError("QConversionPixel: incorrect detector direction2 "
                             "syntax (%s)" % detectorDir2)
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
            delta = numpy.array(kwargs['delta'], dtype=numpy.double)
            if delta.size != Nd - 1:
                raise InputError("QConversionPixel: keyword argument delta "
                                 "does not have an appropriate shape")
        else:
            delta = numpy.zeros(Nd)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Nargs:
            raise InputError("QConversionPixel: wrong amount (%d) of arguments"
                             " given, number of arguments should be %d"
                             % (len(args), Nargs))

        try:
            Npoints = len(args[0])
        except (TypeError, IndexError):
            Npoints = 1

        dAngles = numpy.array((), dtype=numpy.double)
        for i in range(Nd - 1):
            arg = args[i]
            if not isinstance(arg, (numbers.Number, tuple,
                                    list, numpy.ndarray)):
                raise TypeError("QConversionPixel: invalid type for one of "
                                "the detector coordinates, must be scalar, "
                                "list or array")
            elif isinstance(arg, numbers.Number):
                arg = numpy.array([arg], dtype=numpy.double)
            elif isinstance(arg, list):
                arg = numpy.array(arg, dtype=numpy.double)
            arg = arg - delta[i]
            dAngles = numpy.concatenate((dAngles, arg))
        # add detector rotation around primary beam
        dAngles = numpy.concatenate((dAngles,
                                     numpy.ones(arg.shape, dtype=numpy.double)
                                     * _area_rot))

        # read channel numbers
        n1 = numpy.array((), dtype=numpy.double)
        n2 = numpy.array((), dtype=numpy.double)

        arg = args[Nd - 1]
        if not isinstance(arg, (numbers.Number, tuple, list, numpy.ndarray)):
            raise TypeError("QConversionPixel: invalid type for one of the "
                            "detector coordinates, must be scalar, list or "
                            "array")
        elif isinstance(arg, numbers.Number):
            arg = numpy.array([arg], dtype=numpy.double)
        elif isinstance(arg, list):
            arg = numpy.array(arg, dtype=numpy.double)
        n1 = arg

        arg = args[Nd]
        if not isinstance(arg, (numbers.Number, tuple, list, numpy.ndarray)):
            raise TypeError("QConversionPixel: invalid type for one of the "
                            "detector coordinates, must be scalar, list or "
                            "array")
        elif isinstance(arg, numbers.Number):
            arg = numpy.array([arg], dtype=numpy.double)
        elif isinstance(arg, list):
            arg = numpy.array(arg, dtype=numpy.double)
        n2 = arg

        dAngles.shape = (Nd, Npoints)
        dAngles = dAngles.transpose()

        if deg:
            dAngles = numpy.radians(dAngles)

        qpos = cxrayutilities.ang2q_conversion_area_pixel(
            dAngles, n1, n2, _area_ri, _detectorAxis_str,
            _area_cch1, _area_cch2, _area_pwidth1 / _area_distance,
            _area_pwidth2 / _area_distance, _area_detdir1, _area_detdir2,
            _area_tiltazimuth, _area_tilt, wl, config.NTHREADS)

        return qpos[:, 0], qpos[:, 1], qpos[:, 2]

    def afunc(param, x, detectorDir1, detectorDir2, r_i, detectorAxis, wl):
        """
        function for fitting the detector parameters
        basically this is a wrapper for the areapixel function

        parameters
        ----------
         param           fit parameters
                         (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                          detrot, outerangle_offset)
         x               independent variables
                         (angle1, angle2, n1, n2) with shape (4, Npoints)
         detectorDir1:   direction of the detector (along the pixel
                         direction 1); e.g. 'z+' means higher pixel numbers at
                         larger z positions
         detectorDir2:   direction of the detector (along the pixel
                         direction 2); e.g. 'x+'
         r_i:            primary beam direction e.g. 'x+'
         detectorAxis:   list or tuple of detector circles
                         e.g. ['z+', 'y-'] would mean a detector arm with a two
                         rotations
         wl:             wavelength of the experiment in Angstroem

        Returns
        -------
         reciprocal space position of detector pixels n1,n2 in a numpy.ndarray
         of shape (3, x.shape[1])
        """

        angle1 = x[0, :]
        angle2 = x[1, :]
        n1 = x[2, :]
        n2 = x[3, :]

        # use only positive tilt
        param[5] = numpy.abs(param[5])

        (qx, qy, qz) = areapixel(
            param[:-1], detectorDir1, detectorDir2, r_i, detectorAxis,
            angle1, angle2, n1, n2, delta=[param[-1], 0.], distance=1., wl=wl)

        return qx ** 2 + qy ** 2 + qz ** 2

    Npoints = len(ang1)

    # guess initial parameters
    # center channel and detector pixel direction and pixel size
    (s1, i1, r1, dummy, dummy) = scipy.stats.linregress(ang1 - start[3], n1)
    (s2, i2, r2, dummy, dummy) = scipy.stats.linregress(ang1 - start[3], n2)
    (s3, i3, r3, dummy, dummy) = scipy.stats.linregress(ang2, n1)
    (s4, i4, r4, dummy, dummy) = scipy.stats.linregress(ang2, n2)

    if r1 ** 2 > r2 ** 2:
        cch1 = i1
        cch2 = i4
        pwidth1 = 2 / numpy.abs(float(s1)) * numpy.tan(numpy.radians(0.5))
        pwidth2 = 2 / numpy.abs(float(s4)) * numpy.tan(numpy.radians(0.5))
    else:
        cch1 = i3
        cch2 = i2
        pwidth1 = 2 / numpy.abs(float(s3)) * numpy.tan(numpy.radians(0.5))
        pwidth2 = 2 / numpy.abs(float(s2)) * numpy.tan(numpy.radians(0.5))

    tilt = numpy.abs(start[1])
    tiltazimuth = start[0]
    detrot = start[2]
    outerangle_offset = start[3]
    # parameters for the fitting
    param = (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
             detrot, outerangle_offset)
    if debug:
        print("initial parameters: ")
        print("primary beam / detector pixel directions / distance: "
              "%s / %s %s / %e" % (r_i, detdir1, detdir2, 1.))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,"
              "outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f" % param)

    # set data
    x = numpy.empty((4, Npoints), dtype=numpy.double)
    x[0, :] = ang1
    x[1, :] = ang2
    x[2, :] = n1
    x[3, :] = n2
    data = odr.Data(x, y=1)
    # define model for fitting
    model = odr.Model(afunc, extra_args=(detdir1, detdir2, r_i, detaxis, wl),
                      implicit=True)
    # check if parameters need to be fixed
    ifixb = ()
    for i in range(len(fix)):
        ifixb += (int(not fix[i]),)

    my_odr = odr.ODR(data, model, beta0=param, ifixb=(1, 1, 1, 1) + ifixb,
                     ifixx =(0, 0, 0, 0), stpb=(0.4, 0.4, pwidth1 / 50.,
                     pwidth2 / 50., 2, 0.125, 0.01, 0.01),
                     sclb=(1 / numpy.abs(cch1), 1 / numpy.abs(cch2),
                     1 / pwidth1, 1 / pwidth2, 1 / 90., 1 / 0.2, 1 / 0.2,
                     1 / 0.2), maxit=1000, ndigit=12, sstol=1e-11,
                     partol=1e-11)
    if debug:
        my_odr.set_iprint(final=1)
        my_odr.set_iprint(iter=2)

    fit = my_odr.run()

    (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
        outerangle_offset) = fit.beta
    # fix things in parameters
    tiltazimuth = tiltazimuth % 360.
    tilt = numpy.abs(tilt)

    final_q = afunc([cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                     detrot, outerangle_offset],
                    x, detdir1, detdir2, r_i,
                    detaxis, wl)
    final_error = numpy.mean(final_q)

    if False:  # inactive code path
        if fig:
            plt.figure(fig.number)
        else:
            plt.figure("CCD Calib fit")
        plt.grid(True)
        plt.xlabel("Image number")
        plt.ylabel(r"|$\Delta$Q|")
        errp1, = plt.semilogy(afunc(my_odr.beta0, x, detdir1, detdir2, r_i,
                                    detaxis, wl),
                              'x-', label='initial param')
        errp2, = plt.semilogy(afunc(fit.beta, x, detdir1, detdir2, r_i,
                                    detaxis, wl),
                              'x-', label='param: %.1f %.1f %5.2g %5.2g %.1f '
                              '%.2f %.3f %.3f'
                              % (cch1, cch2, pwidth1, pwidth2, tiltazimuth,
                                 tilt, detrot, outerangle_offset))

    if debug:
        print("fitted parameters: (%d,%s) " % (fit.info, repr(fit.stopreason)))
        print("primary beam / detector pixel directions / distance: "
              "%s / %s %s / %g" % (r_i, detdir1, detdir2, 1.))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,"
              "outerangle_offset)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f"
              % (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
                 outerangle_offset))

    if full_output:
        return final_error, (cch1, cch2, pwidth1, pwidth2,
                             tiltazimuth, tilt, detrot, outerangle_offset), fit
    else:
        return final_error


# #####################################################
#  detector parameter calculation from scan with
#  area detector (determine maximum by center of mass)
# #####################################################
def area_detector_calib_hkl(sampleang, angle1, angle2, ccdimages, hkls,
                            experiment, material, detaxis, r_i, plot=True,
                            cut_off=0.1, start=(0, 0, 0, 0, 0, 0, 'config'),
                            fix=(False, False, False, False, False, False,
                                 False),
                            fig=None, plotlog=False, debug=False):
    """
    function to calibrate the detector parameters of an area detector
    it determines the detector tilt possible rotations and offsets in the
    detector arm angles

    in this variant not only scans through the primary beam but also scans at a
    set of symmetric reflections can be used for the detector parameter
    determination. for this not only the detector parameters but in addition
    the sample orientation and wavelength need to be fit.  Both images from the
    primary beam hkl = (0,0,0) and from a symmetric reflection hkl = (h,k,l)
    need to be given for a successful run.

    parameters
    ----------
     sampleang .. sample rocking angle (needed to align the reflections (same
                  rotation direction as inner detector rotation)) other sample
                  angle are not allowed to be changed during the scans
     angle1 ..... outer detector arm angle
     angle2 ..... inner detector arm angle
     ccdimages .. images of the ccd taken at the angles given above
     hkls ....... array/list of hkl values for every image
     experiment . Experiment class object needed to get the UB matrix for the
                  hkl peak treatment
     material ... material used as reference crystal
     detaxis .... detector arm rotation axis
                  default: ['z+','y-']
     r_i ........ primary beam direction [xyz][+-]
                  default 'x+'

    keyword_arguments:
        plot .... flag to determine if results and intermediate results should
                  be plotted. default: True
        cut_off . cut off intensity to decide if image is used for the
                  determination or not. default: 0.7 = 70%
        start ... sequence of start values of the fit for parameters,
                  which can not be estimated automatically.  these are:
                  tiltazimuth, tilt, detector_rotation, outerangle_offset,
                  sampletilt, sampletiltazimuth, wavelength.
                  By default (0, 0, 0, 0, 0, 0, 'config') is used.
        fix ..... fix parameters of start (default: (False, False, False,
                  False, False, False, False))
        fig ..... matplotlib figure used for plotting the error.
                  default: None (creates own figure)
        plotlog . flag to specify if the created error plot should be on
                  log-scale
        debug ... flag to tell if you want to see debug output of the script
                  (switch this to true only if you can handle it :))
    """

    if plot:
        try:
            plt.__name__
        except NameError:
            print("XU.analyis.area_detector_calib_hkl: Warning: plot "
                  "functionality not available")
            plot = False

    if start[-1] == 'config':
        start[-1] = config.WAVELENGTH
    elif isinstance(start[-1], str):
        start[-1] = utilities.wavelength(start[-1])

    t0 = time.time()
    Npoints = len(angle1)
    if debug:
        print("number of given images: %d" % Npoints)

    # determine center of mass position from detector images
    # also use only images with an intensity larger than 70% of the average
    # intensity.
    # the image selection is only performed for images in the primary beam
    n1 = numpy.zeros(0, dtype=numpy.double)
    n2 = n1
    ang1 = n1
    ang2 = n1
    sang = n1
    usedhkls = []

    avg = 0
    imgpbcnt = 0
    for i in range(Npoints):
        if (numpy.all(hkls[i] == (0, 0, 0))):
            avg += numpy.sum(ccdimages[i])
            imgpbcnt += 1

    if imgpbcnt > 0:
        avg /= float(imgpbcnt)
    else:
        raise ValueError("XU.analyis.area_detector_calib_hkl: no required "
                         "images in the primary beam given!")
    (N1, N2) = ccdimages[0].shape

    if debug:
        print("average intensity per image in the primary beam: %.1f" % avg)

    for i in range(Npoints):
        img = ccdimages[i]
        if ((numpy.sum(img) > cut_off * avg)
                or (numpy.all(hkls[i] != (0, 0, 0)))):
            [cen1, cen2] = center_of_mass(img)
            if debug:
                plt.figure("_ccd")
                plt.imshow(utilities.maplog(img), origin='low')
                plt.plot(cen2, cen1, "wo", mfc='none')
                plt.axis([cen2 - 25, cen2 + 25, cen1 - 25, cen1 + 25])
                plt.savefig("xu_calib_hkl_ccd_img%d.png" % i)
                plt.close("_ccd")
            n1 = numpy.append(n1, cen1)
            n2 = numpy.append(n2, cen2)
            ang1 = numpy.append(ang1, angle1[i])
            ang2 = numpy.append(ang2, angle2[i])
            sang = numpy.append(sang, sampleang[i])
            usedhkls.append(hkls[i])
            # if debug:
            #     print("%8.3f %8.3f \t%.2f %.2f" % (angle1[i], angle2[i],
            #                                        cen1, cen2))
    Nused = len(ang1)
    usedhkls = numpy.array(usedhkls)

    if debug:
        print("Nused / Npoints: %d / %d" % (Nused, Npoints))

    # guess initial parameters
    n10 = numpy.zeros(0, dtype=numpy.double)
    n20 = n10
    ang10 = n10
    ang20 = n10
    for i in range(Nused):
        if numpy.all(usedhkls[i] == (0, 0, 0)):
            n10 = numpy.append(n10, n1[i])
            n20 = numpy.append(n20, n2[i])
            ang10 = numpy.append(ang10, ang1[i])
            ang20 = numpy.append(ang20, ang2[i])

    detdir1, detdir2 = _determine_detdir(ang10 - start[3], ang20, n10, n20,
                                         detaxis, r_i)

    epslist = []
    paramlist = []
    epsmin = numpy.inf
    fitmin = None

    print("tiltaz   tilt   detrot   offset  sampletilt+azimuth wavelength:  "
          "error (relative) (fittime)")
    print("------------------------------------------------------------")
    # find optimal detector rotation (however keep other parameters free)
    detrot = start[2]
    if not fix[2]:
        for detrotstart in numpy.linspace(start[2] - 1, start[2] + 1, 20):
            start = start[:2] + (detrotstart,) + start[3:]
            eps, param, fit = _area_detector_calib_fit2(
                sang, ang1, ang2, n1, n2, usedhkls, experiment, material,
                detaxis, r_i, detdir1, detdir2, start=start, fix=fix,
                full_output=True)
            epslist.append(eps)
            paramlist.append(param)
            if epslist[-1] < epsmin:
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
        Ntilt = Ntilt * 5 if not fix[1] else Ntilt
        Ntiltaz = Ntiltaz * 5 if not fix[0] else Ntiltaz

    startparam = start[:2] + (detrot,) + start[3:]

    for tiltazimuth in numpy.linspace(startparam[0] if fix[0] else 0, 360,
                                      Ntiltaz, endpoint=False):
        for tilt in numpy.linspace(startparam[1] if fix[1] else 0, 4, Ntilt):
            for offset in numpy.linspace(
                    startparam[3] if fix[3] else - 3 + startparam[3],
                    3 + startparam[3], Noffset):
                t1 = time.time()
                start = (tiltazimuth, tilt, detrot, offset) + start[4:]
                eps, param, fit = _area_detector_calib_fit2(
                    sang, ang1, ang2, n1, n2, usedhkls, experiment, material,
                    detaxis, r_i, detdir1, detdir2, start=start, fix=fix,
                    full_output=True)
                epslist.append(eps)
                paramlist.append(param)
                t2 = time.time()
                print("%6.1f %6.2f %8.3f %8.3f %8.3f %7.2f %8.4f: "
                      "%10.4e (%4.2f) (%5.2fsec)"
                      % (start + (epslist[-1], epslist[-1] / epsmin, t2 - t1)))

                if epslist[-1] < epsmin:
                    print("************************")
                    print("new global minimum found")
                    epsmin = epslist[-1]
                    parammin = param
                    fitmin = fit
                    print("new best parameters: %.2f %.2f %10.4e %10.4e %.1f "
                          "%.2f %.3f %.3f %.3f %.2f %.4f" % parammin)
                    print("************************\n")

    (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
     outerangle_offset, stilt, stazimuth, wavelength) = parammin

    if plot:
        if fig:
            plt.figure(fig.number)
        else:
            plt.figure("CCD Calib fit")
        nparams = numpy.array(paramlist)
        neps = numpy.array(epslist)
        labels = (
            'cch1 (1)',
            'cch2 (1)',
            r'pwidth1 ($\mu$m@1m)',
            r'pwidth2 ($\mu$m@1m)',
            'tiltazimuth (deg)',
            'tilt (deg)',
            'detrot (deg)',
            'outerangle offset (deg)',
            'sample tilt (deg)',
            'st azimuth (deg)',
            'wavelength (AA)')
        xscale = (1., 1., 1.e6, 1.e6, 1., 1., 1., 1., 1., 1., 1.)
        # plt.suptitle("best fit: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f"
        #              %(cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
        #                detrot, outerangle_offset))
        for p in range(11):
            plt.subplot(3, 4, p + 1)
            if plotlog:
                plt.semilogy(nparams[:, p] * xscale[p], neps, 'k.')
            else:
                plt.scatter(nparams[:, p] * xscale[p], neps, c=nparams[:, -1],
                            s=10, marker='o', cmap=plt.cm.gnuplot,
                            edgecolor='none')
            plt.xlabel(labels[p])

        for p in range(11):
            plt.subplot(3, 4, p + 1)
            if plotlog:
                plt.semilogy(parammin[p] * xscale[p], epsmin, 'ko',
                             ms=8, mew=2.5, mec='k', mfc='w')
            else:
                plt.plot(parammin[p] * xscale[p], epsmin, 'ko',
                         ms=8, mew=2.5, mec='k', mfc='w')
                plt.ylim(epsmin * 0.7, epsmin * 2.)
            plt.locator_params(nbins=4, axis='x')
        plt.tight_layout()

    if config.VERBOSITY >= config.INFO_LOW:
        print("total time needed for fit: %.2fsec" % (time.time() - t0))
        print("fitted parameters: epsilon: %10.4e (%d,%s) "
              % (epsmin, fitmin.info, repr(fitmin.stopreason)))
        print("param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,"
              "outerangle_offset,sampletilt,stazimuth,wavelength)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f %.3f %.2f "
              "%.4f" % (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                        detrot, outerangle_offset, stilt, stazimuth,
                        wavelength))

    if config.VERBOSITY > 0:
        print("please check the resulting data (consider setting plot=True)")
        print("detector rotation axis / primary beam direction (given by user)"
              ": %s / %s" % (repr(detaxis), r_i))
        print("detector pixel directions / distance: %s %s / %g"
              % (detdir1, detdir2, 1.))
        print("\tdetector initialization with: init_area('%s', '%s', "
              "cch1=%.2f, cch2=%.2f, Nch1=%d, Nch2=%d, pwidth1=%.4e, "
              "pwidth2=%.4e, distance=1., detrot=%.3f, tiltazimuth=%.1f, "
              "tilt=%.3f)" % (detdir1, detdir2, cch1, cch2, N1, N2, pwidth1,
                              pwidth2, detrot, tiltazimuth, tilt))
        print("AND ALWAYS USE an (additional) OFFSET of %.4fdeg in the "
              "OUTER DETECTOR ANGLE!" % (outerangle_offset))

    return (cch1, cch2, pwidth1, pwidth2, tiltazimuth,
            tilt, detrot, outerangle_offset), eps


def _area_detector_calib_fit2(sang, ang1, ang2, n1, n2, hkls, experiment,
                              material, detaxis, r_i, detdir1, detdir2,
                              start=(0, 0, 0, 0, 0, 0, 1.0),
                              fix=(False, False, False, False, False, False,
                                   False),
                              full_output=False, debug=False):
    """
    INTERNAL FUNCTION
    function to calibrate the detector parameters of an area detector
    it determines the detector tilt possible rotations and offsets in the
    detector arm angles

    parameters
    ----------
     sang ....... sample rocking angle (rotation direction of inner detector
                  rotation)
     angle1 ..... outer detector arm angle
     angle2 ..... inner detector arm angle
     n1,n2 ...... pixel number at which the beam was observed
     hkls ....... Miller indices of the reflection were the images were taken
                  (use (0,0,0)) for primary beam
     experiment . Experiment class object needed to get the UB matrix needed
                  for the hkl peak treatment
     material ... material used as reference crystal
     detaxis .... detector arm rotation axis
                  default: ['z+','y-']
     detdir1,2 .. detector pixel directions of first and second pixel
                  coordinates; e.g. 'y+'
     r_i ........ primary beam direction [xyz][+-]
                  default 'x+'

    keyword_arguments:
        start ... sequence of start values of the fit for parameters,
                  which can not be estimated automatically.
                  these are:
                  tiltazimuth, tilt, detector_rotation, outerangle_offset,
                  sampletilt, sampletiltazimuth, wavelength.
                  By default: (0, 0, 0, 0, 0, 0, 1.0)
        fix ..... fix parameters of start
        full_output .. flag to tell if to return fit object with final
                       parameters and detector directions
        debug ... flag to tell if you want to see debug output of the script
                  (switch this to true only if you can handle it :))

    returns
    -------
        eps   final epsilon of the fit

    if full_output:
        eps,param,fit
    """

    def areapixel2(params, detectorDir1, detectorDir2, r_i, detectorAxis,
                   *args, **kwargs):
        """
        angular to momentum space conversion for pixels of an area detector the
        center pixel is in direction of self.r_i when detector angles are zero

        the detector geometry must be given to the routine

        Parameters
        ----------
        *args:          sample,detector angles and channel numbers
                sAngle  sample rocking angle as numpy array, lists or Scalars
                dAngles as numpy array, lists or Scalars.
                        in total len(detectorAxis) values must be given
                        starting with the outer most circle. all arguments must
                        have the same shape or length.
                channel numbers n1 and n2 where the primary beam hits the
                        detector with the same length as the detector values

        params:         parameters of the detector calibration model
                        (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                         detrot)
                cch1,2: center pixel, in direction of self.r_i at zero
                        detectorAngles
                pwidth1,2: width of one pixel (same unit as distance)
                tiltazimuth: direction of the tilt vector in the detector plane
                             (in degree)
                tilt: tilt of the detector plane around an axis normal to the
                      direction given by the tiltazimuth
                detrot: detector rotation around the primary beam direction as
                        given by r_i

        detectorDir1:   direction of the detector (along the pixel
                        direction 1); e.g. 'z+' means higher pixel numbers at
                        larger z positions
        detectorDir2:   direction of the detector (along the pixel
                        direction 2); e.g. 'x+'

        r_i:            primary beam direction e.g. 'x+'
        detectorAxis:   list or tuple of detector circles.
                        e.g. ['z+','y-'] would mean a detector arm with a two
                        rotations

        **kwargs:       possible keyword arguments
            delta:      giving delta angles to correct the given ones for
                        misalignment. delta must be an numpy array or list of
                        len(*dAngles). used angles are than *args - delta
            UB:         orientation matrix of the sample
            wl:         x-ray wavelength in angstroem
            distance:   distance of center pixel from center of rotation
                        (default: 1. (since it does not matter here))
            deg:        flag to tell if angles are passed as degree (default:
                        True)

        Returns
        -------
        reciprocal space position of detector pixels n1,n2 in a numpy.ndarray
        of shape ( len(args) , 3 )
        """

        # check detector circle argument
        if isinstance(detectorAxis, str):
            dAxis = list([detectorAxis])
        else:
            dAxis = list(detectorAxis)

        _sampleAxis_str = dAxis[-1]
        # add detector rotation around primary beam
        dAxis += [r_i]
        _detectorAxis_str = ''
        for circ in dAxis:
            _detectorAxis_str += circ

        Nd = len(dAxis)
        Nargs = Nd + 2 - 1 + 1

        # check detectorDir
        _area_detdir1 = detectorDir1
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
            raise ValueError("no wavelength given!")

        if 'UB' in kwargs:
            UB = kwargs['UB']
        else:
            UB = numpy.identity(3)

        if 'deg' in kwargs:
            deg = kwargs['deg']
        else:
            deg = True

        if 'delta' in kwargs:
            delta = numpy.array(kwargs['delta'], dtype=numpy.double)
            if delta.size != Nd - 1 + 1:
                raise InputError("QConversionPixel: keyword argument delta "
                                 "does not have an appropriate shape")
        else:
            delta = numpy.zeros(Nd)

        # prepare angular arrays from *args
        # need one sample angle and one detector angle array
        if len(args) != Nargs:
            raise InputError("QConversionPixel: wrong amount (%d) of "
                             "arguments given, number of arguments should be "
                             "%d" % (len(args), Nargs))

        try:
            Npoints = len(args[0])
        except (TypeError, IndexError):
            Npoints = 1

        sAngles = numpy.array((), dtype=numpy.double)
        for i in range(1):
            arg = args[i]
            if isinstance(arg, numbers.Number):
                arg = numpy.array([arg], dtype=numpy.double)
            elif isinstance(arg, list):
                arg = numpy.array(arg, dtype=numpy.double)
            arg = arg - delta[i]
            sAngles = numpy.concatenate((sAngles, arg))

        dAngles = numpy.array((), dtype=numpy.double)
        for i in range(1, Nd):
            arg = args[i]
            if isinstance(arg, numbers.Number):
                arg = numpy.array([arg], dtype=numpy.double)
            elif isinstance(arg, list):
                arg = numpy.array(arg, dtype=numpy.double)
            arg = arg - delta[i]
            dAngles = numpy.concatenate((dAngles, arg))
        # add detector rotation around primary beam
        dAngles = numpy.concatenate((
            dAngles,
            numpy.ones(arg.shape, dtype=numpy.double) * _area_rot))

        # read channel numbers
        n1 = numpy.array((), dtype=numpy.double)
        n2 = numpy.array((), dtype=numpy.double)

        arg = args[Nd]
        if isinstance(arg, numbers.Number):
            arg = numpy.array([arg], dtype=numpy.double)
        elif isinstance(arg, list):
            arg = numpy.array(arg, dtype=numpy.double)
        n1 = arg

        arg = args[Nd + 1]
        if isinstance(arg, numbers.Number):
            arg = numpy.array([arg], dtype=numpy.double)
        elif isinstance(arg, list):
            arg = numpy.array(arg, dtype=numpy.double)
        n2 = arg

        sAngles.shape = (1, Npoints)
        sAngles = sAngles.transpose()
        dAngles.shape = (Nd, Npoints)
        dAngles = dAngles.transpose()

        if deg:
            sAngles = numpy.radians(sAngles)
            dAngles = numpy.radians(dAngles)

        qpos = cxrayutilities.ang2q_conversion_area_pixel2(
            sAngles, dAngles, n1, n2, _area_ri, _sampleAxis_str,
            _detectorAxis_str, _area_cch1, _area_cch2,
            _area_pwidth1 / _area_distance, _area_pwidth2 / _area_distance,
            _area_detdir1, _area_detdir2, _area_tiltazimuth, _area_tilt,
            UB, wl, config.NTHREADS)

        return qpos[:, 0], qpos[:, 1], qpos[:, 2]

    def afunc(param, x, detectorDir1, detectorDir2, r_i, detectorAxis):
        """
        function for fitting the detector parameters
        basically this is a wrapper for the areapixel function

        parameters
        ----------
         param           fit parameters (cch1, cch2, pwidth1, pwidth2,
                         tiltazimuth, tilt, detrot, outerangle_offset,
                         sampletilt, sampletiltazimuth, wavelength)
         x               independent variables; contains (sang, angle1, angle2,
                         n1, n2, hkls) with shape (8, Npoints)
         detectorDir1:   direction of the detector (along the pixel
                         direction 1); e.g. 'z+' means higher pixel numbers at
                         larger z positions
         detectorDir2:   direction of the detector (along the pixel
                         direction 2); e.g. 'x+'
         r_i:            primary beam direction e.g. 'x+'
         detectorAxis:   list or tuple of detector circles; e.g. ['z+','y-']
                         would mean a detector arm with a two rotations

        Returns
        -------
         reciprocal space position of detector pixels n1,n2 in a numpy.ndarray
         of shape ( 3, x.shape[1] )
        """

        sang = x[0, :]
        angle1 = x[1, :]
        angle2 = x[2, :]
        n1 = x[3, :]
        n2 = x[4, :]
        h = x[5, :]
        k = x[6, :]
        l = x[7, :]

        # use only positive tilt and sample tilt
        param[5] = numpy.abs(param[5])
        param[8] = numpy.abs(param[8])
        wl = param[10]
        cp = numpy.cos(numpy.radians(param[9]))
        sp = numpy.sin(numpy.radians(param[9]))
        cc = numpy.cos(numpy.radians(param[8]))
        sc = numpy.sin(numpy.radians(param[8]))

        # UB matrix due to tilt at symmetric peak
        U1 = numpy.array(((cp * cc, -sp, cp * sc),
                          (sp * cc, cp, sp * sc),
                          (-sc, 0, cc)), dtype=numpy.double)
        U2 = experiment._transform.matrix
        U = numpy.dot(U1, U2)
        B = material.B
        ubmat = numpy.dot(U, B)

        (qx, qy, qz) = areapixel2(
            param[:-4], detectorDir1, detectorDir2, r_i, detectorAxis,
            sang, angle1, angle2, n1, n2, delta=[0, param[7], 0.],
            distance=1., UB=ubmat, wl=wl)

#        f= plt.figure("afunc")
#        plt.ion()
#        f.clear()
#        plt.plot(qx-h,'k-',label='x')
#        plt.plot(qy-k,'r-',label='y')
#        plt.plot(qz-l,'g-',label='z')
#        print("param: (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, "
#              "detrot, outerangle_offset)")
#        print(param)
#        plt.legend()
#        plt.draw()
#        plt.waitforbuttonpress()

        return (qx - h) ** 2 + (qy - k) ** 2 + (qz - l) ** 2

    Npoints = len(ang1)

    # guess initial parameters
    n10 = numpy.zeros(0, dtype=numpy.double)
    n20 = n10
    ang10 = n10
    ang20 = n10
    n1s = numpy.zeros(0, dtype=numpy.double)
    n2s = n1s
    ang1s = n1s
    ang2s = n1s
    sangs = n1s

    for i in range(Npoints):
        if numpy.all(hkls[i] == (0, 0, 0)):
            n10 = numpy.append(n10, n1[i])
            n20 = numpy.append(n20, n2[i])
            ang10 = numpy.append(ang10, ang1[i])
            ang20 = numpy.append(ang20, ang2[i])
        else:
            n1s = numpy.append(n1s, n1[i])
            n2s = numpy.append(n2s, n2[i])
            ang1s = numpy.append(ang1s, ang1[i])
            ang2s = numpy.append(ang2s, ang2[i])
            sangs = numpy.append(sangs, sang[i])

    # center channel and detector pixel direction and pixel size
    (s1, i1, r1, dummy, dummy) = scipy.stats.linregress(ang10 - start[3], n10)
    (s2, i2, r2, dummy, dummy) = scipy.stats.linregress(ang10 - start[3], n20)
    (s3, i3, r3, dummy, dummy) = scipy.stats.linregress(ang20, n10)
    (s4, i4, r4, dummy, dummy) = scipy.stats.linregress(ang20, n20)

    if r1 ** 2 > r2 ** 2:
        cch1 = i1
        cch2 = i4
        pwidth1 = 2 / numpy.abs(float(s1)) * numpy.tan(numpy.radians(0.5))
        pwidth2 = 2 / numpy.abs(float(s4)) * numpy.tan(numpy.radians(0.5))
    else:
        cch1 = i3
        cch2 = i2
        pwidth1 = 2 / numpy.abs(float(s3)) * numpy.tan(numpy.radians(0.5))
        pwidth2 = 2 / numpy.abs(float(s2)) * numpy.tan(numpy.radians(0.5))

    tilt = numpy.abs(start[1])
    tiltazimuth = start[0]
    detrot = start[2]
    outerangle_offset = start[3]
    sampletilt = start[4]
    stazimuth = start[5]
    wavelength = start[6]
    # parameters for the fitting
    param = (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
             outerangle_offset, sampletilt, stazimuth, wavelength)

    # determine better start values for sample tilt and azimuth
    (qx, qy, qz) = areapixel2(
        param[:-4], detdir1, detdir2, r_i, detaxis, sangs, ang1s, ang2s,
        n1s, n2s, delta=[0, param[7], 0.], distance=1., wl=wavelength)
    if debug:
        print("average qx: %.3f(%.3f)" % (numpy.average(qx), numpy.std(qx)))
        print("average qy: %.3f(%.3f)" % (numpy.average(qy), numpy.std(qy)))
        print("average qz: %.3f(%.3f)" % (numpy.average(qz), numpy.std(qz)))

    qvecav = (numpy.average(qx), numpy.average(qy), numpy.average(qz))
    sampletilt = math.VecAngle(experiment.Transform(experiment.ndir),
                               qvecav, deg=True)
    stazimuth = -math.VecAngle(
        experiment.Transform(experiment.idir),
        qvecav - math.VecDot(experiment.Transform(experiment.ndir), qvecav) *
        experiment.Transform(experiment.ndir), deg=True)
    param = (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
             outerangle_offset, sampletilt, stazimuth, wavelength)

    if debug:
        print("initial parameters: ")
        print("primary beam / detector pixel directions / distance: %s / "
              "%s %s / %e" % (r_i, detdir1, detdir2, 1.))
        print("param: (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, "
              "detrot, outerangle_offset, sampletilt, stazimuth, wavelength)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f %.3f %.2f "
              "%.4f" % param)

    # set data
    x = numpy.empty((8, Npoints), dtype=numpy.double)
    x[0, :] = sang
    x[1, :] = ang1
    x[2, :] = ang2
    x[3, :] = n1
    x[4, :] = n2
    x[5, :] = hkls[:, 0]
    x[6, :] = hkls[:, 1]
    x[7, :] = hkls[:, 2]

    data = odr.Data(x, y=1)
    # define model for fitting
    model = odr.Model(afunc, extra_args=(detdir1, detdir2, r_i, detaxis),
                      implicit=True)
    # check if parameters need to be fixed
    ifixb = ()
    for i in range(len(fix)):
        ifixb += (int(not fix[i]),)

    my_odr = odr.ODR(data, model, beta0=param, ifixb=(1, 1, 1, 1) + ifixb,
                     ifixx =(0, 0, 0, 0, 0, 0, 0, 0),
                     stpb=(0.4, 0.4, pwidth1 / 50., pwidth2 / 50., 2, 0.125,
                           0.01, 0.01, 0.01, 1., 0.0001),
                     sclb=(1 / numpy.abs(cch1), 1 / numpy.abs(cch2),
                           1 / pwidth1, 1 / pwidth2, 1 / 90., 1 / 0.2, 1 / 0.2,
                           1 / 0.2, 1 / 0.1, 1 / 90., 1.),
                     maxit=1000, ndigit=12, sstol=1e-11, partol=1e-11)
    # if debug:
    #    my_odr.set_iprint(final=1)
    #    my_odr.set_iprint(iter=2)

    fit = my_odr.run()

    (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
     outerangle_offset, sampletilt, stazimuth, wavelength) = fit.beta
    # fix things in parameters
    tiltazimuth = tiltazimuth % 360.
    stazimuth = stazimuth % 360.
    tilt = numpy.abs(tilt)
    sampletilt = numpy.abs(sampletilt)

    final_q = afunc([cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, detrot,
                     outerangle_offset, sampletilt, stazimuth, wavelength],
                    x, detdir1, detdir2, r_i, detaxis)
    final_error = numpy.mean(final_q)

    if False:  # inactive code path
        f = plt.figure("CCD Calib fit")
        f.clear()
        plt.grid(True)
        plt.xlabel("Image number")
        plt.ylabel(r"|$\Delta$Q|")
        errp1, = plt.semilogy(afunc(my_odr.beta0, x, detdir1, detdir2, r_i,
                                    detaxis), 'x-', label='initial param')
        errp2, = plt.semilogy(afunc(fit.beta, x, detdir1, detdir2, r_i,
                                    detaxis),
                              'x-', label='param: %.1f %.1f %5.2g %5.2g %.1f '
                                          '%.2f %.3f %.3f %.3f %.2f %.4f'
                                          % (cch1, cch2, pwidth1, pwidth2,
                                             tiltazimuth, tilt, detrot,
                                             outerangle_offset, sampletilt,
                                             stazimuth, wavelength))
        plt.waitforbuttonpress()

    if debug:
        print("fitted parameters: (%d,%s) " % (fit.info, repr(fit.stopreason)))
        print("primary beam / detector pixel directions / distance: %s / %s "
              "%s / %g" % (r_i, detdir1, detdir2, 1.))
        print("param: (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt, "
              "detrot, outerangle_offset, sampletilt, sampletiltazimuth, "
              "wavelength)")
        print("param: %.2f %.2f %10.4e %10.4e %.1f %.2f %.3f %.3f %.3f %.2f "
              "%.4f" % (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                        detrot, outerangle_offset, sampletilt, stazimuth,
                        wavelength))

    if full_output:
        return final_error, (cch1, cch2, pwidth1, pwidth2, tiltazimuth, tilt,
                             detrot, outerangle_offset, sampletilt, stazimuth,
                             wavelength), fit
    else:
        return final_error


#################################################
def psd_refl_align(primarybeam, angles, channels, plot=True):
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

    (a_s, b_s, r, tt, stderr) = scipy.stats.linregress(channels, angles)

    zeropos = scipy.polyval(numpy.array([a_s, b_s]), primarybeam)

    try:
        plt.__name__
    except NameError:
        print("XU.analyis.psd_refl_align: Warning: plot functionality not "
              "available")
        plot = False

    if plot:
        xmin = min(min(channels), primarybeam)
        xmax = max(max(channels), primarybeam)
        ymin = min(min(angles), zeropos)
        ymax = max(max(angles), zeropos)
        # open new figure for the plot
        plt.figure()
        plt.plot(channels, angles, 'kx', ms=8., mew=2.)
        plt.plot([xmin - (xmax - xmin) * 0.1, xmax + (xmax - xmin) * 0.1],
                 scipy.polyval(numpy.array([a_s, b_s]),
                               [xmin - (xmax - xmin) * 0.1,
                                xmax + (xmax - xmin) * 0.1]),
                 'g-',
                 linewidth=1.5)
        ax = plt.gca()
        plt.grid()
        ax.set_xlim(xmin - (xmax - xmin) * 0.15, xmax + (xmax - xmin) * 0.15)
        ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax + (ymax - ymin) * 0.15)
        plt.vlines(primarybeam,
                   ymin - (ymax - ymin) * 0.1,
                   ymax + (ymax - ymin) * 0.1,
                   linewidth=1.5)
        plt.xlabel("PSD Channel")
        plt.ylabel("sample angle")

    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.psd_refl_align: sample is parallel to beam at "
              "goniometer angle %8.4f (R=%6.4f)" % (zeropos, r))
    return zeropos

#################################################
#  miscut calculation from alignment in 2 and
#  more azimuths
#################################################


def miscut_calc(phi, aomega, zeros=None, omega0=None, plot=True):
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
     omega0:    if specified the nominal value of the reflection is not
                included as fit parameter, but is fixed to the specified
                value. This value is MANDATORY if ONLY TWO AZIMUTHs are
                given.
     plot:      flag to specify if a visualization of the fit is wanted.
                default: True

    Returns
    -------
    [omega0,phi0,miscut]

    list with fitted values for
     omega0:    the omega value of the reflection should be close to
                the nominal one
     phi0:      the azimuth in which the primary beam looks upstairs
     miscut:    amplitude of the sinusoidal variation == miscut angle
    """

    if zeros is not None:
        om = (numpy.array(aomega) - numpy.array(zeros))
    else:
        om = numpy.array(aomega)

    a = numpy.array(phi)

    if omega0 is None:
        # first guess for the parameters
        # omega0,phi0,miscut
        p0 = (om.mean(), a[om.argmax()], om.max() - om.min())
        fitfunc = lambda p, phi: numpy.abs(p[2]) * \
            numpy.cos(numpy.radians(phi - (p[1] % 360.))) + p[0]
    else:
        # first guess for the parameters
        p0 = (a[om.argmax()], om.max() - om.min())  # omega0,phi0,miscut
        fitfunc = lambda p, phi: numpy.abs(p[1]) * \
            numpy.cos(numpy.radians(phi - (p[0] % 360.))) + omega0
    errfunc = lambda p, phi, om: fitfunc(p, phi) - om

    p1, success = optimize.leastsq(errfunc, p0, args=(a, om), maxfev=10000)
    if config.VERBOSITY >= config.INFO_ALL:
        print("xu.analysis.misfit_calc: leastsq optimization return value: "
              "%d" % success)

    if plot:
        try:
            plt.__name__
        except NameError:
            print("XU.analyis.misfit_calc: Warning: plot functionality not "
                  "available")
            plot = False

    if plot:
        plt.figure()
        plt.plot(a, om, 'kx', mew=2, ms=8)
        linx = numpy.linspace(a.min() - 45, a.min() + 360 - 45, num=1000)
        plt.plot(linx, fitfunc(p1, linx), 'g-', linewidth=1.5)
        plt.grid()
        plt.xlabel("azimuth")
        plt.ylabel("aligned sample angle")

    if omega0 is None:
        ret = [p1[0], p1[1] % 360., numpy.abs(p1[2])]
    else:
        ret = [omega0] + [p1[0] % 360., numpy.abs(p1[1])]

    if config.VERBOSITY >= config.INFO_LOW:
        print("xu.analysis.misfit_calc: \n"
              "\t fitted reflection angle: %8.3f \n"
              "\t looking upstairs at phi: %8.2f \n"
              "\t mixcut angle: %8.3f \n" % (ret[0], ret[1], ret[2]))

    return ret

#################################################
#  correct substrate Bragg peak position in
#  reciprocal space maps
#################################################


def fit_bragg_peak(om, tt, psd, omalign, ttalign, exphxrd, frange=(0.03, 0.03),
                   peaktype='Gauss', plot=True):
    """
    helper function to determine the Bragg peak position in a reciprocal
    space map used to obtain the position needed for correction of the data.
    the determination is done by fitting a two dimensional Gaussian
    (xrayutilities.math.Gauss2d) or Lorentzian
    (xrayutilities.math.Lorentz2d)

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
     peaktype: can be 'Gauss' or 'Lorentz' to fit either of the two peak
               shapes
     plot:  if True (default) function will plot the result of the fit in
            comparison with the measurement.

    Returns
    -------
    omfit,ttfit,params,covariance: fitted angular values, and the fit
            parameters (of the Gaussian/Lorentzian) as well as their errors
    """
    if peaktype == 'Gauss':
        func = math.Gauss2d
    elif peaktype == 'Lorentz':
        func = math.Lorentz2d
    else:
        raise InputError("peaktype must be either 'Gauss' or 'Lorentz'")

    if om.size != psd.size:
        [qx, qy, qz] = exphxrd.Ang2Q.linear(om, tt)
    else:
        [qx, qy, qz] = exphxrd.Ang2Q(om, tt)
    [qxsub, qysub, qzsub] = exphxrd.Ang2Q(omalign, ttalign)
    params = [qysub, qzsub, 0.001, 0.001, psd.max(), 0, 0.]
    drange = [qysub - frange[0], qysub + frange[0], qzsub - frange[1],
              qzsub + frange[1]]
    params, covariance = math.fit_peak2d(
        qy.flatten(), qz.flatten(), psd.flatten(), params, drange,
        func, maxfev=10000)
    # correct params
    params[6] = params[6] % (numpy.pi)
    if params[5] < 0:
        params[5] = 0

    [omfit, dummy, dummy, ttfit] = exphxrd.Q2Ang((0, params[0], params[1]),
                                                 trans=False, geometry="real")
    if config.VERBOSITY >= config.INFO_LOW:
        print("XU.analysis.fit_bragg_peak:fitted peak angles: \n\tom =%8.4f\n"
              "\ttt =%8.4f" % (omfit, ttfit))

    if plot:
        plt.figure()
        plt.clf()
        from ..gridder2d import Gridder2D
        from .. import utilities
        gridder = Gridder2D(50, 50)
        mask = (qy.flatten() > drange[0]) * (qy.flatten() < drange[1]) * \
               (qz.flatten() > drange[2]) * (qz.flatten() < drange[3])
        gridder(qy.flatten()[mask], qz.flatten()[mask], psd.flatten()[mask])
        # calculate intensity which should be plotted
        INT = utilities.maplog(gridder.data.transpose(), 4, 0)
        QXm = gridder.xmatrix
        QZm = gridder.ymatrix
        cl = plt.contour(gridder.xaxis, gridder.yaxis,
                         utilities.maplog(func(QXm, QZm, *params), 4, 0).T,
                         8, colors='k', linestyles='solid')
        cf = plt.contourf(gridder.xaxis, gridder.yaxis, INT, 35)
        cf.collections[0].set_label('data')
        cl.collections[0].set_label('fit')
        # plt.legend() # for some reason not working?
        plt.colorbar(extend='min')
        plt.title("plot shows only coarse data! fit used raw data!")

    return omfit, ttfit, params, covariance
