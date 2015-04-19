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
# Copyright (C) 2014 Raphael Grifone <raphael.grifone@esrf.fr>
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
modules to help with the analysis of FastScan data acquired at the ESRF.
FastScan data are X-ray data (various detectors possible) acquired during
scanning the sample in real space with a Piezo Scanner.  The same functions
might be used to analze traditional SPEC mesh scans.

The module provides three core classes:

* FastScan
* FastScanCCD
* FastScanSeries

where the first two are able to parse single mesh/FastScans when one is
interested in data of a single channel detector or are detector and the last
one is able to parse full series of such mesh scans with either type of
detector

see examples/xrayutilities_kmap_ESRF.py for an example script
"""

import os.path
import numpy
import re
import glob

# relative imports
from . import SPECFile
from . import EDFFile
from ..exception import InputError
from ..gridder import delta
from ..gridder2d import Gridder2D
from ..gridder2d import Gridder2DList
from ..gridder3d import Gridder3D
from ..normalize import blockAverage2D
from .. import config
from .. import utilities

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str

SPEC_ImageFile = re.compile(r"^#C imageFile")


class FastScan(object):

    """
    class to help parsing and treating fast scan data.  FastScan is the
    aquisition of X-ray data while scanning the sample with piezo stages in
    real space. It's is available at several beamlines at the ESRF synchrotron
    light-source.
    """

    def __init__(self, filename, scannr,
                 xmotor='adcX', ymotor='adcY', path=""):
        """
        Constructor routine for the FastScan object. It initializes the object
        and parses the spec-scan for the needed data which are saved in
        properties of the FastScan object.

        Parameters
        ----------
         filename:  file name of the fast scan spec file
         scannr:    scannr of the to be parsed fast scan
         optional:
          xmotor:   motor name of the x-motor (default: 'adcX' (ID01))
          ymotor:   motor name of the y-motor (default: 'adcY' (ID01))
          path:     optional path of the FastScan spec file
        """
        self.scannr = scannr
        self.xmotor = xmotor
        self.ymotor = ymotor

        if isinstance(filename, SPECFile):
            self.specfile = filename
            self.filename = self.specfile.filename
            self.full_filename = self.specfile.full_filename
            self.specscan = self.specfile.__getattr__('scan%d' % self.scannr)
        else:
            self.filename = filename
            self.full_filename = os.path.join(path, filename)
            self.filename = os.path.basename(self.full_filename)
            self.specscan = None

        # read the scan
        self.parse()

    def parse(self):
        """
        parse the specfile for the scan number specified in the constructor and
        store the needed informations in the object properties
        """

        # parse the file
        if not self.specscan:
            self.specfile = SPECFile(self.full_filename)
            self.specscan = self.specfile.__getattr__('scan%d' % self.scannr)
        self.specscan.ReadData()

        self.xvalues = self.specscan.data[self.xmotor]
        self.yvalues = self.specscan.data[self.ymotor]

        self.data = self.specscan.data

    def motorposition(self, motorname):
        """
        read the position of motor with name given by motorname from the data
        file. In case the motor is included in the data columns the returned
        object is an array with all the values from the file (although retrace
        clean is respected if already performed). In the case the motor is not
        moved during the scan only one value is returned.

        Parameters
        ----------
         motorname: name of the motor for which the position is wanted

        Returns
        -------
         val:   motor position(s) of motor with name motorname during the scan
        """
        if self.specscan:
            # try reading value from data
            try:
                return self.data[motorname]
            except ValueError:
                try:
                    return self.specscan.init_motor_pos['INIT_MOPO_%s'
                                                        % motorname]
                except ValueError:
                    raise ValueError("given motorname '%s' not found in the "
                                     "Spec-data" % motorname)
        else:
            return None

    def retrace_clean(self):
        """
        function to clean the data of the scan from retrace artifacts created
        by the zig-zag scanning motion of the piezo actuators the function
        cleans the xvalues, yvalues and data attribute of the FastScan object.
        """

        # set window to determin the slope
        window = [-1, 0, 1]
        # calc the slope of x_motor movement using a window for better acuracy
        slope = numpy.convolve(self.xvalues, window, mode='same') / \
            numpy.convolve(numpy.arange(len(self.xvalues)), window, 'same')
        # select where slope is above the slope mean value
        # this can be modified if data points are missing of the retrace does
        # not clean all points
        mask = numpy.where(slope > slope.mean())

        # reduce data size by cutting out retrace
        self.xvalues = self.xvalues[mask]
        self.yvalues = self.yvalues[mask]
        self.data = self.data[mask]

    def grid2D(self, nx, ny, **kwargs):
        """
        function to grid the counter data and return the gridded X,Y and
        Intensity values.

        Parameters
        ----------
         nx,ny:     number of bins in x,y direction

        optional keyword arguments:
         counter:   name of the counter to use for gridding (default: 'mpx4int'
                    (ID01))
         gridrange: range for the gridder: format: ((xmin,xmax),(ymin,ymax))

        Returns
        -------
         Gridder2D object with X,Y,data on regular x,y-grid
        """
        if 'counter' in kwargs:
            self.counter = kwargs['counter']
            kwargs.pop("counter")
        else:
            self.counter = 'mpx4int'

        if 'gridrange' in kwargs:
            gridrange = kwargs['gridrange']
            kwargs.pop("gridrange")
        else:
            gridrange = None

        # define gridder
        g2d = Gridder2D(nx, ny)
        if gridrange:
            g2d.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])

        # check if counter is in data fields
        try:
            inte = self.data[self.counter]
        except ValueError:
            raise ValueError("field named '%s' not found in data parsed from "
                             "scan #%d in file %s"
                             % (self.counter, self.scannr, self.filename))

        # grid data
        g2d(self.xvalues, self.yvalues, self.data[self.counter])

        # return gridded data
        return g2d


class FastScanCCD(FastScan):

    """
    class to help parsing and treating fast scan data including CCD frames.
    FastScan is the aquisition of X-ray data while scanning the sample with
    piezo stages in real space. It's is available at several beamlines at the
    ESRF synchrotron light-source. During such fast scan at every grid point
    CCD frames are recorded and need to be analyzed
    """

    def _gridCCDnumbers(self, nx, ny, ccdnr, gridrange=None):
        """
        internal function to grid the CCD frame number to produce a list of
        ccd-files per bin needed for the further treatment

        Parameters
        ----------
         nx,ny:       number of bins in x,y direction
         ccdnr:       array with ccd file numbers of length
                      length(FastScanCCD.data) OR a string with the data
                      column name for the file ccd-numbers

        optional:
         gridrange:   range for the gridder: format: ((xmin,xmax),(ymin,ymax))

        Returns
        -------
         gridder object:    regular x,y-grid as well as 4-dimensional data
                            object
        """
        g2l = Gridder2DList(nx, ny)
        if gridrange:
            g2l.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])

        if isinstance(ccdnr, basestring):
            # check if counter is in data fields
            try:
                ccdnumbers = self.data[ccdnr]
            except ValueError:
                raise ValueError("field named '%s' not found in data parsed "
                                 "from scan #%d in file %s"
                                 % (ccdnr, self.scannr, self.filename))
        elif isinstance(ccdnr, (list, tuple, numpy.ndarray)):
            ccdnumbers = ccdnr
        else:
            raise ValueError("xu.FastScanCCD: wrong data type for "
                             "argument 'ccdnr'")

        # assign ccd frames to grid
        g2l(self.xvalues, self.yvalues, ccdnumbers)

        return g2l

    def getccdFileTemplate(self, specscan, datadir=None, keepdir=0,
                           numfmt='%04d'):
        """
        function to extract the CCD file template string from the comment
        in the SPEC-file scan-header

        Parameters
        ----------
         specscan:  spec-scan object from which header the CCD directory should
                    be extracted
         datadir:   the CCD filenames are usually parsed from the scan object.
                    With this option the directory used for the data can be
                    overwritten.  Specify the datadir as simple string.
                    Alternatively the innermost directory structure can be
                    automatically taken from the specfile. If this is needed
                    specify the number of directories which should be kept
                    using the keepdir option.
         keepdir:   number of directories which should be taken from the
                    specscan. (default: 0)
         numfmt:    format string for the CCD file number (optional)

        Returns
        -------
         fmtstr:    format string for the CCD file name using one number to
                    build the real file name
        """
        for line in specscan.header:
            if SPEC_ImageFile.match(line):
                for substr in line.split(' '):
                    t = substr.split('[')
                    if len(t) == 2:
                        if t[0] == 'dir':
                            dir = t[1].strip(']')
                        elif t[0] == 'prefix':
                            prefix = t[1].strip(']')
                        elif t[0] == 'suffix':
                            suffix = t[1].strip(']')

        ccdtmp = os.path.join(dir, prefix + numfmt + suffix)
        return utilities.exchange_filepath(ccdtmp, datadir, keepdir)

    def gridCCD(self, nx, ny, ccdnr, roi=None, datadir=None, keepdir=0,
                nav=[1, 1], gridrange=None, filterfunc=None, imgoffset=0):
        """
        function to grid the internal data and ccd files and return the gridded
        X,Y and DATA values. DATA represents a 4D with first two dimensions
        representing X,Y and the remaining two dimensions representing detector
        channels

        Parameters
        ----------
         nx,ny:         number of bins in x,y direction
         ccdnr:         array with ccd file numbers of length
                        length(FastScanCCD.data) OR a string with the data
                        column name for the file ccd-numbers

        optional:
         roi:          region of interest on the 2D detector. should be a list
                       of lower and upper bounds of detector channels for the
                       two pixel directions (default: None)
         datadir:      the CCD filenames are usually parsed from the SPEC file.
                       With this option the directory used for the data can be
                       overwritten.  Specify the datadir as simple string.
                       Alternatively the innermost directory structure can be
                       automatically taken from the specfile. If this is needed
                       specify the number of directories which should be kept
                       using the keepdir option.
         keepdir:      number of directories which should be taken from the
                       SPEC file. (default: 0)
         nav:          number of detector pixel which will be averaged together
                       (reduces the date size)
         gridrange:    range for the gridder: format: ((xmin,xmax),(ymin,ymax))
         filterfunc:   function applied to the CCD-frames before any
                       processing. this function should take a single argument
                       which is the ccddata which need to be returned with the
                       same shape!  e.g. remove hot pixels, flat/darkfield
                       correction

        Returns
        -------
         X,Y,DATA:      regular x,y-grid as well as 4-dimensional data object
        """

        g2l = self._gridCCDnumbers(nx, ny, ccdnr, gridrange=gridrange)
        gdata = g2l.data

        self.ccdtemplate = self.getccdFileTemplate(self.specscan, datadir,
                                                   keepdir)

        # read ccd shape from first image
        filename = sorted(glob.glob(self.ccdtemplate.replace('%04d', '*')))[0]
        if config.VERBOSITY >= config.INFO_ALL:
            print('XU.io.FastScanCCD: open file %s' % filename)
        e = EDFFile(filename, keep_open=True)
        ccdshape = blockAverage2D(e.ReadData(), nav[0], nav[1], roi=roi).shape
        self.ccddata = numpy.zeros((nx, ny, ccdshape[0], ccdshape[1]))
        nimage = e.nimages

        # go through the gridded data and average the ccdframes
        for j in range(gdata.shape[1]):
            for i in range(gdata.shape[0]):
                if len(gdata[i, j]) == 0:
                    continue
                else:
                    framecount = 0
                    # read ccdframes and average them
                    for imgnum in gdata[i, j]:
                        filenumber = (imgnum - imgoffset) // nimage
                        imgindex = int((imgnum - imgoffset) % nimage)
                        newfile = self.ccdtemplate % (filenumber)
                        if e.filename != newfile:
                            if config.VERBOSITY >= config.INFO_ALL:
                                print('XU.io.FastScanCCD: open file %s'
                                      % newfile)
                            e = EDFFile(newfile, keep_open=True)
                        if filterfunc:
                            ccdfilt = filterfunc(e.ReadData(imgindex))
                        else:
                            ccdfilt = e.ReadData(imgindex)
                        ccdframe = blockAverage2D(ccdfilt, nav[0], nav[1],
                                                  roi=roi)
                        self.ccddata[i, j, :, :] += ccdframe
                        framecount += 1
                    self.ccddata[i, j, :, :] = self.ccddata[i, j, :, :] / \
                        float(framecount)

        return g2l.xmatrix, g2l.ymatrix, self.ccddata


class FastScanSeries(object):

    """
    class to help parsing and treating a series of fast scan data including CCD
    frames.  FastScan is the aquisition of X-ray data while scanning the sample
    with piezo stages in real space. It's is available at several beamlines at
    the ESRF synchrotron light-source. During such fast scan at every grid
    point CCD frames are recorded and need to be analyzed.

    For the series of FastScans we assume that they are measured at different
    goniometer angles and therefore transform the data to reciprocal space.
    """

    def __init__(self, filenames, scannrs, nx, ny, *args, **kwargs):
        """
        Constructor routine for the FastScanSeries object. It initializes the
        object and creates a list of FastScanCCD objects. Importantly it also
        expects the motor names of the angles needed for reciprocal space
        conversion.

        Parameters
        ----------
         filenames: file names of the fast scan spec files, in case of more
                    than one filename supply a list of names and also a list of
                    scan numbers for the different files in the 'scannrs'
                    argument
         scannrs:   scannrs of the to be parsed fast scans. in case of one
                    specfile this is a list of numbers (e.g. [1,2,3]). when
                    multiple filenames are given supply a separate list for
                    every file (e.g.  [[1,2,3],[2,4]])
         nx,ny:     grid-points for the real space grid
         *args:     motor names for the Reciprocal space conversion. The order
                    needs be as required by the QConversion.area() function.
         optional keyword arguments:
          xmotor:   motor name of the x-motor (default: 'adcX' (ID01))
          ymotor:   motor name of the y-motor (default: 'adcY' (ID01))
          ccdnr:    name of the ccd-number data column
                    (default: 'imgnr' (ID01))
          counter:  name of a defined counter (roi) in the spec file
                    (default: 'ccdint1' (ID01))
          path:     optional path of the FastScan spec file (default: '')

        """

        if 'ccdnr' in kwargs:
            self.ccdnr = kwargs['ccdnr']
            kwargs.pop("ccdnr")
        else:
            self.ccdnr = 'imgnr'

        if 'counter' in kwargs:
            self.counter = kwargs['counter']
            kwargs.pop("counter")
        else:
            self.counter = 'ccdint1'

        if 'path' in kwargs:
            self.path = kwargs['path']
            kwargs.pop("path")
        else:
            self.path = ''

        self.fastscans = []
        self.nx = nx
        self.ny = ny
        self.motor_pos = None
        self.gridded = False

        self.gonio_motors = []
        # save motor names
        for arg in args:
            if not isinstance(arg, basestring):
                raise ValueError("one of the motor name arguments is not of "
                                 "type 'str' but %s" % str(type(arg)))
            self.gonio_motors.append(arg)

        # create list of FastScans
        if isinstance(filenames, basestring):
            filenames = [filenames]
            scannrs = [scannrs]
        if isinstance(filenames, (tuple, list)):
            for fname in filenames:
                full_filename = os.path.join(self.path, fname)
                specfile = SPECFile(full_filename)
                for snrs in scannrs[filenames.index(fname)]:
                    self.fastscans.append(FastScanCCD(specfile, snrs, **kwargs))
        else:
            raise ValueError("argument 'filenames' is not of "
                             "appropriate type!")

        self.xmin = numpy.min(self.fastscans[0].xvalues)
        self.ymin = numpy.min(self.fastscans[0].yvalues)
        self.xmax = numpy.max(self.fastscans[0].xvalues)
        self.ymax = numpy.max(self.fastscans[0].yvalues)
        for fs in self.fastscans:
            if numpy.max(fs.xvalues) > self.xmax:
                self.xmax = numpy.max(fs.xvalues)
            if numpy.max(fs.yvalues) > self.ymax:
                self.ymax = numpy.max(fs.yvalues)
            if numpy.min(fs.xvalues) < self.xmin:
                self.xmin = numpy.min(fs.xvalues)
            if numpy.min(fs.yvalues) < self.ymin:
                self.ymin = numpy.min(fs.yvalues)

    def retrace_clean(self):
        """
        perform retrace clean for every FastScan in the series
        """
        self.gridded = False
        self.xmin = numpy.min(self.fastscans[0].xvalues)
        self.ymin = numpy.min(self.fastscans[0].yvalues)
        self.xmax = numpy.max(self.fastscans[0].xvalues)
        self.ymax = numpy.max(self.fastscans[0].yvalues)

        for fs in self.fastscans:
            fs.retrace_clean()

            if numpy.max(fs.xvalues) > self.xmax:
                self.xmax = numpy.max(fs.xvalues)
            if numpy.max(fs.yvalues) > self.ymax:
                self.ymax = numpy.max(fs.yvalues)
            if numpy.min(fs.xvalues) < self.xmin:
                self.xmin = numpy.min(fs.xvalues)
            if numpy.min(fs.yvalues) < self.ymin:
                self.ymin = numpy.min(fs.yvalues)

    def align(self, deltax, deltay):
        """
        Since a sample drift or shift due to rotation often occurs between
        different FastScans it should be corrected before combining them. Since
        determining such a shift is not straight-forward in general the user
        needs to supply the routine with the shifts in order correct the
        x,y-values for the different FastScans. Such a routine could for
        example use the integrated CCD intensities and determine the shift
        using a cross-convolution.

        Parameters
        ----------
         deltax:    list of shifts in x-direction for every FastScan in the
                    data structure
         deltay:    same for the y-direction
        """
        self.gridded = False
        self.xmin = numpy.min(self.fastscans[0].xvalues)
        self.ymin = numpy.min(self.fastscans[0].yvalues)
        self.xmax = numpy.max(self.fastscans[0].xvalues)
        self.ymax = numpy.max(self.fastscans[0].yvalues)
        for fs in self.fastscans:
            i = self.fastscans.index(fs)
            fs.xvalues += deltax[i]
            fs.yvalues += deltay[i]

            if numpy.max(fs.xvalues) > self.xmax:
                self.xmax = numpy.max(fs.xvalues)
            if numpy.max(fs.yvalues) > self.ymax:
                self.ymax = numpy.max(fs.yvalues)
            if numpy.min(fs.xvalues) < self.xmin:
                self.xmin = numpy.min(fs.xvalues)
            if numpy.min(fs.yvalues) < self.ymin:
                self.ymin = numpy.min(fs.yvalues)

    def read_motors(self):
        """
        read motor values from the series of fast scans
        """
        self.motor_pos = numpy.zeros((len(self.fastscans),
                                      len(self.gonio_motors)))
        for i in range(len(self.fastscans)):
            fs = self.fastscans[i]
            for j in range(len(self.gonio_motors)):
                mname = self.gonio_motors[j]
                self.motor_pos[i, j] = fs.motorposition(mname)

    def getCCDFrames(self, posx, posy, typ='real'):
        """
        function to determine the list of ccd-frame numbers for a specific real
        space position. The real space position must be within the data limits
        of the FastScanSeries otherwise an ValueError is thrown

        Parameters
        ----------
         posx:  real space x-position or index in x direction
         posy:  real space y-position or index in y direction

        optional:
         typ:   type of coordinates. specifies if the position is specified as
                real space coordinate or as index. valid values are 'real' and
                'index'. (default: 'real')

        Returns
        -------
        [[motorpos1, ccdnrs1], [motorpos2, ccdnrs2], ...]  where motorposN is
          from the N-ths FastScan in the series and ccdnrsN is the list of
          according CCD-frames
        """

        # determine grid point for position x,y
        if typ == 'real':
            # grid point calculation
            def gindex(x, min, delt):
                return numpy.round((x - min) / delt)

            xdelta = delta(self.xmin, self.xmax, self.nx)
            ydelta = delta(self.ymin, self.ymax, self.ny)
            xidx = gindex(posx, self.xmin, xdelta)
            yidx = gindex(posy, self.ymin, ydelta)
        elif typ == 'index':
            xidx = posx
            yidx = posy
        else:
            raise ValueError("given value of 'typ' is invalid.")

        if xidx >= self.nx or xidx < 0:
            raise ValueError("specified x-position is out of the data range")
        if yidx > self.ny or yidx < 0:
            raise ValueError("specified y-position is out of the data range")

        # read motor values and perform gridding for all subscans
        if not self.gridded:
            self.read_motors()
            self.glist = []
            for fs in self.fastscans:
                g2l = fs._gridCCDnumbers(
                    self.nx, self.ny, self.ccdnr,
                    gridrange=((self.xmin, self.xmax), (self.ymin, self.ymax)))
                self.glist.append(g2l)  # contains the ccdnumbers in g2l.data

            self.gridded = True

        # return the ccdnumbers and goniometer angles for this position
        ret = []
        for i in range(len(self.glist)):
            motorpos = self.motor_pos[i]
            ccdnrs = self.glist[i].data[xidx, yidx]
            ret.append([motorpos, ccdnrs])

        return ret

    def rawRSM(self, posx, posy, qconv, roi=None, nav=[1, 1], typ='real',
               datadir=None, keepdir=0, filterfunc=None, **kwargs):
        """
        function to return the reciprocal space map data at a certain
        x,y-position from a series of FastScan measurements. It necessary to
        give the QConversion-object to be used for the reciprocal space
        conversion.  The QConversion-object is expected to have the 'area'
        conversion routines configured properly.

        Parameters
        ----------
         posx:          real space x-position or index in x direction
         posy:          real space y-position or index in y direction
         qconv:         QConversion-object to be used for the conversion of the
                        CCD-data to reciprocal space

        optional:
         roi:           region of interest on the 2D detector. should be a list
                        of lower and upper bounds of detector channels for the
                        two pixel directions (default: None)
         nav:           number of detector pixel which will be averaged
                        together (reduces the date size)
         typ:           type of coordinates. specifies if the position is
                        specified as real space coordinate or as index. valid
                        values are 'real' and 'index'. (default: 'real')
         filterfunc:    function applied to the CCD-frames before any
                        processing. this function should take a single argument
                        which is the ccddata which need to be returned with the
                        same shape!  e.g. remove hot pixels, flat/darkfield
                        correction
         UB:            sample orientation matrix
         datadir:       the CCD filenames are usually parsed from the SPEC
                        file.  With this option the directory used for the data
                        can be overwritten.  Specify the datadir as simple
                        string.  Alternatively the innermost directory
                        structure can be automatically taken from the specfile.
                        If this is needed specify the number of directories
                        which should be kept using the keepdir option.
         keepdir:       number of directories which should be taken from the
                        SPEC file. (default: 0)

        Returns
        -------
        qx,qy,qz,ccddata,valuelist:  raw data of the reciprocal space map and
                                     valuelist containing the ccdframe numbers
                                     and corresponding motor positions
        """
        if 'UB' in kwargs:
            U = kwargs['UB']
            kwargs.pop("UB")
        else:
            U = numpy.identity(3)

        if 'imgoffset' in kwargs:
            imgoffset = kwargs['imgoffset']
            kwargs.pop("imgoffset")
        else:
            imgoffset = 0

        # get CCDframe numbers and motor values
        valuelist = self.getCCDFrames(posx, posy, typ)
        # load ccd frames and convert to reciprocal space
        fsccd = self.fastscans[0]
        self.ccdtemplate = fsccd.getccdFileTemplate(fsccd.specscan, datadir,
                                                    keepdir)
        # read ccd shape from first image
        filename = glob.glob(self.ccdtemplate.replace('%04d', '*'))[0]
        e = EDFFile(filename, keep_open=True)
        ccdshape = blockAverage2D(e.ReadData(), nav[0], nav[1], roi=roi).shape
        ccddata = numpy.zeros((len(self.fastscans), ccdshape[0], ccdshape[1]))
        motors = []
        nimage = e.nimages

        for i in range(len(self.gonio_motors)):
            motors.append(numpy.zeros(0))
        # go through the gridded data and average the ccdframes
        for i in range(len(self.fastscans)):
            imotors, ccdnrs = valuelist[i]
            fsccd = self.fastscans[i]
            # append motor positions
            for j in range(len(self.gonio_motors)):
                motors[j] = numpy.append(motors[j], imotors[j])
            # read CCD
            if len(ccdnrs) == 0:
                continue
            else:
                self.ccdtemplate = fsccd.getccdFileTemplate(fsccd.specscan)
                framecount = 0
                # read ccdframes and average them
                for imgnum in ccdnrs:
                    filenumber = (imgnum - imgoffset) // nimage
                    imgindex = int((imgnum - imgoffset) % nimage)
                    newfile = self.ccdtemplate % (filenumber)
                    if e.filename != newfile:
                        e = EDFFile(newfile, keep_open=True)
                    if filterfunc:
                        ccdfilt = filterfunc(e.ReadData(imgindex))
                    else:
                        ccdfilt = e.ReadData(imgindex)
                    ccdframe = blockAverage2D(ccdfilt, nav[0], nav[1], roi=roi)
                    ccddata[i, :, :] += ccdframe
                    framecount += 1
                ccddata[i, :, :] = ccddata[i, :, :] / float(framecount)

        qx, qy, qz = qconv.area(*motors, roi=roi, Nav=nav, UB=U)
        return qx, qy, qz, ccddata, valuelist

    def gridRSM(self, posx, posy, qnx, qny, qnz, qconv, roi=None, nav=[1, 1],
                typ='real', filterfunc=None, **kwargs):
        """
        function to calculate the reciprocal space map at a certain
        x,y-position from a series of FastScan measurements it is necessary to
        specify the number of grid-oints for the reciprocal space map and the
        QConversion-object to be used for the reciprocal space conversion.  The
        QConversion-object is expected to have the 'area' conversion routines
        configured properly.

        Parameters
        ----------
         posx:      real space x-position or index in x direction
         posy:      real space y-position or index in y direction
         qnx:       number of points in the Qx direction of the gridded
                    reciprocal space map
         qny:       same for y direction
         qnz:       same for z directino
         qconv:     QConversion-object to be used for the conversion of the
                    CCD-data to reciprocal space

        optional:
         roi:       region of interest on the 2D detector. should be a list
                    of lower and upper bounds of detector channels for the
                    two pixel directions (default: None)
         nav:       number of detector pixel which will be averaged together
                    (reduces the date size)
         typ:       type of coordinates. specifies if the position is
                    specified as real space coordinate or as index. valid
                    values are 'real' and 'index'. (default: 'real')
         filterfunc:  function applied to the CCD-frames before any
                      processing. this function should take a single argument
                      which is the ccddata which need to be returned with the
                      same shape!  e.g. remove hot pixels, flat/darkfield
                      correction
         UB:          sample orientation matrix

        Returns
        -------
        Gridder3D object with gridded reciprocal space map
        """
        qx, qy, qz, ccddata, vallist = self.rawRSM(
            posx, posy, qconv, roi=roi, nav=nav,
            typ=typ, filterfunc=filterfunc, **kwargs)
        # perform 3D gridding and return the data or gridder
        g = Gridder3D(qnx, qny, qnz)
        g(qx, qy, qz, ccddata)
        return g

    def grid2Dall(self, nx, ny, **kwargs):
        """
        function to grid the counter data and return the gridded X,Y and
        Intensity values from all the FastScanSeries.

        Parameters
        ----------
         nx,ny:     number of bins in x,y direction

        optional keyword arguments:
         counter:   name of the counter to use for gridding (default: 'mpx4int'
                    (ID01))
         gridrange: range for the gridder: format: ((xmin,xmax),(ymin,ymax))

        Returns
        -------
         Gridder2D object with X,Y,data on regular x,y-grid
        """
        if 'counter' in kwargs:
            counter = kwargs['counter']
            kwargs.pop("counter")
        else:
            counter = 'mpx4int'

        if 'gridrange' in kwargs:
            gridrange = kwargs['gridrange']
            kwargs.pop("gridrange")
        else:
            gridrange = ((self.xmin, self.xmax), (self.ymin, self.ymax))

        # define gridder
        g2d = Gridder2D(nx, ny)
        if gridrange:
            g2d.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])
        g2d.KeepData(True)

        for fs in self.fastscans:
            # check if counter is in data fields
            try:
                inte = fs.data[counter]
            except ValueError:
                raise ValueError("field named '%s' not found in data parsed "
                                 "from scan #%d in file %s"
                                 % (counter, fs.scannr, fs.filename))

            # grid data
            g2d(fs.xvalues, fs.yvalues, fs.data[counter])

        # return gridded data
        return g2d
