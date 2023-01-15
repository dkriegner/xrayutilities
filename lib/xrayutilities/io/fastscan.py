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
# Copyright (c) 2014-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

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
import re

import h5py
import numpy

from .. import config, utilities
from ..gridder import delta
from ..gridder2d import Gridder2D, Gridder2DList
from ..gridder3d import Gridder3D
from ..normalize import blockAverage2D
from .edf import EDFFile
from .spec import SPECFile


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
        filename :      str
            file name of the fast scan spec file
        scannr :        int
            scannr of the to be parsed fast scan
        xmotor :        str, optional
            motor name of the x-motor (default: 'adcX' (ID01))
        ymotor :        str, optional
            motor name of the y-motor (default: 'adcY' (ID01))
        path :          str, optional
            optional path of the FastScan spec file
        """
        self.scannr = scannr
        self.xmotor = xmotor
        self.ymotor = ymotor

        if isinstance(filename, SPECFile):
            self.specfile = filename
            self.filename = self.specfile.filename
            self.full_filename = self.specfile.full_filename
            self.specscan = getattr(self.specfile, 'scan%d' % self.scannr)
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
            self.specscan = getattr(self.specfile, 'scan%d' % self.scannr)
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
        motorname :     str
            name of the motor for which the position is wanted

        Returns
        -------
        ndarray
            motor position(s) of motor with name motorname during the scan
        """
        if self.specscan:
            # try reading value from data
            try:
                return self.data[motorname]
            except ValueError:
                try:
                    return self.specscan.init_motor_pos['INIT_MOPO_%s'
                                                        % motorname]
                except KeyError:
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
        function to grid the counter data and return the gridded X, Y and
        Intensity values.

        Parameters
        ----------
        nx, ny :    int
            number of bins in x, and y direction
        counter :   str, optional
            name of the counter to use for gridding (default: 'mpx4int' (ID01))
        gridrange : tuple, optional
            range for the gridder: format: ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Gridder2D
            Gridder2D object with X, Y, data on regular x, y-grid
        """
        self.counter = kwargs.get('counter', 'mpx4int')
        gridrange = kwargs.get('gridrange', None)

        # define gridder
        g2d = Gridder2D(nx, ny)
        if gridrange:
            g2d.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])

        # check if counter is in data fields
        if self.counter not in self.data.dtype.fields:
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

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        imagefiletype : str, optional
            image file extension, either 'edf' / 'edf.gz' (default) or 'h5'

        other parameters are passed on to FastScanCCD
        """
        self.imagefiletype = kwargs.pop('imagefiletype', 'edf')
        self.imgfile = None
        self.nimages = None
        super().__init__(*args, **kwargs)

    def _getCCDnumbers(self, ccdnr):
        """
        internal function to return the ccd frame numbers from the data object
        or take them from the argument.
        """
        if isinstance(ccdnr, str):
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
        return ccdnumbers

    def _gridCCDnumbers(self, nx, ny, ccdnr, gridrange=None):
        """
        internal function to grid the CCD frame number to produce a list of
        ccd-files per bin needed for the further treatment

        Parameters
        ----------
        nx, ny :        int
            number of bins in x, and y direction
        ccdnr :         str or array-like
            array with ccd file numbers of length length(FastScanCCD.data) OR a
            string with the data column name for the file ccd-numbers

        gridrange :     tuple, optional
            range for the gridder: format: ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        gridder-object
            regular x, y-grid as well as 4-dimensional data object
        """
        g2l = Gridder2DList(nx, ny)
        if gridrange:
            g2l.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])

        ccdnumbers = self._getCCDnumbers(ccdnr)
        # assign ccd frames to grid
        g2l(self.xvalues, self.yvalues, ccdnumbers)

        return g2l

    def _read_image(self, filename, imgindex, nav, roi, filterfunc):
        """
        helper function to obtain one frame from an EDF/HDF5 file

        Parameters
        ----------
        filename :      str
            EDF file name
        imgindex :      int
            index of frame inside the given EDF file
        nav :           tuple or list
            number of detector pixel which will be averaged together (reduces
            the date size)
        roi :           tuple
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        filterfunc :    callable
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction

        Returns
        -------
        ndarray
            numpy 2D array with the detector frame
        """
        if roi is None:
            kwdict = {}
        else:
            kwdict = {'roi': roi}
        if 'edf' in self.imagefiletype:
            if not self.imgfile:
                self.imgfile = EDFFile(filename, keep_open=True)
            else:
                if self.imgfile.filename != filename:
                    self.imgfile = EDFFile(filename, keep_open=True)
            ccdfilt = self.imgfile.ReadData(imgindex)
        else:
            fileroot = os.path.splitext(os.path.splitext(filename)[0])[0]
            if not self.imgfile:
                self.imgfile = h5py.File(fileroot + '.h5', 'r')
            ccdfilt = self.imgfile.get(
                os.path.split(fileroot)[-1] + '_%04d' % imgindex).value
        if filterfunc:
            ccdfilt = filterfunc(ccdfilt)
        if roi is None and nav[0] == 1 and nav[1] == 1:
            return ccdfilt
        else:
            return blockAverage2D(ccdfilt, nav[0], nav[1], **kwdict)

    def _get_image_number(self, imgnum, imgoffset, fileoffset, ccdfiletmp):
        """
        function to obtain the image and file number. The logic for obtain this
        is likely to change between beamtimes.

        Parameters
        ----------
        imgnum :        int
            running image number from the data file
        imgoffset :     int
            offset in the image number
        fileoffset :    int
            offset in the file number
        ccdfiletmp :    str
            ccd file template string
        """
        if 'edf' in self.imagefiletype:
            if not self.imgfile:
                self.imgfile = EDFFile(ccdfiletmp % fileoffset, keep_open=True)
            if self.nimages is None:
                self.nimages = self.imgfile.nimages
        else:
            fileroot = os.path.splitext(os.path.splitext(ccdfiletmp
                                                         % fileoffset)[0])[0]
            if not self.imgfile:
                self.imgfile = h5py.File(fileroot + '.h5', 'r')
            if self.nimages is None:
                self.nimages = len(self.imgfile.items())
        filenumber = int((imgnum - imgoffset) // self.nimages + fileoffset)
        imgindex = int((imgnum - imgoffset) % self.nimages)
        return imgindex, filenumber

    def getccdFileTemplate(self, specscan, datadir=None, keepdir=0,
                           replacedir=None):
        """
        function to extract the CCD file template string from the comment
        in the SPEC-file scan-header.

        Parameters
        ----------
        specscan :      SpecScan
            spec-scan object from which header the CCD directory should be
            extracted
        datadir :       str, optional
            the CCD filenames are usually parsed from the scan object.  With
            this option the directory used for the data can be overwritten.
            Specify the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept using the keepdir option.
        keepdir :       int, optional
            number of directories which should be taken from the specscan.
            (default: 0)
        replacedir :    int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.

        Returns
        -------
        fmtstr :    str
            format string for the CCD file name using one number to build the
            real file name
        filenr :    int
            starting file number
        """
        hline = specscan.getheader_element('C imageFile')
        re_ccdfiles = re.compile(r'dir\[([a-zA-Z0-9_.%/]*)\] '
                                 r'prefix\[([a-zA-Z0-9_.%/]*)\] '
                                 r'idxFmt\[([a-zA-Z0-9_.%/]*)\] '
                                 r'nextNr\[([0-9]*)\] '
                                 r'suffix\[([a-zA-Z0-9_.%/]*)\]')
        m = re_ccdfiles.match(hline)
        if m:
            path, prefix, idxFmt, num, suffix = m.groups()
        else:
            ValueError('spec-scan does not contain images or the '
                       'corresponding header line is not detected correctly')
        ccdtmp = os.path.join(path, prefix + idxFmt + suffix)
        r = utilities.exchange_filepath(ccdtmp, datadir, keepdir, replacedir)
        return r, int(num)

    def getCCD(self, ccdnr, roi=None, datadir=None, keepdir=0,
               replacedir=None, nav=[1, 1], filterfunc=None):
        """
        function to read the ccd files and return the raw X, Y and DATA values.
        DATA represents a 3D object with first dimension representing the data
        point index and the remaining two dimensions representing detector
        channels

        Parameters
        ----------
        ccdnr :     array-like or str
            array with ccd file numbers of length length(FastScanCCD.data) OR a
            string with the data column name for the file ccd-numbers

        roi :       tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        datadir :   str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept using the keepdir option.
        keepdir :   int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir : int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.
        nav :       tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        filterfunc : callable
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction

        Returns
        -------
        X, Y :  ndarray
            x, y-array (1D)
        DATA :  ndarray
            3-dimensional data object
        """
        ccdnumbers = self._getCCDnumbers(ccdnr)

        ccdtemplate, nextNr = self.getccdFileTemplate(
            self.specscan, datadir, keepdir=keepdir, replacedir=replacedir)

        # read ccd shape from first image
        filename = ccdtemplate % nextNr
        ccdshape = self._read_image(filename, 0, nav, roi, filterfunc).shape
        ccddata = numpy.empty((self.xvalues.size, ccdshape[0], ccdshape[1]))
        if config.VERBOSITY >= config.INFO_ALL:
            print('XU.io.FastScanCCD: allocated ccddata array with %d bytes'
                  % ccddata.nbytes)

        # go through the ccd-frames
        for i, imgnum in enumerate(ccdnumbers):
            # read ccd-frames
            imgindex, filenumber = self._get_image_number(imgnum, nextNr,
                                                          nextNr, ccdtemplate)
            filename = ccdtemplate % filenumber
            ccd = self._read_image(filename, imgindex, nav, roi, filterfunc)
            ccddata[i, :, :] = ccd
        return self.xvalues, self.yvalues, ccddata

    def processCCD(self, ccdnr, roi, datadir=None, keepdir=0,
                   replacedir=None, filterfunc=None):
        """
        function to read a region of interest (ROI) from the ccd files and
        return the raw X, Y and intensity from ROI.

        Parameters
        ----------
        ccdnr :     array-like or str
            array with ccd file numbers of length length(FastScanCCD.data) OR a
            string with the data column name for the file ccd-numbers
        roi :       tuple or list
            region of interest on the 2D detector. Either a list of lower and
            upper bounds of detector channels for the two pixel directions as
            tuple or a list of mask arrays
        datadir :   str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept using the keepdir option.
        keepdir :   int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir : int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.
        filterfunc : callable, optional
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction

        Returns
        -------
        X, Y, DATA :     ndarray
            x, y-array (1D) as well as 1-dimensional data object
        """
        ccdnumbers = self._getCCDnumbers(ccdnr)

        ccdtemplate, nextNr = self.getccdFileTemplate(
            self.specscan, datadir, keepdir=keepdir, replacedir=replacedir)

        if isinstance(roi, list):
            lmask = roi
            lroi = None
        else:
            lmask = [numpy.ones((roi[1]-roi[0], roi[3]-roi[2])), ]
            lroi = roi
        ccdroi = numpy.empty((len(lmask), self.xvalues.size))

        # go through the ccd-frames
        for i, imgnum in enumerate(ccdnumbers):
            # read ccd-frames
            imgindex, filenumber = self._get_image_number(imgnum, nextNr,
                                                          nextNr, ccdtemplate)
            filename = ccdtemplate % filenumber
            ccd = self._read_image(filename, imgindex, [1, 1], lroi,
                                   filterfunc)
            for j, m in enumerate(lmask):
                ccdroi[j, i] = numpy.sum(ccd[m])
        if len(lmask) == 1:
            return self.xvalues, self.yvalues, ccdroi[0]
        else:
            return self.xvalues, self.yvalues, ccdroi

    def gridCCD(self, nx, ny, ccdnr, roi=None, datadir=None, keepdir=0,
                replacedir=None, nav=[1, 1], gridrange=None, filterfunc=None):
        """
        function to grid the internal data and ccd files and return the gridded
        X, Y and DATA values. DATA represents a 4D object with first two
        dimensions representing X, Y and the remaining two dimensions
        representing detector channels

        Parameters
        ----------
        nx, ny :    int
            number of bins in x, and y direction
        ccdnr :     array-like or str
            array with ccd file numbers of length length(FastScanCCD.data) OR a
            string with the data column name for the file ccd-numbers

        roi :       tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        datadir :   str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept using the keepdir option.
        keepdir :   int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir : int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.
        nav :       tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        gridrange : tuple
            range for the gridder: format: ((xmin, xmax), (ymin, ymax))
        filterfunc : callable
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction

        Returns
        -------
        X, Y:       ndarray
            regular x, y-grid
        DATA :      ndarray
            4-dimensional data object
        """

        g2l = self._gridCCDnumbers(nx, ny, ccdnr, gridrange=gridrange)
        gdata = g2l.data

        ccdtemplate, nextNr = self.getccdFileTemplate(
            self.specscan, datadir, keepdir=keepdir, replacedir=replacedir)

        # read ccd shape from first image
        filename = ccdtemplate % nextNr
        ccdshape = self._read_image(filename, 0, nav, roi, filterfunc).shape
        ccddata = numpy.empty((self.xvalues.size, ccdshape[0], ccdshape[1]))
        if config.VERBOSITY >= config.INFO_ALL:
            print('XU.io.FastScanCCD: allocated ccddata array with %d bytes'
                  % ccddata.nbytes)

        # go through the gridded data and average the ccd-frames
        for i in range(gdata.shape[0]):
            for j in range(gdata.shape[1]):
                if not gdata[i, j]:
                    continue
                else:
                    framecount = 0
                    # read ccd-frames and average them
                    for imgnum in gdata[i, j]:
                        imgindex, filenumber = self._get_image_number(
                            imgnum, nextNr, nextNr, ccdtemplate)
                        filename = ccdtemplate % filenumber
                        ccd = self._read_image(filename, imgindex, nav,
                                               roi, filterfunc)
                        ccddata[i, j, ...] += ccd
                        framecount += 1
                    ccddata[i, j, ...] /= float(framecount)

        return g2l.xmatrix, g2l.ymatrix, ccddata


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
        filenames : list or str
            file names of the fast scan spec files, in case of more than one
            filename supply a list of names and also a list of scan numbers for
            the different files in the 'scannrs' argument
        scannrs :   list
            scannrs of the to be parsed fast scans. in case of one specfile
            this is a list of numbers (e.g. [1, 2, 3]). when multiple filenames
            are given supply a separate list for every file (e.g.  [[1, 2,
            3],[2, 4]])
        nx, ny :    int
            grid-points for the real space grid
        args :      str
            motor names for the Reciprocal space conversion. The order needs be
            as required by the ``QConversion.area()`` function.
        xmotor :    str, optional
            motor name of the x-motor (default: 'adcX' (ID01))
        ymotor :    str, optional
            motor name of the y-motor (default: 'adcY' (ID01))
        ccdnr :     str, optional
            name of the ccd-number data column (default: 'imgnr' (ID01))
        counter :   str, optional
            name of a defined counter (roi) in the spec file (default:
            'mpx4int' (ID01))
        path :      str, optional
            path of the FastScan spec file (default: '')
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
            self.counter = 'mpx4int'

        if 'path' in kwargs:
            self.path = kwargs['path']
            kwargs.pop("path")
        else:
            self.path = ''

        self.fastscans = []
        self.nx = nx
        self.ny = ny
        self.motor_pos = None

        self.gonio_motors = []
        # save motor names
        for arg in args:
            if not isinstance(arg, str):
                raise ValueError("one of the motor name arguments is not of "
                                 "type 'str' but %s" % str(type(arg)))
            self.gonio_motors.append(arg)

        # create list of FastScans
        if isinstance(filenames, str):
            filenames = [filenames]
            scannrs = [scannrs]
        if isinstance(filenames, (tuple, list)):
            for fname in filenames:
                full_filename = os.path.join(self.path, fname)
                specfile = SPECFile(full_filename)
                for snrs in scannrs[filenames.index(fname)]:
                    self.fastscans.append(FastScanCCD(specfile,
                                                      snrs, **kwargs))
        else:
            raise ValueError("argument 'filenames' is not of "
                             "appropriate type!")

        self._init_minmax()
        for fs in self.fastscans:
            self._update_minmax(fs)

    def _init_minmax(self):
        self.gridded = False
        self.xmin = numpy.min(self.fastscans[0].xvalues)
        self.ymin = numpy.min(self.fastscans[0].yvalues)
        self.xmax = numpy.max(self.fastscans[0].xvalues)
        self.ymax = numpy.max(self.fastscans[0].yvalues)

    def _update_minmax(self, fs):
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
        self._init_minmax()

        for fs in self.fastscans:
            fs.retrace_clean()
            self._update_minmax(fs)

    def align(self, deltax, deltay):
        """
        Since a sample drift or shift due to rotation often occurs between
        different FastScans it should be corrected before combining them. Since
        determining such a shift is not straight-forward in general the user
        needs to supply the routine with the shifts in order correct the
        x, y-values for the different FastScans. Such a routine could for
        example use the integrated CCD intensities and determine the shift
        using a cross-convolution.

        Parameters
        ----------
        deltax, deltay :    list
            list of shifts in x/y-direction for every FastScan in the data
            structure
        """
        self._init_minmax()
        for fs in self.fastscans:
            i = self.fastscans.index(fs)
            fs.xvalues += deltax[i]
            fs.yvalues += deltay[i]
            self._update_minmax(fs)

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

    def get_average_RSM(self, qnx, qny, qnz, qconv, datadir=None, keepdir=0,
                        replacedir=None, roi=None, nav=(1, 1),
                        filterfunc=None):
        """
        function to return the reciprocal space map data averaged over all x, y
        positions from a series of FastScan measurements. It necessary to give
        the QConversion-object to be used for the reciprocal space conversion.
        The QConversion-object is expected to have the 'area' conversion
        routines configured properly. This function needs to read all detector
        images, so be prepared to lean back for a moment!

        Parameters
        ----------
        qnx, qny, qnz : int
            number of points used for the 3D Gridder
        qconv :         QConversion
            QConversion-object to be used for the conversion of the CCD-data to
            reciprocal space

        roi :           tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        nav :           tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        filterfunc :    callable, optional
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction
        datadir :       str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept/replaced using the keepdir/replacedir option.
        keepdir :       int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir :    int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.

        Returns
        -------
        Gridder3D
            gridded reciprocal space map
        """
        if self.motor_pos is None:
            self.read_motors()

        # determine q-coordinates
        kwargs = {'Nav': nav}
        if roi:
            kwargs['roi'] = roi
        qx, qy, qz = qconv.area(*self.motor_pos.T, **kwargs)

        # define gridder with fixed optimized q-range
        g3d = Gridder3D(qnx, qny, qnz)
        g3d.keep_data = True
        g3d.dataRange(qx.min(), qx.max(), qy.min(),
                      qy.max(), qz.min(), qz.max(), fixed=True)

        # start parsing the images and grid the data frame by frame
        for fsidx, fsccd in enumerate(self.fastscans):
            ccdtemplate, nextNr = fsccd.getccdFileTemplate(
                fsccd.specscan, datadir, keepdir=keepdir,
                replacedir=replacedir)

            ccdnumbers = fsccd._getCCDnumbers(self.ccdnr)
            ccdav = numpy.zeros_like(qx[fsidx, ...])

            # go through the ccdframes
            for i, imgnum in enumerate(ccdnumbers):
                # read ccdframes
                imgindex, filenumber = fsccd._get_image_number(
                    imgnum, nextNr, nextNr, ccdtemplate)
                filename = ccdtemplate % filenumber
                ccd = fsccd._read_image(filename, imgindex, nav, roi,
                                        filterfunc)
                ccdav += ccd
            g3d(qx[fsidx, ...], qy[fsidx, ...], qz[fsidx, ...], ccdav)

        return g3d

    def get_sxrd_for_qrange(self, qrange, qconv, datadir=None, keepdir=0,
                            replacedir=None, roi=None, nav=(1, 1),
                            filterfunc=None):
        """
        function to return the real space data averaged over a certain q-range
        from a series of FastScan measurements. It necessary to give the
        QConversion-object to be used for the reciprocal space conversion.  The
        QConversion-object is expected to have the 'area' conversion routines
        configured properly.

        Note:
            This function assumes that all FastScans were performed in the same
            real space positions, no gridding or aligning is performed!

        Parameters
        ----------
        qrange :    list or tuple
            q-limits defining a box in reciprocal space.  six values are
            needed: [minx, maxx, miny, ..., maxz]
        qconv :     QConversion
            QConversion object to be used for the conversion of the CCD-data to
            reciprocal space

        roi :       tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        nav :       tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        filterfunc : callable, optional
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction
        datadir :   str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept/replaced using the keepdir/replacedir option.
        keepdir :   int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir : int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.

        Returns
        -------
        xvalues, yvalues, data :    ndarray
            x, y, and data values
        """
        if self.motor_pos is None:
            self.read_motors()

        # determine q-coordinates
        kwargs = {'Nav': nav}
        if roi:
            kwargs['roi'] = roi
        qx, qy, qz = qconv.area(*self.motor_pos.T, **kwargs)
        output = numpy.zeros_like(self.fastscans[0].xvalues)

        # parse the images only if some q coordinates fall into the ROI
        for fsidx, fsccd in enumerate(self.fastscans):
            mask = numpy.logical_and.reduce((
                qx[fsidx] > qrange[0], qx[fsidx] < qrange[1],
                qy[fsidx] > qrange[2], qy[fsidx] < qrange[3],
                qz[fsidx] > qrange[4], qz[fsidx] < qrange[5]))

            if numpy.any(mask):
                ccdtemplate, nextNr = fsccd.getccdFileTemplate(
                    fsccd.specscan, datadir, keepdir=keepdir,
                    replacedir=replacedir)

                ccdnumbers = fsccd._getCCDnumbers(self.ccdnr)

                # go through the ccdframes
                for i, imgnum in enumerate(ccdnumbers):
                    # read ccdframes
                    imgindex, filenumber = fsccd._get_image_number(
                        imgnum, nextNr, nextNr, ccdtemplate)
                    filename = ccdtemplate % filenumber
                    ccd = fsccd._read_image(filename, imgindex, nav, roi,
                                            filterfunc)
                    output[i] += numpy.sum(ccd[mask])

        return fsccd.xvalues, fsccd.yvalues, output

    def getCCDFrames(self, posx, posy, typ='real'):
        """
        function to determine the list of ccd-frame numbers for a specific real
        space position. The real space position must be within the data limits
        of the FastScanSeries otherwise an ValueError is thrown

        Parameters
        ----------
        posx :      float
            real space x-position or index in x direction
        posy :      float
            real space y-position or index in y direction

        typ :       {'real', 'index'}, optional
            type of coordinates. specifies if the position is specified as real
            space coordinate or as index. (default: 'real')

        Returns
        -------
        list
            ``[[motorpos1, ccdnrs1], [motorpos2, ccdnrs2], ...]``  where
            motorposN is from the N-ths FastScan in the series and ccdnrsN is
            the list of according CCD-frames
        """

        # determine grid point for position x, y
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
               datadir=None, keepdir=0, replacedir=None, filterfunc=None,
               **kwargs):
        """
        function to return the reciprocal space map data at a certain
        x, y-position from a series of FastScan measurements. It necessary to
        give the QConversion-object to be used for the reciprocal space
        conversion.  The QConversion-object is expected to have the 'area'
        conversion routines configured properly.

        Parameters
        ----------
        posx :      float
            real space x-position or index in x direction
        posy :      float
            real space y-position or index in y direction
        qconv :     QConversion
            QConversion-object to be used for the conversion of the CCD-data to
            reciprocal space

        roi :       tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        nav :       tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        typ :       {'real', 'index'}, optional
            type of coordinates. specifies if the position is specified as real
            space coordinate or as index. (default: 'real')
        filterfunc : callable, optional
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction
        UB :        array-like, optional
            sample orientation matrix
        datadir :   str, optional
            the CCD filenames are usually parsed from the SPEC file.  With this
            option the directory used for the data can be overwritten.  Specify
            the datadir as simple string.  Alternatively the innermost
            directory structure can be automatically taken from the specfile.
            If this is needed specify the number of directories which should be
            kept using the keepdir option.
        keepdir :   int, optional
            number of directories which should be taken from the SPEC file.
            (default: 0)
        replacedir : int, optional
            number of outer most directory names which should be replaced in
            the output (default = None). One can either give keepdir, or
            replacedir, with replace taking preference if both are given.

        Returns
        -------
        qx, qy, qz :    ndarray
            reciprocal space positions of the reciprocal space map
        ccddata :       ndarray
            raw data of the reciprocal space map
        valuelist :     ndarray
            valuelist containing the ccdframe numbers and corresponding motor
            positions
        """
        U = kwargs.get('UB', numpy.identity(3))

        # get CCDframe numbers and motor values
        valuelist = self.getCCDFrames(posx, posy, typ)
        # load ccd frames and convert to reciprocal space
        fsccd = self.fastscans[0]
        ccdtemplate, nextNr = fsccd.getccdFileTemplate(
            fsccd.specscan, datadir, keepdir=keepdir, replacedir=replacedir)

        # read ccd shape from first image
        imgindex, filenumber = fsccd._get_image_number(
            valuelist[0][1], nextNr, nextNr, ccdtemplate)
        filename = ccdtemplate % filenumber
        ccdshape = fsccd._read_image(filename, imgindex, nav, roi,
                                     filterfunc).shape
        ccddata = numpy.zeros((len(self.fastscans), ccdshape[0], ccdshape[1]))
        motors = []

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
            if not ccdnrs:
                continue
            else:
                ccdtemplate, nextNr = fsccd.getccdFileTemplate(fsccd.specscan)
                framecount = 0
                # read ccd-frames and average them
                for imgnum in ccdnrs:
                    imgindex, filenumber = fsccd._get_image_number(
                        imgnum, nextNr, nextNr, ccdtemplate)
                    filename = ccdtemplate % filenumber
                    ccd = fsccd._read_image(filename, imgindex, nav, roi,
                                            filterfunc)
                    ccddata[i, ...] += ccd
                    framecount += 1
                ccddata[i, ...] /= float(framecount)

        qx, qy, qz = qconv.area(*motors, roi=roi, Nav=nav, UB=U)
        return qx, qy, qz, ccddata, valuelist

    def gridRSM(self, posx, posy, qnx, qny, qnz, qconv, roi=None, nav=[1, 1],
                typ='real', filterfunc=None, **kwargs):
        """
        function to calculate the reciprocal space map at a certain
        x, y-position from a series of FastScan measurements it is necessary to
        specify the number of grid-oints for the reciprocal space map and the
        QConversion-object to be used for the reciprocal space conversion.  The
        QConversion-object is expected to have the 'area' conversion routines
        configured properly.

        Parameters
        ----------
        posx :      float
            real space x-position or index in x direction
        posy :      float
            real space y-position or index in y direction
        qnx, qny, qnz : int
            number of points in the Qx, Qy, Qz direction of the gridded
            reciprocal space map
        qconv :     QConversion
            QConversion-object to be used for the conversion of the CCD-data to
            reciprocal space
        roi :       tuple, optional
            region of interest on the 2D detector. should be a list of lower
            and upper bounds of detector channels for the two pixel directions
            (default: None)
        nav :       tuple or list, optional
            number of detector pixel which will be averaged together (reduces
            the date size)
        typ :       {'real', 'index'}, optional
            type of coordinates. specifies if the position is specified as real
            space coordinate or as index. (default: 'real')
        filterfunc : callable, optional
            function applied to the CCD-frames before any processing. this
            function should take a single argument which is the ccddata which
            need to be returned with the same shape!  e.g. remove hot pixels,
            flat/darkfield correction
        UB :         ndarray
            sample orientation matrix

        Returns
        -------
        Gridder3D
            object with gridded reciprocal space map
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
        function to grid the counter data and return the gridded X, Y and
        Intensity values from all the FastScanSeries.

        Parameters
        ----------
        nx, ny :    int
            number of bins in x, and y direction

        counter :   str, optional
            name of the counter to use for gridding (default: 'mpx4int' (ID01))
        gridrange : tuple, optional
            range for the gridder: format: ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Gridder2D
            object with X, Y, data on regular x, y-grid
        """
        counter = kwargs.get('counter', 'mpx4int')
        gridrange = kwargs.get('gridrange', ((self.xmin, self.xmax),
                                             (self.ymin, self.ymax)))

        # define gridder
        g2d = Gridder2D(nx, ny)
        if gridrange:
            g2d.dataRange(gridrange[0][0], gridrange[0][1],
                          gridrange[1][0], gridrange[1][1])
        g2d.KeepData(True)

        for fs in self.fastscans:
            # check if counter is in data fields
            if counter not in fs.data.dtype.fields:
                raise ValueError("field named '%s' not found in data parsed "
                                 "from scan #%d in file %s"
                                 % (counter, fs.scannr, fs.filename))

            # grid data
            g2d(fs.xvalues, fs.yvalues, fs.data[counter])

        # return gridded data
        return g2d
