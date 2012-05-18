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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@aol.at>

import numpy
import time
import os
import subprocess

# relative imports from xrutils
from .. import config
from ..exception import InputError

class RoperCCD(object):
    """
    parse RoperScientific CCD frames (*.bin) to numpy arrays
    Ignore the header since it seems to contain no useful data

    The routine was tested only for files with 4096x4096 pixel images
    created at Hasylab Hamburg which save an 16bit integer per point.
    """
    def __init__(self,flatfield=None,darkfield=None,nop1=4096,nop2=4096):
        """
        initialize the RoperCCD reader, which includes setting the dimension of
        the images as well as defining the data used for flat- and darkfield
        correction!

        Parameter
        ---------
         flatfield: filename or data for flatfield correction. supported file
                    types include (*.bin *.bin.xz and *.npy files). otherwise
                    a 2D numpy array should be given
         darkfield: filename or data for darkfield correction. same types as
                    for flat field are supported.
         nop1,nop2: number of pixels in the first and second dimension of the
                    image
        """

        # save number of pixels per image
        self.nop1= nop1
        self.nop2= nop2
        self.dtype = numpy.int16

        # read flatfield
        if flatfield:
            if isinstance(flatfield,str):
                if os.path.splitext(flatfield)[1] == '.npy':
                    self.flatfield = numpy.load(flatfield)
                elif os.path.splitext(flatfield)[1] == '.xz' or os.path.splitext(flatfield)[1] == '.bin':
                    self.filename = self.readImage(flatfield) # read without flatc and darkc
                else:
                    raise InputError("Error: unknown filename for flatfield correction!")
            elif isinstance(flatfield,numpy.ndarray):
                self.flatfield = flatfield
            else:
                raise InputError("Error: unsupported type for flatfield correction!")

        # read darkfield
        if darkfield:
            if isinstance(darkfield,str):
                if os.path.splitext(darkfield)[1] == '.npy':
                    self.darkfield = numpy.load(darkfield)
                elif os.path.splitext(darkfield)[1] == '.xz' or os.path.splitext(darkfield)[1] == '.bin':
                    self.filename = self.readImage(darkfield) # read without flatc and darkc
                else:
                    raise InputError("Error: unknown filename for darkfield correction!")
            elif isinstance(darkfield,numpy.ndarray):
                self.darkfield = darkfield
            else:
                raise InputError("Error: unsupported type for darkfield correction!")

        if flatfield:
            self.flatc = True
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.RoperCCD: flatfield correction enabled")
        if darkfield:
            self.darkc = True
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.RoperCCD: darkfield correction enabled")


    def readImage(self,filename):
        """
        read RoperScientic image file
        and correct for dark- and flatfield in case the necessary data are
        available.

        returned data = ((image data)-(darkfield))/flatfield*average(flatfield)

        Parameter
        ---------
         filename: filename of the image to be read. so far only single
                   filenames are supported. The data might be compressed.
                   supported extensions: .bin and .bin.xz
        """

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.RoperCCD.readImage: file %s"%(filename))
            t1 = time.time()

        if filename[-2:] == 'xz':
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.RoperCCD.readImage: uncompressing file %s"%(filename))

            subprocess.call("xz --decompress --verbose --keep %s" %(filename),shell=True)
            fh = open(filename[:-3], 'rb')
        else:
            fh = open(filename, 'rb')

        # jump over header
        fh.seek(216)
        # read image
        img = numpy.fromfile(fh,dtype=self.dtype)
        img.shape = (self.nop1,self.nop2) # reshape the data
        # darkfield correction
        if self.darkc:
            imgf = (img-self.darkfield).astype(numpy.float32)
        else: imgf = img.astype(numpy.float32)
        # kill negativ pixels
        numpy.clip(imgf,1e-6,numpy.Inf,out=imgf)
        # flatfield correction
        if self.flatc:
            imgf = imgf/self.flatfield
        fh.close()
        if os.path.splitext(filename)[1] == '.xz':
            subprocess.call(["rm","%s"%(filename[:-3])])

        if config.VERBOSITY >= config.INFO_ALL:
            t2 = time.time()
            print("XU.io.RoperCCD.readImage: parsing time %8.3f"%(t2-t1))

        return imgf


class ImageReader(object):
    """
    parse CCD frames in the form of tiffs or binary data (*.bin) 
    to numpy arrays. ignore the header since it seems to contain 
    no useful data

    The routine was tested so far with
     RoperScientific files with 4096x4096 pixels created at Hasylab Hamburg,
     which save an 16bit integer per point.
     Perkin Elmer images created at Hasylab Hamburg with 2048x2048 pixels.
    """
    def __init__(self,nop1,nop2,hdrlen=0,flatfield=None,darkfield=None,dtype=numpy.int16,byte_swap=False):
        """
        initialize the ImageReader reader, which includes setting the dimension of
        the images as well as defining the data used for flat- and darkfield
        correction!

        Parameter
        ---------
         nop1,nop2: number of pixels in the first and second dimension of the
                    image
         hdrlen:    length of the file header which should be ignored
         flatfield: filename or data for flatfield correction. supported file
                    types include (*.bin *.bin.xz and *.npy files). otherwise
                    a 2D numpy array should be given
         darkfield: filename or data for darkfield correction. same types as
                    for flat field are supported.
         dtype:     datatype of the stored values (default: numpy.int16)
         byte_swap: flag which determines bytes are swapped after reading
        """

        # save number of pixels per image
        self.nop1= nop1
        self.nop2= nop2
        self.dtype = dtype
        self.hdrlen = hdrlen
        self.byteswap = byte_swap

        # read flatfield
        if flatfield:
            if isinstance(flatfield,str):
                if os.path.splitext(flatfield)[1] == '.npy':
                    self.flatfield = numpy.load(flatfield)
                elif os.path.splitext(flatfield)[1] in ['.xz','.bin','.tif']:
                    self.filename = self.readImage(flatfield) # read without flatc and darkc
                else:
                    raise InputError("Error: unknown filename for flatfield correction!")
            elif isinstance(flatfield,numpy.ndarray):
                self.flatfield = flatfield
            else:
                raise InputError("Error: unsupported type for flatfield correction!")

        # read darkfield
        if darkfield:
            if isinstance(darkfield,str):
                if os.path.splitext(darkfield)[1] == '.npy':
                    self.darkfield = numpy.load(darkfield)
                elif os.path.splitext(flatfield)[1] in ['.xz','.bin','.tif']:
                    self.filename = self.readImage(darkfield) # read without flatc and darkc
                else:
                    raise InputError("Error: unknown filename for darkfield correction!")
            elif isinstance(darkfield,numpy.ndarray):
                self.darkfield = darkfield
            else:
                raise InputError("Error: unsupported type for darkfield correction!")

        if flatfield:
            self.flatc = True
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.ImageReader: flatfield correction enabled")
        else: 
            self.flatc = False
        if darkfield:
            self.darkc = True
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.ImageReader: darkfield correction enabled")
        else:
            self.darkc = False


    def readImage(self,filename):
        """
        read image file
        and correct for dark- and flatfield in case the necessary data are
        available.

        returned data = ((image data)-(darkfield))/flatfield*average(flatfield)

        Parameter
        ---------
         filename: filename of the image to be read. so far only single
                   filenames are supported. The data might be compressed.
                   supported extensions: .tiff, .bin and .bin.xz
        """

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.ImageReader.readImage: file %s"%(filename))
            t1 = time.time()

        if filename[-2:] == 'xz':
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.ImageReader.readImage: uncompressing file %s"%(filename))

            subprocess.call("xz --decompress --verbose --keep %s" %(filename),shell=True)
            fh = open(filename[:-3], 'rb')
        else:
            fh = open(filename, 'rb')

        # jump over header
        fh.seek(self.hdrlen)
        # read image
        img = numpy.fromfile(fh,dtype=self.dtype,count=self.nop1*self.nop2)
        if self.byteswap:
            img = img.byteswap()
        img.shape = (self.nop1,self.nop2) # reshape the data
        # darkfield correction
        if self.darkc:
            imgf = (img-self.darkfield).astype(numpy.float32)
        else: imgf = img.astype(numpy.float32)
        # kill negativ pixels
        #numpy.clip(imgf,1e-6,numpy.Inf,out=imgf)
        # flatfield correction
        if self.flatc:
            imgf = imgf/self.flatfield
        fh.close()
        if os.path.splitext(filename)[1] == '.xz':
            subprocess.call(["rm","%s"%(filename[:-3])])

        if config.VERBOSITY >= config.INFO_ALL:
            t2 = time.time()
            print("XU.io.ImageReader.readImage: parsing time %8.3f"%(t2-t1))

        return imgf
