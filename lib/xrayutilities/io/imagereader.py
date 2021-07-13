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
# Copyright (C) 2012-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import time

import numpy

from .. import config
from ..exception import InputError
# relative imports from xrayutilities
from .helper import xu_open


class ImageReader(object):

    """
    parse CCD frames in the form of tiffs or binary data (``*.bin``)
    to numpy arrays. ignore the header since it seems to contain
    no useful data

    The routine was tested so far with

    1. RoperScientific files with 4096x4096 pixels created at Hasylab Hamburg,
       which save an 16bit integer per point.
    2. Perkin Elmer images created at Hasylab Hamburg with 2048x2048 pixels.
    """

    def __init__(self, nop1, nop2, hdrlen=0, flatfield=None, darkfield=None,
                 dtype=numpy.int16, byte_swap=False):
        """
        initialize the ImageReader reader, which includes setting the dimension
        of the images as well as defining the data used for flat- and darkfield
        correction!

        Parameters
        ----------
        nop1, nop2 :    int
            number of pixels in the first and second dimension of the image
        hdrlen :        int, optional
            length of the file header which should be ignored
        flatfield :     str or ndarray, optional
            filename or data for flatfield correction. supported file types
            include (.bin/.tif (also compressed .xz or .gz) and .npy files).
            otherwise a 2D numpy array should be given
        darkfield :     str or ndarray, optional
            filename or data for darkfield correction. same types as for flat
            field are supported.
        dtype :         numpy.dtype, optional
            datatype of the stored values (default: numpy.int16)
        byte_swap :     bool, optional
            flag which determines bytes are swapped after reading
        """

        # save number of pixels per image
        self.nop1 = nop1
        self.nop2 = nop2
        self.dtype = dtype
        self.hdrlen = hdrlen
        self.byteswap = byte_swap

        # read flatfield
        if flatfield:
            if isinstance(flatfield, str):
                if os.path.splitext(flatfield)[1] == '.npy':
                    self.flatfield = numpy.load(flatfield)
                elif os.path.splitext(flatfield)[1] in \
                        ['.gz', '.xz', '.bin', '.tif']:
                    # read without flatc and darkc
                    self.flatfield = self.readImage(flatfield)
                else:
                    raise InputError("Error: unknown filename for "
                                     "flatfield correction!")
            elif isinstance(flatfield, numpy.ndarray):
                self.flatfield = flatfield
            else:
                raise InputError("Error: unsupported type for "
                                 "flatfield correction!")

        # read darkfield
        if darkfield:
            if isinstance(darkfield, str):
                if os.path.splitext(darkfield)[1] == '.npy':
                    self.darkfield = numpy.load(darkfield)
                elif os.path.splitext(darkfield)[1] in \
                        ['.gz', '.xz', '.bin', '.tif']:
                    # read without flatc and darkc
                    self.darkfield = self.readImage(darkfield)
                else:
                    raise InputError("Error: unknown filename for "
                                     "darkfield correction!")
            elif isinstance(darkfield, numpy.ndarray):
                self.darkfield = darkfield
            else:
                raise InputError("Error: unsupported type for "
                                 "darkfield correction!")

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

    def readImage(self, filename, path=None):
        """
        read image file
        and correct for dark- and flatfield in case the necessary data are
        available.

        returned data = ((image data)-(darkfield))/flatfield*average(flatfield)

        Parameters
        ----------
        filename :  str
            filename of the image to be read. so far only single filenames are
            supported. The data might be compressed.  supported extensions:
            .tif, .bin and .bin.xz
        path :      str, optional
            path of the data files
        """
        if path:
            full_filename = os.path.join(path, filename)
        else:
            full_filename = filename

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.ImageReader.readImage: file %s" % (full_filename))
            t1 = time.time()

        with xu_open(full_filename) as fh:
            # jump over header
            fh.seek(self.hdrlen)
            # read image
            rlen = numpy.dtype(self.dtype).itemsize * self.nop1 * self.nop2
            img = numpy.frombuffer(fh.read(rlen), dtype=self.dtype)
            if self.byteswap:
                img = img.byteswap()
            img.shape = (self.nop1, self.nop2)  # reshape the data
            # darkfield correction
            if self.darkc:
                img = (img - self.darkfield).astype(numpy.float32)
            # flatfield correction
            if self.flatc:
                img = img.astype(numpy.float32) / self.flatfield

        if config.VERBOSITY >= config.INFO_ALL:
            t2 = time.time()
            print("XU.io.ImageReader.readImage: parsing time %8.3f"
                  % (t2 - t1))

        return img


dlen = {'char': 1,
        'byte': 1,
        'word': 2,
        'dword': 4,
        'rational': 8,  # not implemented correctly
        'float': 4,
        'double': 8}

dtypes = {1: 'byte',
          2: 'char',
          3: 'word',
          4: 'dword',
          5: 'rational',  # not implemented correctly
          6: 'byte',
          7: 'byte',
          8: 'word',
          9: 'dword',
          10: 'rational',  # not implemented correctly
          11: 'float',
          12: 'double'}

nptyp = {1: numpy.byte,
         2: numpy.char,
         3: numpy.uint16,
         4: numpy.uint32,
         5: numpy.uint32,
         6: numpy.int8,
         7: numpy.byte,
         8: numpy.int16,
         9: numpy.int32,
         10: numpy.int32,
         11: numpy.float32,
         12: numpy.float64}

tiffdtype = {1: {8:  numpy.uint8, 16: numpy.uint16, 32: numpy.uint32},
             2: {8:  numpy.int8, 16: numpy.int16, 32: numpy.int32},
             3: {16: numpy.float16, 32: numpy.float32}}

tifftags = {256: 'ImageWidth',  # width
            257: 'ImageLength',  # height
            258: 'BitsPerSample',
            259: 'Compression',
            262: 'PhotometricInterpretation',
            270: 'ImageDescription',
            272: 'Model',
            273: 'StripOffsets',
            278: 'RowsPerStrip',
            279: 'StripByteCount',
            282: 'XResolution',
            283: 'YResolution',
            305: 'Software',
            306: 'DateTime',
            315: 'Artist',
            339: 'SampleFormat',
            36864: 'ExifVersion'}


class TIFFRead(ImageReader):
    """
    class to Parse a TIFF file including extraction of information from the
    file header in order to determine the image size and data type

    The data stored in the image are available in the 'data' property.
    """

    def __init__(self, filename, path=None):
        """
        initialization of the class which will prepare the parser and parse
        the files content into class properties

        Parameters
        ----------
        filename :  str
            file name of the TIFF-like image file
        path :      str, optional
            path of the data file
        """
        if path:
            full_filename = os.path.join(path, filename)
        else:
            full_filename = filename

        with xu_open(full_filename, 'rb') as fh:
            self.byteorder = fh.read(2*dlen['char'])
            self.version = numpy.frombuffer(fh.read(dlen['word']),
                                            dtype=numpy.uint16)[0]
            if self.byteorder not in (b'II', b'MM') or self.version != 42:
                raise TypeError("Not a TIFF file (%s)" % filename)
            if self.byteorder != b'II':
                raise NotImplementedError("The 'MM' byte order is not yet "
                                          "implemented, please file a bug!")

            fh.seek(4)
            self.ifdoffset = numpy.frombuffer(fh.read(dlen['dword']),
                                              dtype=numpy.uint32)[0]
            fh.seek(self.ifdoffset)

            self.ntags = numpy.frombuffer(fh.read(dlen['word']),
                                          dtype=numpy.uint16)[0]

            self._parseImgTags(fh, self.ntags)

            fh.seek(self.ifdoffset + 2 + 12 * self.ntags)
            nextimgoffset = numpy.frombuffer(fh.read(dlen['dword']),
                                             dtype=numpy.uint32)[0]
            if nextimgoffset != 0:
                raise NotImplementedError("Multiple images per file are not "
                                          "supported, please file a bug!")

            # check if image type is supported
            if self.imgtags.get('Compression', 1) != 1:
                raise NotImplementedError("Compression is not supported, "
                                          "please file a bug report!")
            if self.imgtags.get('PhotometricInterpretation', 0) not in (0, 1):
                raise NotImplementedError("RGB and colormap is not supported")

        sf = self.imgtags.get('SampleFormat', 1)
        bs = self.imgtags.get('BitsPerSample', 1)
        if isinstance(self.imgtags['StripOffsets'], numpy.ndarray):
            hdrlen = self.imgtags['StripOffsets'][0]
        else:
            hdrlen = self.imgtags['StripOffsets']
        ImageReader.__init__(self,
                             self.imgtags['ImageLength'],
                             self.imgtags['ImageWidth'],
                             hdrlen=hdrlen,
                             dtype=tiffdtype[sf][bs],
                             byte_swap=False)

        self.data = self.readImage(filename, path)

    def _parseImgTags(self, fh, ntags):
        """
        parse TIFF image tags from Image File Directory header

        Parameters
        ----------
        fh :    file-handle
            file handle of the TIFF file
        ntags : int
            number of tags in the Image File Directory
        """

        self.imgtags = {}
        for i in range(ntags):
            ftag = numpy.frombuffer(fh.read(dlen['word']),
                                    dtype=numpy.uint16)[0]
            ftype = numpy.frombuffer(fh.read(dlen['word']),
                                     dtype=numpy.uint16)[0]
            flength = numpy.frombuffer(fh.read(dlen['dword']),
                                       dtype=numpy.uint32)[0]
            fdoffset = numpy.frombuffer(fh.read(dlen['dword']),
                                        dtype=numpy.uint32)[0]

            pos = fh.tell()
            if flength*dlen[dtypes[ftype]] <= 4:
                fdoffset = pos - dlen['dword']
            fh.seek(fdoffset)
            if ftype == 2:
                fdata = fh.read(flength * dlen[dtypes[ftype]])
                fdata = fdata.split(b'\x00')[0].decode("ascii",
                                                       errors="replace")
            else:
                rlen = flength * dlen[dtypes[ftype]]
                fdata = numpy.frombuffer(fh.read(rlen), dtype=nptyp[ftype])
            if flength == 1:
                fdata = fdata[0]
            fh.seek(pos)

            # add field to tags
            self.imgtags[tifftags.get(ftag, ftag)] = fdata


class PerkinElmer(ImageReader):
    """
    parse PerkinElmer CCD frames (``*.tif``) to numpy arrays
    Ignore the header since it seems to contain no useful data

    The routine was tested only for files with 2048x2048 pixel images
    created at Hasylab Hamburg which save an 32bit float per point.
    """

    def __init__(self, **keyargs):
        """
        initialize the PerkinElmer reader, which includes setting the dimension
        of the images as well as defining the data used for flat- and darkfield
        correction!

        Parameters
        ----------
        flatfield :     str or ndarray, optional
            filename or data for flatfield correction. supported file types
            include (.bin .bin.xz and .npy files). otherwise a 2D numpy
            array should be given
        darkfield :     str or ndarray, optional
            filename or data for darkfield correction. same types as for flat
            field are supported.
        """

        ImageReader.__init__(self, 2048, 2048, hdrlen=8, dtype=numpy.float32,
                             byte_swap=False, **keyargs)


class RoperCCD(ImageReader):

    """
    parse RoperScientific CCD frames (``*.bin``) to numpy arrays
    Ignore the header since it seems to contain no useful data

    The routine was tested only for files with 4096x4096 pixel images
    created at Hasylab Hamburg which save an 16bit integer per point.
    """

    def __init__(self, **keyargs):
        """
        initialize the RoperCCD reader, which includes setting the dimension of
        the images as well as defining the data used for flat- and darkfield
        correction!

        Parameters
        ----------
        flatfield :     str or ndarray, optional
            filename or data for flatfield correction. supported file types
            include (.bin .bin.xz and .npy files). otherwise a 2D numpy
            array should be given
        darkfield :     str or ndarray, optional
            filename or data for darkfield correction. same types as for flat
            field are supported.
        """

        ImageReader.__init__(self, 4096, 4096, hdrlen=216, dtype=numpy.int16,
                             byte_swap=False, **keyargs)


class Pilatus100K(ImageReader):
    """
    parse Dectris Pilatus 100k frames (``*.tiff``) to numpy arrays
    Ignore the header since it seems to contain no useful data
    """

    def __init__(self, **keyargs):
        """
        initialize the Piulatus100k reader, which includes setting the
        dimension of the images as well as defining the data used for flat- and
        darkfield correction!

        Parameters
        ----------
        flatfield :     str or ndarray, optional
            filename or data for flatfield correction. supported file types
            include (.bin .bin.xz and .npy files). otherwise a 2D numpy
            array should be given
        darkfield :     str or ndarray, optional
            filename or data for darkfield correction. same types as for flat
            field are supported.
        """

        ImageReader.__init__(self, 195, 487, hdrlen=4096, dtype=numpy.int32,
                             byte_swap=False, **keyargs)


def get_tiff(filename, path=None):
    """
    read tiff image file and return the data

    Parameters
    ----------
    filename :  str
        filename of the image to be read. so far only single filenames are
        supported. The data might be compressed.
    path :      str, optional
        path of the data file
    """

    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.io.get_tiff: file %s" % (filename))
        t1 = time.time()

    t = TIFFRead(filename, path=path)

    if config.VERBOSITY >= config.INFO_ALL:
        t2 = time.time()
        print("XU.io.get_tiff: parsing time %8.3f" % (t2 - t1))

    return t.data
