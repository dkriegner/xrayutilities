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
# Copyright (C) 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu
import matplotlib.pyplot as plt


def getimage(fname, hotpixelmap, roi=None):
    cbf = xu.io.CBFFile(fname)
    ccdr = removehotpixel(cbf.data, hotpixelmap)
    ccd = xu.blockAverage2D(ccdr, 1, 1, roi=roi)
    if ccd.max() / ccd.mean() < 1e2:
        print('no clear maximum in %s?' % fname)
    ccd[ccd < 2 * ccd.mean()] = 0  # make center of mass position easier
    return ccd


def removehotpixel(ccd, hotpixelmap):
    """
    Removes hotpixel of a ccd image.  Replaces the hotpixel by the mean value
    of the sourrounding non-hotpixel-pixels.

    Parameters
    ----------
     ccd: ccd image stored in 2D numpy array
     hotpixelmap: a tupel with two numpy arrays with the respective x and y
                  coordinates of the hotpixels.

    Returns
    -------
        2D numpy array with the removed hotpixel image data
    """
    shape = ccd.shape
    ccdb = numpy.ones((shape[0] + 2, shape[1] + 2))
    ccdb[1:-1, 1:-1] = ccd

    hotpixelimage = numpy.zeros(shape)
    hotpixelimage[hotpixelmap] = 1

    hotpixelimageb = numpy.zeros((shape[0] + 2, shape[1] + 2))
    hotpixelimageb[1:-1, 1:-1] = hotpixelimage

    y = hotpixelmap[0]
    x = hotpixelmap[1]

    directions = [(0, 1), (0, -1), (1, 1), (1, -1)]
    neighbors = numpy.zeros(len(directions))
    for xc, yc in zip(x, y):
        c = [yc + 1, xc + 1]
        neighbors[...] = 0

        for j, item in enumerate(directions):
            while hotpixelimageb[c[0], c[1]] != 0:
                c[item[0]] += item[1]
            neighbors[j] = ccdb[c[0], c[1]]
            c = [yc + 1, xc + 1]
        ccd[yc, xc] = neighbors.mean()
    return ccd


datadir = "data"
name = "align"
datfile = os.path.join(datadir, name, name + "_%05d.dat")
ccdfile = os.path.join(datadir, name, name + "_%05d_eiger",
                       name + "_%05d_eiger_%05d.cbf")
fullfilename = ccdfile % (2, 2, 104)


@unittest.skipIf(not os.path.isfile(fullfilename) or True,  # skip by default!
                 "additional test data needed -> ask the authors")
class TestCCD_Normalizer(unittest.TestCase):
    en = 10000.0  # x-ray energy in eV
    roi = (551, 1065, 0, 1030)  # get data of the good part of the detector
    hkls = ((0, 0, 0), (0, 0, 0), (0, 0, 4), (0, 0, 4), (0, 0, 2), (0, 0, 2))
    step = 16  # take only every 8'th point
    slices = (numpy.s_[108::step], numpy.s_[::step],
              numpy.s_[57::step], numpy.s_[::step],
              numpy.s_[56:95:step], numpy.s_[::step])

    @classmethod
    def setUpClass(cls):
        hotpix = xu.io.CBFFile(fullfilename)
        cls.hotpixelmap = numpy.where(hotpix.data > 1)

    def test_area_calib(self):
        scannrs = (2, 4)
        ang1, ang2 = [numpy.empty(0), ] * 2
        r = self.roi
        detdata = numpy.empty((0, r[1] - r[0], r[3] - r[2]))

        for scannr, scanhkl, sl in zip(scannrs, self.hkls, self.slices):
            (tth, tt), angles = xu.io.gettty08_scan(datfile, scannr,
                                                    'tth', 'tt')
            intensity = numpy.zeros((len(angles[sl]),
                                     r[1] - r[0], r[3] - r[2]))
            # read images
            for i, nr in enumerate(angles[sl]['Number']):
                fname = ccdfile % (scannr, scannr, nr)
                intensity[i] = getimage(fname, self.hotpixelmap, roi=self.roi)

            ang1 = numpy.concatenate((ang1, tth[sl]))
            ang2 = numpy.concatenate((ang2, tt[sl]))
            detdata = numpy.concatenate((detdata, intensity))

        # start calibration
        param, eps = xu.analysis.sample_align.area_detector_calib(
            ang1, ang2, detdata, ['z-', 'y-'], 'x+',
            start=(125.2, 5.23, 0, 2.7),
            fix=(True, True, False, False),
            debug=False, plot=False)

        self.assertTrue(True)

    def test_area_calib_hkl(self):
        scannrs = (2, 4, 43, 44, 46, 47)
        sang, ang1, ang2 = [numpy.empty(0), ]*3
        imghkl = numpy.empty((0, 3))
        r = self.roi
        detdata = numpy.empty((0, r[1] - r[0], r[3] - r[2]))

        for scannr, scanhkl, sl in zip(scannrs, self.hkls, self.slices):
            (om, tth, tt), angles = xu.io.gettty08_scan(datfile, scannr,
                                                        'om', 'tth', 'tt')
            intensity = numpy.zeros((len(angles[sl]),
                                     r[1] - r[0], r[3] - r[2]))
            # read images
            for i, nr in enumerate(angles[sl]['Number']):
                fname = ccdfile % (scannr, scannr, nr)
                intensity[i] = getimage(fname, self.hotpixelmap, roi=self.roi)

            sang = numpy.concatenate((sang, om[sl]))
            ang1 = numpy.concatenate((ang1, tth[sl]))
            ang2 = numpy.concatenate((ang2, tt[sl]))
            detdata = numpy.concatenate((detdata, intensity))
            imghkl = numpy.concatenate((imghkl, (scanhkl,)*len(angles[sl])))

        # start calibration
        GaAs = xu.materials.GaAs
        qconv = xu.experiment.QConversion(['z+', 'y-', 'x+', 'z+'],
                                          ['z-', 'y-'], [1, 0, 0])
        hxrd = xu.HXRD(GaAs.Q(1, 1, 0), GaAs.Q(0, 0, 1),
                       en=self.en, qconv=qconv)

        param, eps = xu.analysis.sample_align.area_detector_calib_hkl(
            sang, ang1, ang2, detdata, imghkl, hxrd, GaAs,
            ['z-', 'y-'], 'x+',
            start=(0, 0, -0.3, 0, 0, 0, xu.en2lam(self.en)),
            fix=(False, False, False, False, False, False, False),
            debug=False, plot=False)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
