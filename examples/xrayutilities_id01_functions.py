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
# Copyright (C) 2012,2018 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrayutilities_example_plot_3D_ESRF_ID01.py

import xrayutilities as xu
import re
import collections
from matplotlib.pylab import *

# define root of the local data directory (needed because we assume the data
# path of detector frames from the specfile are not correct anymore)
datadir = '/local/data/path'
repl_n = len(datadir.split('/'))

# define intensity normalizer class to normalize for count time and
# monitor changes: to have comparable absolute intensities set the keyword
# argument av_mon to a fixed value, otherwise different scans can not be
# compared!
xid01_normalizer = xu.IntensityNormalizer(
    'CCD', time='Seconds', mon='exp1', av_mon=10000.0)


def deadpixelkill(ccdraw, f=1.0):
    """
    fill empty "spacer" pixels of the maxipix 2x2 detector
    """
    ccd = ccdraw.astype(float32)
    ccd[255:258, :] = ccd[255, :]/3*f
    ccd[258:261, :] = ccd[260, :]/3*f
    ccd[:, 255:258] = ones((ccd.shape[0], 3)) * (ccd[:, 255])[:, newaxis]/3*f
    ccd[:, 258:261] = ones((ccd.shape[0], 3)) * (ccd[:, 260])[:, newaxis]/3*f
    return ccd


def getmpx4_filetmp(specscan, replace=repl_n, key='ULIMA_mpx4', ext='.edf.gz',
                    datadir=datadir):
    """
    read MaxiPix file template from the scan header.

    Parameters
    ----------
     specscan:  SPECScan object from which the header should be read
     replace:   number of folders which should be replaced by the new data
                directory
     key:       spec-header key name for the file template
     ext:       file extension of the EDF files
     datadir:   new root of the data directory

    Returns
    -------
     file template for the maxipix pictures
    """
    ret = specscan.getheader_element(key)
    ret = xu.utilities.exchange_path(ret, datadir, replace=replace)
    fn = ret[:-len(ext)]
    sfn = fn.split('_')
    return '_'.join(sfn[:-1]) + '_%05d' + ext


def getmono_energy(specscan, key='UMONO', motor='mononrj'):
    """
    read the monochromator energy from the spec-scan header

    Parameters
    ----------
     specscan:  SPECScan object from which the header should be read
     key:       spec-header key name for the monochromator parameters
     motor:     name of the energy motor

    Returns
    -------
     energy, float in eV
    """
    ret = specscan.getheader_element(key)
    motors = ret.split(',')
    re_val_unit = re.compile(r'([0-9.]*)([a-zA-Z]*)')
    for m in motors:
        mn, mv = m.split('=')
        if mn == motor:
            match = re_val_unit.match(mv)
            en = float(match.groups()[0])*1e3
    return en


def rawmap(specfile, scannr, experiment, angdelta=None, U=identity(3),
           norm=True):
    """
    read ccd frames and and convert them in reciprocal space
    angular coordinates are taken from the spec file. A single
    or multiple scan-numbers can be given.
    """

    [mu, eta, phi, nu, delta, ty, tz, ccdn] = xu.io.getspec_scan(
        specfile, scannr, 'mu', 'eta', 'phi', 'nu',
        'del', 'mpxy', 'mpxz', 'mpx4inr')

    idx = 0
    nav = experiment._A2QConversion._area_nav
    roi = experiment._A2QConversion._area_roi

    if not isinstance(scannr, collections.Iterable):
        scannr = [scannr, ]
    for snr in scannr:
        specscan = getattr(specfile, 'scan%d' % snr)
        ccdfiletmp = getmpx4_filetmp(specscan)
        en = getmono_energy(specscan)
        experiment.energy = en

        for j in range(len(specscan.data)):
            i = ccdn[idx]
            # read ccd image from EDF file
            e = xu.io.EDFFile(ccdfiletmp % i)
            ccd = deadpixelkill(e.data)

            # normalize ccd-data (optional)
            # create data for normalization
            if norm:
                d = {'CCD': ccd, 'exp1': specscan.data['exp1'][idx],
                     'Seconds': specscan.data['Seconds'][idx]}
                ccd = xid01_normalizer(d)
            CCD = xu.blockAverage2D(ccd, nav[0], nav[1], roi=roi)

            if idx == 0:
                intensity = zeros((len(ccdn), ) + CCD.shape)

            intensity[idx, :, :] = CCD
            idx += 1

    # transform scan angles to reciprocal space coordinates for all detector
    # pixels
    if angdelta is None:
        qx, qy, qz = experiment.Ang2Q.area(mu, eta, phi, nu, delta,
                                           ty, tz, UB=U)
    else:
        qx, qy, qz = experiment.Ang2Q.area(mu, eta, phi, nu, delta,
                                           ty, tz, delta=angdelta, UB=U)

    return qx, qy, qz, intensity


def gridmap(specfile, scannr, experiment, nx, ny, nz, **kwargs):
    """
    read ccd frames and grid them in reciprocal space
    angular coordinates are taken from the spec file

    **kwargs are passed to the rawmap function
    """

    qx, qy, qz, intensity = rawmap(specfile, scannr, experiment, **kwargs)

    # convert data to rectangular grid in reciprocal space
    gridder = xu.Gridder3D(nx, ny, nz)
    gridder(qx, qy, qz, intensity)

    return gridder.xaxis, gridder.yaxis, gridder.zaxis, gridder.data, gridder
