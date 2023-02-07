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
# Copyright (c) 2010-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
Panalytical XML (www.XRDML.com) data file parser

based on the native python xml.dom.minidom module.
want to keep the number of dependancies as small as possible
"""

import os.path
import warnings
from xml.etree import cElementTree as ElementTree

import numpy

from .. import config
from .helper import generate_filenames, xu_open


class XRDMLMeasurement:

    """
    class to handle scans in a XRDML datafile
    """

    def __init__(self, measurement, namespace=''):
        """
        initialization routine for a XRDML measurement which parses are all
        scans within this measurement.
        """

        self.namespace = namespace
        # get scans in <xrdMeasurement>
        slist = measurement.findall(self.namespace + "scan")

        self.hkl = (numpy.nan, numpy.nan, numpy.nan)
        self.material = ""
        self.ddict = {}
        for field in ["countTime", "detector", "counts",
                      "beamAttenuationFactors", "hkl"]:
            self.ddict[field] = []
        is_scalar = 0

        # loop over all scan entries - scan points
        for s in slist:
            # check if scan is complete
            scanstatus = s.get("status")
            if scanstatus in ("Aborted", "Not finished") and len(slist) > 1:
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.io.XRDMLFile: subscan has been aborted "
                          "(part of the data unavailable)!")
            else:
                self.scanmotname = s.get("scanAxis")
                reflection = s.find(self.namespace + "reflection")
                if reflection:
                    m = reflection.find(self.namespace + "material")
                    if m is not None:
                        self.material = m.text
                    hkl = reflection.find(self.namespace + "hkl")
                    if hkl:
                        hkl_h = int(hkl.find(self.namespace + "h").text)
                        hkl_k = int(hkl.find(self.namespace + "k").text)
                        hkl_l = int(hkl.find(self.namespace + "l").text)
                        self.hkl = (hkl_h, hkl_k, hkl_l)
                points = s.find(self.namespace + "dataPoints")

                # add count time to output data
                countTime = points.find(self.namespace +
                                        "commonCountingTime").text
                self.ddict["countTime"].append(float(countTime))

                # check for intensities first to get number of points in scan
                int_elem = points.find(self.namespace + "intensities")
                if int_elem is not None:
                    data = int_elem.text
                    hascounts = False
                else:
                    ct_elem = points.find(self.namespace + "counts")
                    ct_npy = numpy.fromstring(ct_elem.text, sep=" ")
                    self.ddict["counts"].append(ct_npy.tolist())
                    data = ct_elem.text
                    hascounts = True
                # count time normalization; output is counts/sec
                ct_rate = numpy.fromstring(data, sep=" ") / float(countTime)
                nofpoints = ct_rate.size
                # if present read beamAttenuationFactors
                # they are already corrected in the data file, but may be
                # interesting
                attfact = points.find(self.namespace +
                                      "beamAttenuationFactors")
                if attfact is not None:
                    atten = numpy.fromstring(attfact.text, sep=" ")
                    atten_list = atten.tolist()
                    self.ddict["beamAttenuationFactors"].append(atten_list)
                    hasatten = True
                else:
                    hasatten = False

                if hascounts and hasatten:
                    self.ddict["detector"].append((ct_rate*atten).tolist())
                else:
                    self.ddict["detector"].append(ct_rate.tolist())

                # read the axes position
                pos = points.findall(self.namespace + "positions")
                for p in pos:
                    # read axis name and unit
                    aname = p.get("axis")
                    # aunit = p.get("unit")

                    # read axis data
                    listp = p.findall(self.namespace + "listPositions")
                    s = p.findall(self.namespace + "startPosition")
                    e = p.findall(self.namespace + "endPosition")
                    c = p.find(self.namespace + "commonPosition")
                    if listp:  # listPositions
                        listp = listp[0]
                        data_list = numpy.fromstring(listp.text, sep=" ")
                        data_list = data_list.tolist()
                    elif s and e:  # start endPosition
                        data_list = numpy.linspace(
                            float(s[0].text), float(e[0].text),
                            nofpoints).tolist()
                    elif c is not None:  # commonPosition
                        data_list = numpy.fromstring(c.text, sep=" ")
                        data_list = data_list.tolist()
                        is_scalar = 1
                    else:
                        raise ValueError(
                            f"no positions for axis {aname} found")

                    # have to append the data to the data dictionary in case
                    # the scan is complete!
                    if aname not in self.ddict:
                        self.ddict[aname] = []
                    if not is_scalar:
                        self.ddict[aname].append(data_list)
                    else:
                        self.ddict[aname].append(data_list[0])
                        is_scalar = 0

        # finally all scan data needs to be converted to numpy arrays
        for k in self.ddict:
            self.ddict[k] = numpy.array(self.ddict[k])

        # flatten output if only one scan was present
        if len(slist) == 1:
            for k in self.ddict:
                self.ddict[k] = numpy.ravel(self.ddict[k])

        # save scanmot-values and detector counts in special arrays
        if self.scanmotname in ['2Theta-Omega', 'Gonio']:
            self.scanmot = self.ddict['2Theta']
        elif self.scanmotname == 'Omega-2Theta':
            self.scanmot = self.ddict['Omega']
        elif self.scanmotname in self.ddict:
            self.scanmot = self.ddict[self.scanmotname]
        else:
            warnings.warn('XU.io: unknown scan motor name in XRDML-File')
        self.int = self.ddict['detector']

    def __getitem__(self, key):
        return self.ddict[key]

    def __str__(self):
        ostr = "XRDML Measurement\n"
        if self.material:
            ostr += f"Material: '{self.material}'; hkl: {str(self.hkl)}\n"
        for k, v in self.ddict.items():
            ostr += f"{k} with {str(v.shape)} points\n"

        return ostr


class XRDMLFile:

    """
    class to handle XRDML data files. The class is supplied with a file
    name and uses the XRDMLScan class to parse the xrdMeasurement in the
    file
    """

    def __init__(self, fname, path=""):
        """
        initialization routine supplied with a filename
        the file is automatically parsed and the data are available
        in the "scan" object. If more <xrdMeasurement> tags are present, which
        should not be the case, their data is present in the "scans" object.

        Parameters
        ----------
        fname :     str
            filename of the XRDML file
        path :      str, optional
            path to the XRDML file
        """
        if isinstance(fname, str):
            self.full_filename = os.path.join(path, fname)
            self.filename = os.path.basename(self.full_filename)
        else:
            self.full_filename = fname
            self.filename = None
        with xu_open(self.full_filename) as fid:
            d = ElementTree.parse(fid)
        root = d.getroot()
        try:
            namespace = root.tag[:root.tag.index('}')+1]
        except ValueError:
            namespace = ''

        slist = root.findall(namespace+"xrdMeasurement")

        # determine the number of scans in the file
        self.nscans = len(slist)
        self.scans = []
        for s in slist:
            self.scans.append(XRDMLMeasurement(s, namespace))

        if self.nscans == 1:
            self.scan = self.scans[0]

    def __str__(self):
        ostr = f"XRDML File: {self.filename}\n"
        for s in self.scans:
            ostr += s.__str__()

        return ostr


def getxrdml_map(filetemplate, scannrs=None, path=".", roi=None):
    """
    parses multiple XRDML file and concatenates the results for parsing the
    xrayutilities.io.XRDMLFile class is used. The function can be used for
    parsing maps measured with the PIXCel 1D detector (and in limited way also
    for data acquired with a point detector -> see getxrdml_scan instead).

    Parameters
    ----------
    filetemplate :  str
        template string for the file names, can contain a %d or other
        replacement variables which are understood be
        :func:`~xrayutilities.io.helper.generate_filenames`. also see the
        scannrs argument which is used to specify the replacement
        variables.
    scannrs :       int or list, optional
        scan number(s)
    path :          str, optional
        common path to the filenames
    roi :           tuple, optional
        region of interest for the PIXCel detector, for other measurements this
        is not useful!

    Returns
    -------
    om, tt, psd :   ndarray
        motor positions and data as flattened numpy arrays

    Examples
    --------
    >>> om, tt, psd = xrayutilities.io.getxrdml_map("samplename_%d.xrdml",
    ... [1, 2], path="data")  # doctest: +SKIP
    """
    def getOmPixcel(omraw, ttraw):
        """
        function to reshape the Omega values into a form needed for
        further treatment with xrayutilities
        """
        return (omraw[:, numpy.newaxis] * numpy.ones(ttraw.shape)).flatten()

    # read raw data and convert to reciprocal space
    om = numpy.zeros(0)
    tt = numpy.zeros(0)
    psd = numpy.zeros(0)
    # create scan names
    files = generate_filenames(filetemplate, scannrs)

    # parse files
    for f in files:
        d = XRDMLFile(os.path.join(path, f))
        s = d.scan
        if len(s['detector'].shape) == 1:
            raise TypeError("XU.getxrdml_map: This function can only be used "
                            "to parse reciprocal space map files")

        if roi is None:
            roi = [0, s['detector'].shape[1]]
        if s['Omega'].size < s['2Theta'].size:
            om = numpy.concatenate(
                (om, getOmPixcel(s['Omega'], s['2Theta'][:, roi[0]:roi[1]])))
            tt = numpy.concatenate(
                (tt, s['2Theta'][:, roi[0]:roi[1]].flatten()))
        elif s['Omega'].size > s['2Theta'].size:
            om = numpy.concatenate((om, s['Omega'].flatten()))
            tt = numpy.concatenate((
                tt,
                numpy.ravel(s['2Theta'][:, numpy.newaxis] *
                            numpy.ones(s['Omega'].shape))))
        else:
            om = numpy.concatenate((om, s['Omega'].flatten()))
            tt = numpy.concatenate(
                (tt, s['2Theta'][:, roi[0]:roi[1]].flatten()))
        psd = numpy.concatenate(
            (psd, s['detector'][:, roi[0]:roi[1]].flatten()))

    return om, tt, psd


def getxrdml_scan(filetemplate, *motors, **kwargs):
    """
    parses multiple XRDML file and concatenates the results for parsing the
    xrayutilities.io.XRDMLFile class is used. The function can be used for
    parsing arbitrary scans and will return the the motor values of the scan
    motor and additionally the positions of the motors given by in the
    ``*motors`` argument

    Parameters
    ----------
    filetemplate :  str
        template string for the file names, can contain a %d or other
        replacement variables which are understood be
        :func:`~xrayutilities.io.helper.generate_filenames`. also see the
        scannrs keyword argument which is used to specify the replacement
        variables.

    motors :        str
        motor names to return: e.g.: 'Omega', '2Theta', ...  one can also use
        abbreviations:

         - 'Omega' = 'om' = 'o'
         - '2Theta' = 'tt' = 't'
         - 'Chi' = 'c'
         - 'Phi' = 'p'

    scannrs :       int or list, optional
        scan number(s)
    path :          str, optional
        common path to the filenames

    Returns
    -------
    scanmot, mot1, mot2,..., detectorint :  ndarray
        motor positions and data as flattened numpy arrays

    Examples
    --------
    >>> scanmot, om, tt, inte = getxrdml_scan(
    ... "samplename_1.xrdml", 'om', 'tt', path="data")  # doctest: +SKIP
    """
    flatten = True
    # parse keyword arguments
    path = kwargs.get('path', '.')
    scannrs = kwargs.get('scannrs', None)

    validmotors = ['Omega', '2Theta', 'Psi', 'Chi', 'Phi', 'Z', 'X', 'Y']
    validmotorslow = [mot.lower() for mot in validmotors]
    # create correct motor names from input values
    motnames = []
    for mot in motors:
        if mot.lower() in validmotorslow:
            motnames.append(validmotors[validmotorslow.index(mot.lower())])
        elif mot.lower() in ['phi', 'p']:
            motnames.append('Phi')
        elif mot.lower() in ['chi', 'c']:
            motnames.append('Chi')
        elif mot.lower() in ['psi']:
            motnames.append('Psi')
        elif mot.lower() in ['tt', 't']:
            motnames.append('2Theta')
        elif mot.lower() in ['om', 'o']:
            motnames.append('Omega')
        else:
            raise ValueError("XU: invalid motor name given")

    motvals = numpy.empty((len(motnames) + 1, 0))
    detvals = numpy.empty(0)
    # create scan names
    files = generate_filenames(filetemplate, scannrs)

    # parse files
    if len(files) == 1:
        flatten = False
    for f in files:
        d = XRDMLFile(os.path.join(path, f))
        s = d.scan
        detshape = s['detector'].shape

        if len(detshape) == 2:
            angles = numpy.ravel(s.scanmot)
            angles.shape = (1, angles.size)
            for mot in motnames:
                if s[mot].shape != detshape:
                    angles = numpy.vstack((
                        angles,
                        numpy.ravel(s[mot][:, numpy.newaxis] *
                                    numpy.ones(detshape))))
                else:
                    angles = numpy.vstack((angles, numpy.ravel(s[mot])))
            motvals = numpy.concatenate((motvals, angles), axis=1)
            dval = numpy.ravel(s['detector'])
            detvals = numpy.concatenate((detvals, dval))
            if not flatten:
                detvals.shape = detshape
                motvals.shape = (len(motnames) + 1, detshape[0], detshape[1])
        else:
            detvals = numpy.concatenate((detvals, s['detector']))
            angles = s.scanmot
            angles.shape = (1, angles.size)
            for mot in motnames:
                try:
                    angles = numpy.vstack((angles, s[mot]))
                except ValueError:  # motor is not array
                    angles = numpy.vstack(
                        (angles, s[mot] * numpy.ones(detshape)))
            motvals = numpy.concatenate((motvals, angles), axis=1)

    # make return value
    ret = []
    for i in range(motvals.shape[0]):
        ret.append(motvals[i, ...])
    ret.append(detvals)
    return ret
