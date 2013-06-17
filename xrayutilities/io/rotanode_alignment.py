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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
parser for the alignment log file of the rotating anode
"""

import numpy
import re

from .. import config

try:
    from matplotlib import pylab as plt
except RuntimeError:
    if config.VERBOSITY >= config.INFO_ALL:
        print("rotanode_alignment: warning; plotting functionality not available")

LOG_comment = re.compile(r"^#C")
LOG_peakname = re.compile(r"^#P")
LOG_motorname = re.compile(r"^#M")
LOG_datetime = re.compile(r"^#D")
LOG_tagline = re.compile(r"^#")
LOG_num_value = re.compile(r"[+-]*\d*\.*\d*e*[+-]*\d+") #denotes a numeric value

class RA_Alignment(object):
    """
    class to parse the data file created by the alignment routine
    (tpalign) at the rotating anode spec installation

    this routine does an iterative alignment procedure and saves the
    center of mass values were it moves after each scan. It iterates
    between two different peaks and iteratively aligns at each peak between
    two different motors (om/chi at symmetric peaks, om/phi at asymmetric
    peaks)
    """

    def __init__(self,filename):
        """
        initialization function to initialize the objects variables and
        opens the file

        Parameters
        ----------
         filename:  filename of the alignment log file
        """

        self.filename = filename
        try:
            self.fid = open(self.filename,'r')
        except:
            self.fid = None
            raise IOError("error opening alignment log file %s" %self.filename)

        self.peaks = []
        self.alignnames = []
        self.motorpos = []
        self.intensities = []
        self.iterations = []

        self.Parse()

    def Parse(self):
        """
        parser to read the alignment log and obtain the aligned values
        at every iteration.
        """

        currentpeakname = None
        currentmotname = None
        opencommenttag = False
        dataline = False
        iteration = 0

        if self.fid == None:
            raise Exception("RA_Alignment: file was not opened by initialization!")

        for line in self.fid.readlines():
            # for loop to read every line in the file

            # check for new tag in the current line
            if LOG_tagline.match(line):
                opencommenttag = False

                if LOG_comment.match(line):
                    # comment line or block starts
                    opencommenttag = True
                    continue

                elif LOG_datetime.match(line):
                    # data is so far ignored
                    continue

                elif LOG_peakname.match(line):
                    # line with peak name found
                    pname = LOG_peakname.sub("",line)
                    pname = pname.strip()
                    # check if we found a new peakname
                    try: self.peaks.index(pname)
                    except ValueError:
                        self.peaks.append(pname)
                    currentpeakname = pname # set current peak name
                    iteration +=1 # increment iteration counter

                elif LOG_motorname.match(line):
                    # line with motorname is found
                    motname = LOG_motorname.sub("",line)
                    motname = motname.strip()
                    # check if a peakname is already set
                    if currentpeakname == None:
                        if config.VERBOSITY >= config.INFO_LOW:
                            print("RA_Alignment: Warning: a peakname should be given before a motor data line")
                        currentpeakname = "somepeak"
                    currentmotname = currentpeakname + "_" + motname
                    # check if we found a new peak/motor name combination
                    try: self.alignnames.index(currentmotname)
                    except ValueError:
                        # new peak/motor combination
                        self.alignnames.append(currentmotname)
                        # create necessary data structures
                        self.motorpos.append([])
                        self.intensities.append([])
                        self.iterations.append([])
                    dataline = True # next line contains motor position and intensity
            elif opencommenttag:
                # ignore line because it is part of a comment block
                continue

            elif dataline:
                # dataline with motorposition and intensity is found
                line_list = LOG_num_value.findall(line)
                idx = self.alignnames.index(currentmotname)
                self.motorpos[idx].append(float(line_list[0]))
                self.intensities[idx].append(float(line_list[1]))
                self.iterations[idx].append(iteration)
                dataline = False

        # convert data to numpy array and combine position and intensity
        self.data = []
        for i,k in enumerate(self.keys()):
            self.data.append(numpy.array((self.motorpos[i],self.intensities[i],self.iterations[i])))

    def __str__(self):
        """
        returns a string describing the content of the alignment file
        """
        ostr = ""
        ostr += "Peaknames: " + repr(self.peaks) + "\n"
        ostr += "aligned values: " + repr(self.alignnames)
        return ostr

    def __del__(self):
        try: fid.close()
        except: pass

    def keys(self):
        """
        returns a list of keys for which aligned values were parsed
        """
        return self.alignnames

    def get(self,key):
        return self.__getitem__(key)

    def __getitem__(self,key):
        """
        returns the values to the corresponding key
        """
        if key in self.alignnames:
            i = self.alignnames.index(key)
            return self.data[i]
        else:
            raise KeyError("RA_Alignment: unknown key given!")

    def plot(self,pname):
        """
        function to plot the alignment history for a given peak

        Parameters
        ----------
         pname:  peakname for which the alignment should be plotted
        """

        try: plt.__version__
        except NameError:
            print("RA_Alignment.plot: error: plot functionality not available")
            return

        if not pname in self.peaks:
            print ("RA_Alignment.plot: error peakname not found!")
            return

        # get number aligned axis for the current peak
        axnames = []
        for k in self.keys():
            if k.find(pname)>=0:
                axnames.append(k)

        fig,ax = plt.subplots(nrows=len(axnames),sharex=True)

        for an,axis in zip(axnames,ax):
            d = self.get(an)
            plt.sca(axis)
            plt.plot(d[2],d[0],'k.-')
            plt.ylabel(re.sub(pname+"_","",an))
            twax = axis.twinx()
            plt.plot(d[2],d[1],'r.-')
            plt.ylabel("Int (cps)")
            plt.grid()

        plt.xlabel("Peak iteration number")
        plt.suptitle(pname)

