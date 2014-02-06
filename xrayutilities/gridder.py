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
# Copyright (C) 2009-2010,2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009 Mario Keplinger <mario.keplinger@jku.at>
# Copyright (C) 2009-2013 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from . import cxrayutilities
from . import exception
from . import config

def check_array(a,dtype):
    """
    Check if an array fits the requirements for the C-code and returns it 
    back to the callee. Such arrays must be aligned and C_CONTIGUOUS which
    means that they have to follow C-ordering.

    Parameters
    ----------
    a ............ array to check
    dtype ........ numpy data type
    """

    return numpy.require(a,dtype=dtype,requirements=["ALIGNED","C_CONTIGUOUS"])

def delta(min_value,max_value,n):
    """
    Compute the stepsize along an axis of a grid. 

    Parameters
    ----------
    min_value ........... axis minimum value
    max_value ........... axis maximum value
    n ................... number of steps
    """

    return (max_value-min_value)/(n-1)

def axis(min_value,max_value,n):
    """
    Compute the a grid axis. 

    Parameters
    ----------
    min_value ........... axis minimum value
    max_value ........... axis maximum value
    n ................... number of steps
    """

    d = delta(min_value,max_value,n)
    a = min_value + d*numpy.arange(0,n)

    return a

def ones(*args):
    """
    Compute ones for matrix generation. The shape is determined by the number of
    input arguments.
    """
    return numpy.ones(args,dtype=numpy.double)


class Gridder(object):
    def __init__(self):
        """
        Basis class for gridders in xrayutilities. A gridder is a function mapping
        irregular spaced data onto a regular grid by binning the data into equally 
        sized elements
        """

        self.flags = 0
        self.keep_data = False # by default every call to gridder will start a new gridding
        self.normalize = True
        self.fixed_range = False # flag to allow for sequential gridding with fixed data range

        if config.VERBOSITY >= config.INFO_ALL:
            self.flags = self.flags|16 # set verbosity flag

    def Normalize(self,bool):
        """
        set or unset the normalization flag.
        Normalization needs to be done to obtain proper gridding but may want to be
        disabled in certain cases when sequential gridding is performed
        """
        if not bool==True or bool==False:
            raise TypeError("Normalize flag must be a boolan value (True/False)!")
        self.normalize = bool
        if not bool:
            self.flags = self.flags^4
        else:
            self.flags = self.flags&(255-4)

    def KeepData(self,bool):
        if not bool==True or bool==False:
            raise TypeError("Keep Data flag must be a boolan value (True/False)!")

        print("XU.Gridder: currently KeepData option is not working and will give random results! Please do not use it.")
        self.keep_data = bool

    def Clear(self):
        """
        Clear so far gridded data to reuse this instance of the Gridder
        """
        try:
            self.gdata[...] = 0
            self.gnorm[...] = 0
        except:
            pass

class Gridder1D(Gridder):
    def __init__(self,nx):
        Gridder.__init__(self)

        self.nx = nx
        self.xmin = 0
        self.xmax = 0
        self.gdata = numpy.zeros(nx,dtype=numpy.double)
        self.gnorm = numpy.zeros(nx,dtype=numpy.double)

    def __get_xaxis(self):
        """
        Returns the xaxis of the gridder
        the returned values correspond to the center of the data bins used by
        the numpy.histogram function
        """
        dx = (self.xmax-self.xmin)/(self.nx) # no -1 here to be consistent with numpy.histogram
        ax = self.xmin+dx*numpy.arange(0,self.nx) + dx/2.
        return ax

    def __get_data(self):
        """
        return gridded data (performs normalization if switched on)
        """
        if self.normalize:
            tmp= numpy.copy(self.gdata)
            tmp[self.gnorm!=0] /= self.gnorm[self.gnorm!=0].astype(numpy.float)
            return tmp
        else:
            return self.gdata.copy()

    xaxis = property(__get_xaxis)
    data = property(__get_data)

    def dataRange(self,min,max,fixed=True):
        """
        define minimum and maximum data range, usually this is deduced
        from the given data automatically, however, for sequential 
        gridding it is usefull to set this before the first call of the
        gridder. data outside the range are simply ignored

        Parameters
        ----------
         min:   minimum value of the gridding range
         max:   maximum value of the gridding range
         fixed: flag to turn fixed range gridding on (True (default)) 
                or of (False)
        """
        self.fixed_range = fixed
        self.xmin = float(min)
        self.xmax = float(max)

    def __call__(self,*args):
        """
        Perform gridding on a set of data.

        Parameters
        ----------
         x ............... numpy array of arbitrary shape with x positions
         data ............ numpy array of arbitrary shape with data values
        """

        x = args[0]
        data = args[1]
        x = x.reshape(x.size)
        data = data.reshape(data.size)

        if x.size!=data.size:
            raise exception.InputError("XU.Gridder1D: size of given datasets (x,data) is not equal!")

        # require correct aligned memory for input arrays
        x = numpy.require(x,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        data = numpy.require(data,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])

        # use only non-NaN data values
        mask = numpy.invert(numpy.isnan(data))
        ldata = data[mask]
        lx = x[mask]

        if not self.fixed_range:
            self.xmin = lx.min()
            self.xmax = lx.max()

        # grid the data using numpy histogram
        tmpgdata,bins = numpy.histogram(lx,weights=ldata,bins=self.nx,range=(self.xmin,self.xmax))
        tmpgnorm,bins = numpy.histogram(lx,bins=self.nx,range=(self.xmin,self.xmax))
        if self.keep_data:
            self.gnorm+= tmpgnorm
            self.gdata+= tmpgdata
        else:
            self.gnorm = tmpgnorm
            self.gdata = tmpgdata

