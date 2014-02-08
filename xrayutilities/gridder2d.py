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

from .gridder import Gridder
from .gridder import check_array
from .gridder import delta
from .gridder import axis
from .gridder import ones

class Gridder2D(Gridder):
    def __init__(self,nx,ny):
        Gridder.__init__(self)

        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0

        self.nx = nx
        self.ny = ny

        self._allocate_memory()

    def _allocate_memory(self):
        """
        Class method to allocate memory for the gridder based on the nx,ny 
        class attributes.
        """
        
        self.gdata = numpy.zeros((self.nx,self.ny),dtype=numpy.double)
        self.gdata = check_array(self.gdata,numpy.double)
        self.gnorm = numpy.zeros((self.nx,self.ny),dtype=numpy.double)
        self.gnorm = check_array(self.gnorm,numpy.double)


    def SetResolution(self,nx,ny):
        """
        Reset the resolution of the gridder. In this case the original data
        stored in the object will be deleted. 

        Parameters
        ----------
        nx ............ number of points in x-direction
        ny ............ number of points in y-direction
        """
        self.nx = nx
        self.ny = ny

        self._allocate_memory()

    def __get_xaxis(self):
        return axis(self.xmin,self.xmax,self.nx)

    def __get_yaxis(self):
        return axis(self.ymin,self.ymax,self.ny)

    def __get_xmatrix(self):
        return ones(self.nx,self.ny)*self.xaxis[:,numpy.newaxis]

    def __get_ymatrix(self):
        return ones(self.nx,self.ny)*self.yaxis[numpy.newaxis,:]

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

    yaxis = property(__get_yaxis)
    xaxis = property(__get_xaxis)
    xmatrix = property(__get_xmatrix)
    ymatrix = property(__get_ymatrix)
    data = property(__get_data)

    def dataRange(self,(xmin,xmax),(ymin,ymax),fixed=True):
        """
        define minimum and maximum data range, usually this is deduced
        from the given data automatically, however, for sequential 
        gridding it is usefull to set this before the first call of the
        gridder. data outside the range are simply ignored

        Parameters
        ----------
         xmin,ymin:   minimum value of the gridding range in x,y
         xmax,ymax:   maximum value of the gridding range in x,y
         fixed: flag to turn fixed range gridding on (True (default)) 
                or of (False)
        """
        self.fixed_range = fixed
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self,*args):
        """
        Perform gridding on a set of data.

        Parameters
        ----------
        x ............... numpy array of arbitrary shape with x positions
        y ............... numpy array of arbitrary shape with y positions
        data ............ numpy array of arbitrary shape with data values
        """
        
        if not self.keep_data:
            self.Clear()

        x = args[0]
        y = args[1]
        data = args[2]
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        data = data.reshape(data.size)

        if x.size != y.size or y.size!=data.size:
            raise exception.InputError("XU.Gridder2D: size of given datasets (x,y,data) is not equal!")
 
        if not self.fixed_range:
            self.xmin = x.min()
            self.xmax = x.max()
            self.ymin = y.min()
            self.ymax = y.max()
            # require correct aligned memory for input arrays
            lx = check_array(x,numpy.double)
            ly = check_array(y,numpy.double)
            ldata = check_array(data,numpy.double)
        else:
            mask = numpy.logical_and(x<=self.xmax,x>=self.xmin)
            mask = numpy.logical_and(mask,numpy.logical_and(y<=self.ymax,y>=self.ymin))
            # require correct aligned memory for input arrays
            lx = check_array(x[mask],numpy.double)
            ly = check_array(y[mask],numpy.double)
            ldata = check_array(data[mask],numpy.double)
                  
        #remove normalize flag for C-code, normalization is always performed in python
        flags = self.flags^4
        cxrayutilities.gridder2d(lx,ly,ldata,self.nx,self.ny,
                                 self.xmin,self.xmax,
                                 self.ymin,self.ymax,
                                 self.gdata,self.gnorm,flags)            

