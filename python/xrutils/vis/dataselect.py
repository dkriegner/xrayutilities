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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>

"""
this module contains functions to select data from
arrays with axes.

General remarks abouth selecting data:
  the selection functions usually take three arguments:


 General functions
   Profile1D_3D
   IntProfile1D_3D
   Profile1D_2D
   IntProfile1D_2D
   Plane
   InPlane

 Special functions
   YProfile1D_3D - select a profile along y-direction from a 3D
                   dataset
   ZProfile1D_3D - select a profile along z-direction from a 3D
                   dataset
   XProfile1D_3D - select a profile along x-direction from a 3D
                   dataset
   XYPlane - select the XY Plane from a 3D dataset
   YZPlane - select the YZ plane from a 3D dataset
   XZPlane - select the XZ plane from a 3D dataset
"""

import numpy

def RSM1DInterpOn2D(data,qx,qz,qxi,qzi,data_aligned=True):
    """

    """
    #{{{
    try:
        n = qxi.shape[0]
    except:
        pass

    try:
        n = qzi.shape[0]
    except:
        pass

    if not isinstance(qxi,numpy.ndarray):
        qxi = qxi*numpy.ones((n),dtype=numpy.float)

    if not isinstance(qzi,numpy.ndarray):
        qzi = qzi*numpy.ones((n),dtype=numpy.float)


    odata = numpy.zeros((n),dtype=numpy.float)

    #shape of the input data
    nxd = qx.shape[0]
    nzd = qz.shape[0]

    for i in range(n):
        #find the cell used for the data
        x = qxi[i]
        z = qzi[i]

        #if the point is outside the domain continue with
        #loop
        if x < qx.min() or x>qx.max(): continue
        if z < qz.min() or z>qz.max(): continue

        x_index = 0
        z_index = 0
        for j in range(nxd-1):
            if x >= qx[j] and x<qx[j+1]:
                x_index = j
                break

        for j in range(nzd-1):
            if z >= qz[j] and z<qz[j+1]:
                z_index = j
                break

        #calculate the step width in the input data
        dx = abs(qx[x_index+1]-qx[x_index])
        dz = abs(qz[z_index+1]-qz[z_index])

        r = (x-qx[x_index])/dx
        s = (z-qz[z_index])/dz

        w0 = (1-r)*(1-s)
        w1 = r*(1-s)
        w2 = s*(1-r)
        w3 = r*s

        if data_aligned:
            d0 = data[z_index,x_index]
            d1 = data[z_index,x_index+1]
            d2 = data[z_index+1,x_index]
            d3 = data[z_index+1,x_index+1]
        else:
            d0 = data[x_index,z_index]
            d1 = data[x_index+1,z_index]
            d2 = data[x_index,z_index+1]
            d3 = data[x_index+1,z_index+1]


        odata[i] = w0*d0+w1*d1+w2*d2+w3*d3

    return odata





    #}}}

def AlignInt(data,frac=0.5,ind=1):
    #{{{
    """
    AlignInt(data):
    This function sets the minimum of a data array to a fraction of the
    second smallest value.
    The idea behind this function is to avoid plotting problems if
    many data points are 0, which causes problems in the case of log10 plots.

    required input arguments:
    data ................. a numpy data array

    optional keyword arguments:
    ind .................. which of the smallest values to choose for alignment
    frac ................. the fraction
    """

    l = (data.reshape(data.size)).tolist()
    l.sort()
    min = l[0]
    for x in l:
        if x>min:
            break

    del l
    data[data<x] = frac*x

    return data
    #}}}

def AlignDynRange(data,order,value = 0):
    """
    AlignDynRange(data,value = 0):
    Align the dynamic range of data to a certain order of magnitudes.

    """
    #{{{

    imax = data.max()
    imin = imax/10**order
    data[data<=imin] = value

    return data
    #}}}

def Align2DData(data):
    #{{{
    """
    Align2DData(data):
    Aligns data for plotting. Due to the storage scheme for data used
    in this xrutils 2D data cannot be plotted in the way users are used to.
    This function performs flip and rotation operations on the data matrix
    to align them for plotting.

    required input arguments:
    data ................ 2D numpy array with data

    return value:
    flipped and rotated 2D numpy array
    """
    return numpy.flipud(numpy.rot90(data))
    #}}}

def GetAxisBounds(axis,x):
    #{{{
    """
    GetAxisBounds(axis,x):
    Return the upper and lower axis node value surounding an
    arbitrary value x. The axis is assumed to be stored in
    asccending order. If x is outside the axis bounds the
    function returns None.

    input arguments:
    axis ............ numpy array or list with the axis values
    x ............... an arbitrary value on the axis

    return value:
    [lb,ub,lb_index,ub_index]
    lb .............. lower node value
    ub .............. upper node value
    lb_index ........ index of the lower bound
    ub_index ........ index of the upper bound
    """

    #return None if x is outside the axis bounds
    if x<axis.min() or x>axis.max():
        return None

    for i in range(axis.shape[0]):
        if axis[i]>=x: break

    return [axis[i-1],axis[i],i-1,i]
    #}}}

#------------------low level 1D Profile functions-----------------
def Profile1D_3D(data,dim,axis,pos):
    #{{{1
    """
    Profile1D_3D(data,dim,axis,pos):
    Select a 1D profile from a 3D data set. The profile is taken along dimension
    dim at position pos on axis. axis is the axis along the direction
    perpendicular to the profile.

    required input arguments:
    data ................ 3D numpy array
    dim ................. [ximd,ydim] the tow dimensions perpendicular
                          to the profile
    axis ................ [x,y] axes along the perpendicular directions
    pos ................. [px,py] positions along the perpendicular
                          directions

    return value:
    1D numpy array with the data
    """

    [lb_x,ub_x,lbi_x,ubi_x] = GetAxisBounds(axis[0],pos[0])
    [lb_y,ub_y,lbi_y,ubi_y] = GetAxisBounds(axis[1],pos[1])


    if dim == 0:
        d1 = data[:,lbi_x,lbi_y]
        d2 = data[:,lbi_x,ubi_y]
        d3 = data[:,ubi_x,lbi_y]
        d4 = data[:,ubi_x,ubi_y]
    elif dim == 1:
        d1 = data[lbi_x,:,lbi_y]
        d2 = data[lbi_x,:,ubi_y]
        d3 = data[ubi_x,:,lbi_y]
        d4 = data[ubi_x,:,ubi_y]
    elif dim == 2:
        d1 = data[lbi_x,lbi_y,:]
        d2 = data[lbi_x,ubi_y,:]
        d3 = data[ubi_x,lbi_y,:]
        d4 = data[ubi_x,ubi_y,:]
    else:
        print "invalid dimension!"
        return None

    #perform linear interpolation
    d = (d1+d2+d4+d4)/4.

    return d
    #}}}1

def Profile1D_2D(data,dim,axis,pos):
    #{{{1
    """
    Prof2D(data,dim,axis,pos):
    Extracts a 1D profile from a 2D dataset. dim is the dimension
    along which the profile should be taken. axis and pos are the
    axis perpendicular to the profile direction and the position on this
    axis where the profile should be taken respectively.

    required input arguments:
    data .............. 2D data array
    dim ............... dimension parallel to the profile direction
    axis .............. axis perpendicular to the profile direction
    pos ............... position at axis where the profile should be taken

    return value:
    1D data array with the profile data
    """

    [lb,ub,lb_index,ub_index] = GetAxisBounds(axis,pos)

    if dim == 0:
        d1 = data[:,lb_index]
        d2 = data[:,ub_index]
    elif dim == 1:
        d1 = data[lb_index,:]
        d2 = data[ub_index,:]
    else:
        print "invalid dimension!"
        return None

    #perform linear interpolation
    d = (d2-d1)*pos/(ub-lb)+(d1*ub-d2*lb)/(ub-lb)

    return d
    #}}}1

def IntProfile1D_3D(data,dim,axis,pos1,pos2):
    #{{{1
    """
    IntProfile1D_3D(data,dim,axis,pos1,pos2):
    Select an 1D integral profile along dimension dim. axis holds a list with
    the two axes perpendicular to the profile direction. pos1 is a list with the
    two lower integration bounds and pos2 a list with the upper bounds.

    required input arguments:
    data ................ 3D numpy array
    dim ................. [ximd,ydim] the tow dimensions perpendicular
                          to the profile
    axis ................ [x,y] axes along the perpendicular directions
    pos1 ................ [px,py] positions along the perpendicular
                          directions - lower integration bound
    pos2 ................ [px,py] positions along the perpendicular
                          directions - upper integration bound

    return value:
    1D numpy array with the data
    """

    [t1,t2,lbi_x,t3] = GetAxisBounds(axis[0],pos1[0])
    [t1,t2,lbi_y,t3] = GetAxisBounds(axis[1],pos1[1])
    [t1,t2,t3,ubi_x] = GetAxisBounds(axis[0],pos1[0])
    [t1,t2,t3,ubi_y] = GetAxisBounds(axis[1],pos1[1])


    if dim == 0:
        d = data[:,lbi_x:ubi_x,lbi_y:ubi_y].sum(axis=2)
        d = d.sum(axis=1)
    elif dim == 1:
        d = data[lbi_x:ubi_x,:,lbi_y:ubi_y].sum(axis=2)
        d = d.sum(axis=0)
    elif dim == 2:
        d = data[lbi_x:ubi_x,lbi_y:ubi_y,:].sum(axis=1)
        d = d.sum(axis=0)
    else:
        print "invalid dimension!"
        return None

    return d
    #}}}1

def IntProfile1D_2D(data,dim,axis,pos1,pos2):
    #{{{1
    """
    IntProfile1D_2D(data,dim,axis,pos1,pos2):
    Selectes a 1D integral profile along dimension dim. axis holds the axes
    perpendicular to the profile dimension. pos1 is the lower integration bound
    on axis and pos2 the upper bound.

    required input arguments:
    data ............. 2D data array
    dim .............. dimension along which the profile should be taken
    axis ............. axis perpendicular to the profile direction
    pos1 ............. starting position for the summation
    pos2 ............. stop position for the summation

    return value:
    1D data array with the profile data
    """
    [lb,ub,s_index,ubi] = GetAxisBounds(axis,pos1)
    [lb,ub,lbi,e_index] = GetAxisBounds(axis,pos2)

    if dim == 0:
        d = data[:,s_index:e_index].sum(axis=1)
    elif dim == 1:
        d = data[s_index:e_index,:].sum(axis=0)
    else:
        print "invalid dimensino!"
        return None

    return d
    #}}}1

#------------------high level 1D Profile functions----------------

def YProfile1D_3D(data,x,z,*pos,**keyargs):
    #{{{1
    """
    YProfile1D_3D(data,x,z,*pos,sum=false):
    Select a profile along y-direction (dim=1) from a 3D dataset.
    Several profiles can be extracted at the same time by providing
    several points in *pos. If the keyword argument sum is set to true
    the number of positions must be even and always two positions are
    taken as upper and lower integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    x .................. numpy array with x-axis values
    z .................. numpy array with z-axis values
    *pos ............... list of points

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    1D numpy array with the profile data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntProfile1D_3D(data,1,[x,z],p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Profile1D_3D(data,1,[x,z],p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def ZProfile1D_3D(data,x,y,*pos,**keyargs):
    #{{{1
    """
    ZProfile1D_3D(data,x,y,*pos,sum=false):
    Select a profile along z-direction (dim=2) from a 3D dataset.
    Several profiles can be extracted at the same time by providing
    several points in *pos. If the keyword argument sum is set to true
    the number of positions must be even and always two positions are
    taken as upper and lower integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    x .................. numpy array with x-axis values
    y .................. numpy array with y-axis values
    *pos ............... list of points

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    1D numpy array with the profile data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntProfile1D_3D(data,2,[x,y],p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Profile1D_3D(data,2,[x,y],p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def XProfile1D_3D(data,y,z,*pos,**keyargs):
    #{{{1
    """
    XProfile1D_3D(data,y,z,*pos,sum=false):
    Select a profile along x-direction (dim=0) from a 3D dataset.
    Several profiles can be extracted at the same time by providing
    several points in *pos. If the keyword argument sum is set to true
    the number of positions must be even and always two positions are
    taken as upper and lower integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    y .................. numpy array with y-axis values
    z .................. numpy array with z-axis values
    *pos ............... list of points

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    1D numpy array with the profile data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntProfile1D_3D(data,0,[y,z],p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Profile1D_3D(data,0,[y,z],p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def XProfile1D_2D(data,y,*pos,**keyargs):
    #{{{1
    """
    XProfile1D_2D(data,y,*pos,sum=false):
    Select a profile along x-direction (dim=0) from a 2D dataset.
    Several profiles can be extracted at the same time by providing
    several points in *pos. If the keyword argument sum is set to true
    the number of positions must be even and always two positions are
    taken as upper and lower integration bound respecively.

    mandatory input arguments:
    data ............... a 2D numpy array with the data
    y .................. numpy array with y-axis values
    *pos ............... list of points

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    1D numpy array with the profile data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntProfile1D_2D(data,0,y,p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Profile1D_2D(data,0,y,p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def YProfile1D_2D(data,x,*pos,**keyargs):
    #{{{1
    """
    YProfile1D_2D(data,x,*pos,sum=false):
    Select a profile along y-direction (dim=1) from a 2D dataset.
    Several profiles can be extracted at the same time by providing
    several points in *pos. If the keyword argument sum is set to true
    the number of positions must be even and always two positions are
    taken as upper and lower integration bound respecively.

    mandatory input arguments:
    data ............... a 2D numpy array with the data
    y .................. numpy array with y-axis values
    *pos ............... list of points

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    1D numpy array with the profile data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntProfile1D_2D(data,1,x,p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Profile1D_2D(data,1,x,p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def Plane(data,dim,axis,pos):
    #{{{1
    """
    Plane(data,dim,axis,pos):
    Select a certain plane from a 3D data block perpendicular to a
    certain dimension "dim" at a given position "pos" on an axis
    "axis". In most cases pos will not reside on a axis node but rather
    somewhere in between. Therefore, linear interpolation will be used
    between the tow surounding data arrays to obtain the data plane.

    required input arugments:
    data ............ 3D data array
    dim ............. dimension perpendicular to the plane
    axis ............ axis along the perpendicular direction
    pas ............. position along the perpendicular axis

    return value:
    a 2D data array
    """

    [lb,ub,lb_index,ub_index] = GetAxisBounds(axis,pos)

    if dim==0:
        d1 = data[lb_index,:,:]
        d2 = data[ub_index,:,:]
    elif dim==1:
        d1 = data[:,lb_index,:]
        d2 = data[:,ub_index,:]
    elif dim == 2:
        d1 = data[:,:,lb_index]
        d2 = data[:,:,ub_index]
    else:
        print "invalid cut dimension!"
        return None

    #perform linear interpolation
    d = (d2-d1)*pos/(ub-lb)+(d1*ub-d2*lb)/(ub-lb)

    return d
    #}}}1

def IntPlane(data,dim,axis,pos1,pos2):
    #{{{1
    """
    DataSum(data,dim,axis,pos1,pos2):
    Summ the data array along a certain dimension dim between
    position pos1 and pos2 along axis.

    required input argumens:
    data .............. 3D data array
    dim ............... dimension along which the summation should take place
    axis .............. axis along the summation direction
    pos1 .............. starting position for the summation
    pos2 .............. end position for the summation

    return value:
    2D array with the summation data
    """

    [lb,ub,s_index,ubi] = GetAxisBounds(axis,pos1)
    [lb,ub,lbi,e_index] = GetAxisBounds(axis,pos2)

    if dim==0:
        d = data[s_index:e_index,:,:].sum(axis=0)
    elif dim==1:
        d = data[:,s_index:e_index,:].sum(axis=1)
    elif dim==2:
        d = data[:,:,s_index:e_index].sum(axis=2)
    else:
        print "invalid sumation dimension!"
        return None

    return d
    #}}}1


def XYPlane(data,z,*pos,**keyargs):
    #{{{1
    """
    XYPlane(data,z,*pos,sum=false):
    Select data on the x-y plane at position pos. Multiple planes can
    be extracted by passing several z-positions to the function.
    If the keyword argument sum is set to true the number of positions
    must be even and always two positions are taken as upper and lower
    integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    z .................. numpy array with z-axis values
    *pos ............... list of points along the z-axis

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    2D numpy array with the plane data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntPlane(data,2,z,p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Plane(data,2,z,p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def YZPlane(data,x,*pos,**keyargs):
    #{{{1
    """
    YZPlane(data,x,*pos,sum=false):
    Select data on the y-z plane at x position pos. Multiple planes can
    be extracted by passing several z-positions to the function.
    If the keyword argument sum is set to true the number of positions
    must be even and always two positions are taken as upper and lower
    integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    x .................. numpy array with x-axis values
    *pos ............... list of points along the x-axis

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    2D numpy array with the plane data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntPlane(data,0,x,p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Plane(data,0,x,p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1

def XZPlane(data,y,*pos,**keyargs):
    #{{{1
    """
    XZPlane(data,y,*pos,sum=false):
    Select data on the x-z plane at y position pos. Multiple planes can
    be extracted by passing several y-positions to the function.
    If the keyword argument sum is set to true the number of positions
    must be even and always two positions are taken as upper and lower
    integration bound respecively.

    mandatory input arguments:
    data ............... a 3D numpy array with the data
    y .................. numpy array with y-axis values
    *pos ............... list of points along the y-axis

    optional keyword arguments:
    sum ................ true/flase select integration

    return value:
    2D numpy array with the plane data
    """
    olist = []
    if keyargs.has_key("sum"):
        sum = keyargs["sum"]
    else:
        sum = flase

    if sum:
        #perform integration
        if len(pos)%2!=0:
            print "number of positions must be even for integration"
            return None

        for i in range(0,len(pos),2):
            p1 = pos[i]
            p2 = pos[i+1]
            d  = IntPlane(data,1,y,p1,p2)
            olist.append(d)
    else:
        for p in pos:
            d = Plane(data,1,y,p)
            olist.append(d)

    if len(pos)==1:
        return olist[0]
    else:
        return olist
    #}}}1
