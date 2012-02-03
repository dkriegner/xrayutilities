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

#figure operations

class FigOps(object):
    def __init__(self,fig,axis,event_str):
        self.figure = fig
        self.axis = axis
        self.canvas = self.figure.canvas
        self.cid = self.canvas.mpl_connect(event_str,self.callback)

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid)

    def callback(self,event):
        pass

class CollectPoints(FigOps,list):
    """
    CollectPoints(FigOps,list):
    This class collects x and y positions of a map
    """
    def __init__(self,fig,axis,*args,**keyargs):
        FigObs.__init__(self,fig,axis,"button_click_event")
        list.__init__(self,*args,**keyargs)

    def callback(self,event):
        if event.inaxes:
            list.append(self,numpy.array([event.xdata,event.ydata],dtype=numpy.double))

class MeassureTilt(FigOps):
    """
    Measure the tilt of an epilayer from the non-zero qx component
    of a symmetric map.
    """
    def __init__(self,fig,axis):
        FigOps.__init__(self,fig,axis,"button_click_event")

class TiltCorrection(FigOps):
    """
    Correct a peak position according to a certain tilt value.
    """
    def __init__(self,angle,fig,axis):
        FigOps.__init__(self,fig,axis,"button_click_event")

class MeassureAngle(FigOps):
    def __init__(self,fig,axis):
        FigOps.__init__(self,fig,axis,"button_click_event")
        self.points = []

    def callback(self,event):
        if event.inaxes:
            l = len(self.points)
            if l==3:
                pass
            else:
                pass


class XProfile(FigOps):
    pass

class YProfile(FigOps):
    pass
