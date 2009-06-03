
from matplotlib.widgets import Cursor
from matplotlib.pyplot import figure
import pygtk
import gtk
from matplotlib import pyplot as xplt
import matplotlib.widgets as widgets
import numpy

class XFigure(object):
    def __init__(self,fig):
        self.figure = fig
        self.canvas = self.figure.canvas
        self.toolbar = self.canvas.toolbar
        self.id1 = self.canvas.mpl_connect("motion_notify_event",self.show_pos)
        #add some additional buttons to the toolbar
        self.button = self.toolbar.append_element(gtk.TOOLBAR_CHILD_BUTTON,None,"test","testbutton",
                                                  "Private",None,None,None)

        self.axis   = self.figure.add_subplot(111)
        self.cursor  = Cursor(self.axis,useblit=True,c="k")
        self.canvas.draw()

        self.data = None
        self.data_min = 0.
        self.data_max = 0.

    def show_pos(self,event):
        if event.inaxes:
            f = event.inaxes.get_figure()
            f.canvas.toolbar.set_message("x=%e y=%e" %(event.xdata,event.ydata))

    def toggle_grid(self):
        self.axis.grid()
        self.canvas.draw()

    def draw(self):
        self.canvas.draw()

    def setdata(self,data):
        pass

    def setdatarange(self,data):
        pass


    

def xfigure(*args,**keyargs):
    f = figure(*args,**keyargs)
    xf = XFigure(f)


    return xf 

class MeasureDistance(object):
    def __init__(self,figure):
        self.figure = figure
        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)
        self.p1 = None
        self.p2 = None
        self.d  = None
        self.d_list = []

    def __button_press_event__(self,event):
        if event.inaxes:
            if self.p1 == None:
                self.p1 = numpy.array([event.xdata,event.ydata])
            else:
                self.p2 = numpy.array([event.xdata,event.ydata])
                self.d = self.p2-self.p1

                print self.d
                self.d_list.append(self.d)
                self.p1 = None
                self.p2 = None

    def Average(self):
        a = numpy.zeros((2),dtype=numpy.double)

        for d in self.d_list:
            a += d

        a = a/len(self.d_list)
        print "average distance: %e %e" %(a[0],a[1])
        return a

    def Clear(self):
        self.d_list = []

    def Disconnect(self):
        self.figure.canvas.mpl_disconnect(self.mid)

    def SetFigure(self,figure):
        self.figure = figure
        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)

class MeasurePeriod(object):
    def __init__(self,figure):
        self.figure = figure
        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)
        self.p1 = None
        self.p2 = None
        self.d  = None
        self.p  = None
        self.p_list = []

    def __button_press_event__(self,event):
        if event.inaxes:
            if self.p1 == None:
                self.p1 = numpy.array([event.xdata,event.ydata])
            else:
                self.p2 = numpy.array([event.xdata,event.ydata])
                self.d = self.p2-self.p1
                self.p = 2.*numpy.pi/self.d

                print self.p
                self.p_list.append(self.p)
                self.p1 = None
                self.p2 = None

    def Average(self):
        a = numpy.zeros((2),dtype=numpy.double)
        for p in self.p_list:
            a += p

        a = a/len(self.p_list)
        print "average period: %e %e" %(a[0],a[1])

        return a

    def Clear(self):
        self.p_list = []

    def Disconnect(self):
        self.figure.canvas.mpl_disconnect(self.mid)

    def SetFigure(self,figure):
        self.figure = figure
        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)



class PointSelect(object):
    def __init__(self,figure):
        self.figure = figure
        self.xpos = 0
        self.ypos = 0

        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)
        self.mid2 = self.figure.canvas.mpl_connect("motion_notify_event",self.__move_event_handler__)
    
    def __move_event_handler__(self,event):
        if event.inaxes:
            #print cursor position to the canvas
            self.figure.canvas.toolbar.set_message("%e %e" %(event.xdata,event.ydata))

    def __button_press_event__(self,event):
        if event.inaxes:
            self.figure.canvas.mpl_disconnect(self.mid)
            self.figure.canvas.mpl_disconnect(self.mid2)
            self.xpos = event.xdata
            self.ypos = event.ydata

    def SetFigure(self,figure):
        self.figure = figure
        self.xpos = 0
        self.ypos = 0

        self.mid = self.figure.canvas.mpl_connect("button_press_event",self.__button_press_event__)
        self.mid2 = self.figure.canvas.mpl_connect("motion_notify_event",self.__move_event_handler__)

    def __str__(self):
        return "%e %e\n" %(self.xpos,self.ypos)


class DataPicker(object):
    def __init__(self,array,**keyargs):
        """
        DataPicker allows picking data from an array.

        required input arguments:
        array ................. array of shape nxm for plotting

        optional keyword arguments:
        x ..................... array of shape m or nxm with x-axis data
        y ..................... array fo shape n or nxm with y-axis data
        pstyle ................ style how to plot the data, can take the 
                                following value:
                                image -> using imshow x and y must be of shape n
                                         and m respectively
                                cont -> contour plot x and y must be both of
                                shape nxm
                                pcol -> pcolor plot x and y as for image.
        xl .................... string with the label for the x-axis
        yl .................... string with the label for the y-axis
        the default plotting style is pcolor

        keybord commands:
        d ...................... clear the point list
        p ...................... toogle picking on of
        c ...................... activate crosshair cursor
        """
        self.figure = xplt.figure()
        self.canvas = self.figure.canvas
        self.axis = self.figure.add_subplot(111)
        self.plist = []

        self.pick_mid = self.figure.canvas.mpl_connect("button_press_event",
                                                       self.__pick_event_handler__)
        self.move_mid = self.figure.canvas.mpl_connect("motion_notify_event",
                                                       self.__move_event_handler__)
        self.key_mid  = self.figure.canvas.mpl_connect("key_press_event",
                                                       self.__key_event_handler__)

        #the positions on the canvas
        b = self.figure.subplotpars.bottom 
        self.figure.subplots_adjust(bottom=b+0.025)
        
        #plot the data
        self.data = array
        self.ncont = 50   #set the default number of contours
        if keyargs.has_key("x"):
            self.x_data = keyargs["x"]
        else:
            self.x_data = numpy.arange(0,array.shape[1])

        if keyargs.has_key("y"):
            self.y_data = keyargs["y"]
        else:
            self.y_data = numpy.arange(0,array.shape[0])

        if keyargs.has_key("pstyle"):
            self.plot_style = keyargs["pstyle"]
        else:
            self.plot_style = "pcolor"

        self.plotdata()
        
        #the list where the annotations are stored that show selected points
        self.anot_sel_list = []
        self.text_fmt = "%e %e"
        self.cursor = None

        self.PickFlag = True


    def plotdata(self):
        #clear the axis from all content 
        self.axis.clear()
        if self.plot_style == "pcolor":
            self.plot = self.axis.pcolor(self.x_data,self.y_data,self.data)
        elif self.plot_style == "cont":
            self.plot = elf.axis.contourf(self.x_data,self.y_data,self.data,self.ncont)
        elif self.plot_style == "image":
            self.plot = self.axis.imshow(self.data,interpolation="nearest")

        self.figure.colorbar(self.plot)
        self.canvas.draw()
        

    def __getitem__(self,index):
        return self.plist[index]

    def __pick_event_handler__(self,event):
        if event.inaxes == self.axis:
            xlim = self.axis.get_xlim()
            ylim = self.axis.get_ylim()
            print xlim
            print ylim
            if event.xdata>=xlim[0] and event.xdata<=xlim[1] and\
               event.ydata>=ylim[0] and event.ydata<=ylim[1] and self.PickFlag:
                print "point: (%e, %e)" %(event.xdata,event.ydata)
                self.plist.append((event.xdata,event.ydata))
                self.anot_sel_list.append(self.axis.plot([event.xdata],[event.ydata],"ko")[0])
                self.canvas.draw()
                self.axis.set_xlim(xlim[0],xlim[1])
                self.axis.set_ylim(ylim[0],ylim[1])
                self.canvas.draw()

    def __move_event_handler__(self,event):
        if event.inaxes:
            #print cursor position to the canvas
            self.figure.canvas.toolbar.set_message(self.text_fmt %(event.xdata,event.ydata))

    def __key_event_handler__(self,event):
        if event.key=="d":
            self.clear()
        elif event.key=="c":
            self.crosshair()
        elif event.key == "g":
            self.grid()
        elif event.key=="p":
            if self.PickFlag:
                self.PickFlag = False
                print "set data selection inactive"
                return None
            else:
                self.PickFlag = True
                print "activate data selection"
                return None

    def __del__(self):
        self.figure.canvas.mpl_disconnect(self.pick_mid)
        self.figure.canvas.mpl_disconnect(self.move_mid)
        self.figure.canvas.mpl_disconnect(self.key_mid)
        xplt.close(self.figure)

    def __str__(self):
        ostr = ""
        cnt = 0
        for p in self.plist:
            ostr += "point %i: (%e %e)\n" %(cnt,p[0],p[1])
            cnt += 1

        return ostr

    def clear(self):
        self.plist = []
        for o in self.anot_sel_list:
            self.axis.lines.remove(o)

        self.anot_sel_list = []

        self.canvas.draw()


    def crosshair(self,*arg):
        if self.cursor == None:
            if len(arg)!=0:
                color = arg[0]
            else:
                color = "white"

            self.cursor = widgets.Cursor(self.axis,useblit=True,
                          color=color,linewidth=1)
        else:
            self.cursor = None

        self.canvas.draw()
            

    def grid(self):
        self.axis.grid()
        self.canvas.draw()

        


