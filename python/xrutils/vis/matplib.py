
from matplotlib.widgets import Cursor
from matplotlib.pyplot import figure
import pygtk
import gtk
from matplotlib import pyplot as xplt
import matplotlib.widgets as widgets
import numpy

def align_int_range(data,frac=0.5,ind=1):
    """
    align_int_range(data):
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
        the default plotting style is pcolor
        """
        self.figure = xplt.figure()
        self.canvas = self.figure.canvas
        self.axis = self.figure.add_subplot(111)
        self.plist = []

        self.pick_mid = self.figure.canvas.mpl_connect("button_press_event",
                                                       self.__pick_event_handler__)
        self.move_mid = self.figure.canvas.mpl_connect("motion_notify_event",
                                                       self.__move_event_handler__)

        #have to ajust the main window to plot 
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

        #plot text on the bottom 
        self.text_fmt = "x=%e, y=%e"
        self.text_x = 0.25
        self.text_y = 0.005
        self.text = self.figure.text(self.text_x,self.text_y,self.text_fmt %(0,0))
        self.text.set_animated(True)

        #here comes the tricky part
        self.figure.draw_artist(self.text)
        self.canvas.blit(self.figure.bbox)
        self.cursor = None

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
        #after plotting everything we have to save a background image
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        

    def __getitem__(self,index):
        return self.plist[index]

    def __pick_event_handler__(self,event):
        if event.inaxes:
            print "point: (%e, %e)" %(event.x,event.y)
            self.plist.append((event.xdata,event.ydata))
            self.anot_sel_list.append(self.axis.plot([event.xdata],[event.ydata],"ko"))
            self.axis.draw()

    def __move_event_handler__(self,event):
        if event.inaxes:
            #print cursor position to the canvas
            self.text.set_text(self.text_fmt %(event.xdata,event.ydata))
            #restore the background image
            self.canvas.restore_region(self.background)
            #draw the text artist
            self.figure.draw_artist(self.text)
            #redraw only the changed area
            self.canvas.blit(self.figure.bbox)


    def __del__(self):
        self.figure.canvas.mpl_disconnect(self.pick_mid)
        self.figure.canvas.mpl_disconnect(self.move_mid)
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


    def crosshair(self,*arg):
        if self.cursor == None:
            if len(arg)!=0:
                color = arg[0]
            else:
                color = "white"

            self.cursor = widgets.Cursor(self.axis,useblit=True,
                          color=color,linewidth=2)
        else:
            self.cursor = None
            

    def grid(self):
        self.axis.grid()
        self.text.set_text("")
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

        


