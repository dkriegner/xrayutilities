#classes and routines for data evaluation
import Numeric
from matplotlib import pylab as pl
import threading


def functionapprox2d(x,y,xi,yi,fij):
    """
    functionapprox2d(x,y,xi,yi,fij):
    Calculates the approximated value of f at the point (x,j)
    by linear interpolation of the function based on the measured
    function fij at points (xi,yi). (xi,yj) define a regular grid.

    required input arguments:
    x ............. the x value where the function should be approximated
    y ............. the y value where the function should be approximated
    xi ............ x-axis grid
    yi ............ y-axis grid
    fij ........... measured function fij

    return value:
    f ............. the approximated value of fij at the point (x,y)
    """
    x0 = xi[0];
    y0 = yi[0];
    dx = xi[1]-xi[0];
    dy = yi[1]-yi[0];

    imax = fij.shape[1]-1;
    jmax = fij.shape[0]-1;

    i = int(Numeric.floor((x-x0)/dx));
    j = int(Numeric.floor((y-y0)/dy));
   
    if i>imax or j>jmax:
        print "point out of measured field - return 0"
        return 0.0

    #calculate the floating point offset from the integer index
    ioff = x - x0 - i*dx;
    joff = y - y0 - j*dy;

    #calculate the parameters for the linear interpolation function
    #and the approximation due to this axis
    if i==imax:
        fapp_x = fij[j,i];
    else:
        kx = (fij[j,i+1]-fij[j,i])/dx;
        dx = fij[j,i];
        fapp_x = kx*ioff+dx;

    if j==jmax:
        fapp_y = fij[j,i];
    else:
        ky = (fij[j+1,i]-fij[j,i])/dy;
        dy = fij[j,i];
        fapp_y = ky*joff+dy;

    #calculate the x an y approximatio

    f = (fapp_x+fapp_y)/2.0;

    return f;

class Profile1D(object):
    """
    A 1D Profile represents the profile data of a 2D map.
    The fundamental requirement for the dataspace is that the
    grid is regular (so one has to grid the data in other cases).
    """

    def __init__(self,p0,p1,**keyargs):
        self.p0 = p0;
        self.p1 = p1;
        self.r01 = self.p1 - self.p0;

        self.r = Numeric.sqrt(Numeric.sum(self.r01**2));
        self.ru = self.r01/self.r;

        if keyargs.has_key("nofp"):
            self.nofp = keyargs["nofp"];
            self.delta = self.r/float(self.nofp);
        else:
            self.nofp = 100;
            self.delta= self.r/float(self.nofp);

        if keyargs.has_key("delta"):
            self.delta = keyargs["delta"];
            self.nofp = int(self.r/self.delta);

    def get_profile_data(self,xi,yi,fij):
        """
        get_profile_data(xi,yi,fij):
        Extracts the data values for the profile.
        
        required input arguments:
        xi ............. x-axis of the regular grid
        yi ............. y-axis of the regular grid
        fij ............ 2D datafield of the measured function
        """

        pxarray = Numeric.zeros((self.nofp),Numeric.Float);
        pyarray = Numeric.zeros((self.nofp),Numeric.Float);
        darray  = Numeric.zeros((self.nofp),Numeric.Float);
        carray  = Numeric.zeros((self.nofp),Numeric.Float);

        for i in range(self.nofp):
            #calculate the new data point
            p = self.p0 + i*self.delta*self.ru;
            c = i*self.delta;
            f = functionapprox2d(p[0],p[1],xi,yi,fij);

            pxarray[i] = p[0]; pyarray[i] = p[1];
            darray[i] = f; carray[i] = c;

        return [pxarray,pyarray,carray,darray];
            
    def get_max(self,xi,yi,fij):
        """
        get_max():
        Determines the maximum points and value of the profile.
        """
        px,py,c,f = self.get_profile_data(xi,yi,fij);
        fmax = max(f.tolist());
        imax = f.tolist().index(fmax);
        cmax = c[imax];
        pxmax = px[imax];
        pymax = py[imax];

        return [pxmax,pymax,cmax,fmax]            


    def plot_profile(self,xi,yi,fij):

        px,py,c,f = self.get_profile_data(xi,yi,fij);
        pl.semilogy(c,f);


class MapProfiler(object):
    """
    An object to select profiles from a 2D map.
    """

    def __init__(self,xi,yi,fij):
        self.xgrid = xi;
        self.ygrid = yi;
        self.fij   = fij;

        self.xaxis = self.xgrid[0,:];
        self.yaxis = self.ygrid[:,0];

        self.FigureHandler = None;
        self.FigureID = None;

        self.ProfileList = [];

        
        #some helper attributes for the class
        self.ProfMode = False; #flag for profiling mode

        self.XPointBuffer = []; #point buffering for x-axis values
        self.YPointBuffer = []; #point buffering for y-axis values
        self.GotFirstPoint = False;

        self.ProfileLineHandler = None;
        self.TmpProfile = None;
        self.ProfileFigureHandler = None;
        self.ProfileFigureID = None;
        self.MoveEventCounter = 0;
        

    def set_data(self,xi,yi,fij):
        #set new data to the MapProfilder
        self.xgrid = xi;
        self.ygrid = yi;
        self.fij   = fij;

        self.xaxis = self.xgrid[0,:];
        self.yaxis = self.ygrid[:,0];

    def run(self):
        self.FigureHandler = pl.figure();
        self.FigureID = self.FigureHandler.number;
        self.ProfileFigureHandler = pl.figure();
        self.ProfileFigureID = self.ProfileFigureHandler.number;

        pl.figure(self.FigureID);
        self.MouseButtonPressHandler = pl.connect("button_press_event",self.mouse_press_event);
        self.KeyEventHandler = pl.connect("key_press_event",self.key_event);
        self.plot_data();
        pl.show();

    def plot_data(self):
        pl.figure(self.FigureID);
        pl.contourf(self.xgrid,self.ygrid,Numeric.log10(self.fij+1.0),50)

    def plot_profile_line(self,px,py):
        pl.figure(self.FigureID);
        ax_back = pl.axis();
        self.ProfileLineHandler, = pl.plot(px,py,"k-D");
        pl.axis(ax_back);
        self.FigureHandler.canvas.draw();

    def mouse_press_event(self,event):
        if event.inaxes and event.button==1 and self.ProfMode:
            if self.XPointBuffer == [] and self.YPointBuffer == [] :
                print "found first point for histogram"
                self.XPointBuffer.append(event.xdata);
                self.YPointBuffer.append(event.ydata);
                self.plot_profile_line([event.xdata,event.xdata],[event.xdata,event.xdata]);
        
            else:
                #create a temporary profile
                px = [self.XPointBuffer[0],event.xdata];
                py = [self.YPointBuffer[0],event.ydata];
                p0 = Numeric.array([px[0],py[0]]);
                p1 = Numeric.array([px[1],py[1]]);
                self.TmpProfile = Profile1D(p0,p1,nofp=100);

                self.plot_profile_line(px,py);

                #plot the temporary profile
                pl.figure(self.ProfileFigureID);
                pl.clf();
                self.TmpProfile.plot_profile(self.xaxis,self.yaxis,self.fij);
                pl.grid(True);
                self.ProfileFigureHandler.canvas.draw();

                #reset buffers
                self.XPointBuffer = []; self.YPointBuffer == [];

        if event.inaxes and event.button == 3 and self.ProfMode:
            print "withdraw the last profile"
            #withdraw the last profile
            self.TmpProfile = None;
            #reset buffers
            self.XPointBuffer = []; self.YPointBuffer == [];

        if event.inaxes and event.button == 2 and self.ProfMode:
            print "add the profile to the profile list"
            #add the profile to the profile list.
            self.ProfileList.append(self.TmpProfile);
            self.TmpProfile = None;
                
    def key_event(self,event):    
        #toogle profiling mode
        if event.key=="p" and event.inaxes:
            if self.ProfMode:
                print "switch to interactive user mode ..."
                self.ProfMode = False;
                #reset the point buffer
                self.XPointBuffer = [];
                self.YPointBuffer = [];
            elif self.ProfMode==False:
                print "switch to profiling mode ..."
                self.ProfMode = True;
        

class DataSelector(object):
    def __init__(self,*data,**options):
        self.xdata = [];
        self.ydata = [];
        self.zdata = [];

        if options.has_key("figid"):
            self.figureID = options["figid"];
        else:
            self.figureID = 1;
            
        self.figure = pl.figure(self.figureID);
        self.AnoList = [];

        pl.figure(self.figureID);
        self.CallbackButtonID = pl.connect("button_press_event",self);
        self.CallbackKeyboardID = pl.connect("key_press_event",self);

        self.FetchMode = False;
        self.WindowMode = False;

        self.WindowList = [];
        self.WindowBuffer = [];
        

        if options.has_key("maxp"):
            self.maxnofp = options["maxp"];            
        else:
            self.maxnofp = 0;

        self.SelectionRunning = True;

        self.PlotOptions = "";
        if options.has_key("popt"):
            self.PlotOptions = options["popt"];

        self.PlotProcedure = "";
        if options.has_key("pproc"):
            self.PlotProcedure = options["pproc"];
        else:
            print "you have to set a plotting routine"

        #evaluate the plotting procedure
        if self.PlotProcedure == "plot":
            #a simple line plot
            pl.plot(data[0],data[1],self.PlotOptions);
        elif self.PlotProcedure == "semilogy":
            #semi logarithmic plot with log10 on y-axis
            pl.semilogy(data[0],data[1],self.PlotOptions);
        elif self.PlotProcedure == "semilogx":
            #semi logarithmic plot with log10 on x-axis
            pl.semilogx(data[0],data[1],self.PlotOptions);
        elif self.PlotProcedure == "pcolor":
            pl.pcolor(data[0],shading="flat");
        elif self.PlotProcedure == "contourf":
            if len(data)==1:
                pl.contourf(data[0]);
            elif len(data)==3:
                pl.contourf(data[0],data[1],data[2]);

        self.axis_range = pl.axis();        
        pl.show();
            

    def print_help(self):
        """
        print the help text on the shell
        """
        help_string = """
        Using the DataSelector class as a callback for matplotlib figures:
        The class behaviour can be controlled with the following
        keyboard commands:

        x ........... finish data selection
        f ........... toggle fetchmode on and off. In fetchmode
                      datapoints are selected at every mouse click.
                      Outside fetch mode the mouse can be used for
                      all kind of image manipulation operations.
        r ........... remove all selected points from the point list
        w ........... toggle window mode: in the window mode the data selector class
                      allows you to define windows in the plotted map.                              
        h ........... print this help text
        """
        

    def __call__(self,event):

        #close the figure
        if event.key == "x":
            pl.close(self.figureID);
            self.SelectionRunning = False;

        #switch between fetchmode on and off
        if event.key == "f":
            if self.FetchMode:
                print "switching mouse no interactive navigation mode ...."
                self.FetchMode = False;
            else:
                print "switching mouse to fetch mode ..."
                self.FetchMode = True;

        #redraw the figure without annotations    
        if event.key == "r":
            print "remove old annotations and flush data buffer";
            self.AnoList = [];
            self.xdata=[];
            self.ydata=[];
            self.WindowBuffer = [];
            self.WindowList = [];
            self.figure.canvas.draw();

        if event.key == "w":
            if self.WindowMode:
                print "switching mouse to interactive navigation mode ..."
                self.WindowMode = False;
                self.WindowBuffer = [];
            else:
                print "switching mouse to window mode ..."
                self.WindowMode = True;

        if event.key == "h":
            self.print_help();
            
        
        if event.inaxes and event.key==None and self.FetchMode:
            #before adding a new data point we have to check if this
            #point would not exceed the maximum number of points limit

            if self.maxnofp!=0 and len(self.xdata)+1>self.maxnofp:
                pl.close(self.figureID);
            
            self.xdata.append(event.xdata);
            self.ydata.append(event.ydata);
            self.draw_cross(event.xdata,event.ydata);

        if event.inaxes and event.key==None and self.WindowMode:
            if self.WindowBuffer == []:
                #no first point has been selected
                print "found first window point"
                self.WindowBuffer.append(event.xdata);
                self.WindowBuffer.append(event.ydata);
            elif len(self.WindowBuffer) == 2:
                print "found the second window point"
                #first point has allready been selected
                self.WindowBuffer.append(event.xdata);
                self.WindowBuffer.append(event.ydata);

                #add the window to the window list
                self.WindowList.append([self.WindowBuffer[0],self.WindowBuffer[1],\
                                        self.WindowBuffer[2],self.WindowBuffer[3]]);
                #reset the Window Buffer
                self.WindowBuffer = [];

                #plot the window frame
                upper_left_corner = [self.WindowList[len(self.WindowList)-1][0],\
                                     self.WindowList[len(self.WindowList)-1][1]];
                lower_left_corner = [self.WindowList[len(self.WindowList)-1][2],\
                                     self.WindowList[len(self.WindowList)-1][3]];
                self.draw_rect(upper_left_corner,lower_left_corner);
                print self.WindowList[len(self.WindowList)-1]
                
            

    def draw_cross(self,xvalue,yvalue):
        pl.figure(self.figureID);
        self.axis_range = pl.axis();
        self.AnoList.append(pl.plot([xvalue],[yvalue],"kD"));
        pl.axis(self.axis_range);
        self.figure.canvas.draw();

    def draw_rect(self,upper_right_corner,lower_left_corner):
        pl.figure(self.figureID);
        self.axis_range = pl.axis();

        #create the x and y coordinate list
        xlist = []; ylist = [];
        xlist.append(upper_right_corner[0]); ylist.append(upper_right_corner[1]);
        xlist.append(lower_left_corner[0]); ylist.append(upper_right_corner[1]);
        xlist.append(lower_left_corner[0]); ylist.append(lower_left_corner[1]);
        xlist.append(upper_right_corner[0]); ylist.append(lower_left_corner[1]);
        xlist.append(upper_right_corner[0]); ylist.append(upper_right_corner[1]);
        pl.plot(xlist,ylist,"k-");
        
        pl.axis(self.axis_range);
        self.figure.canvas.draw();
    

#selects from a given point the 
def getangdelta(ome,tth,psd,omeref,tthref):

    pl.figure(1);
    pl.contourf(ome,tth,Numeric.log10(psd+1),40);
    df = DataSelector(1);

    pl.show();

    ome_refp = df.xdata[0];
    tth_refp = df.ydata[0];

    dome = ome_refp - omeref;
    dtth = tth_refp - tthref;

    return(dome,dtth);
