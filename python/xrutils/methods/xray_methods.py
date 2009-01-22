#module contains functions specific for x-ray data evaluation

import numpy
import misc
import detector
import gridder
from matplotlib import pylab as pl

def get_log_levels(levels):
    
    min_level = levels[0];
    max_level = levels[-1];
    if min_level<=0.0:
        min_order = 0;
    else:
        min_order = int(numpy.log10(min_level));
        
    max_order = int(numpy.log10(max_level));
    clevels = [];
    
    for i in range(max_order-min_order):
        comp_value = 10.0**(min_order+i);
        for x in levels:
            if x>comp_value:
                clevels.append(x);
                break;
                
    levelarray = numpy.array(levels).astype(numpy.Int);
    print levelarray
    
    return clevels;

class xraymethod:
    """
    The class xray method is an abstract class to define a x-ray method
    in general it acts as a superclass for the gid,xrr,gisaxs and hxrd
    classes that provide specific methods for handling data of the
    distinct method
    """    
    energy = 0.0;      #energy of the incidenting beam in eV
    detector = None;   #the detector used for the measurement
    #standard resolution for maps    
    mapplot_xlabel = 'x_axis';
    mapplot_ylabel = 'y_axis';
    mapplot_title  = 'A Reciprocal space map';

    def __init__(self,det,keyarg):
        """
        Initialise the basic x-ray method class.
        """
        self.detector = det;
        
        if keyarg.has_key('en'):
            self.energy = keyarg['en'];
            self.wavelength = misc.en2lam(self.energy);                    
        elif keyarg.has_key('wl'):
            self.wavelength = keyarg['wl'];
            self.energy = misc.lam2en(self.wavelength);
        else:
            print 'you have to provide either a wavelength or an energy as keyword argument'

        #setting the default resolution for maps
        if keyarg.has_key('nx'):
            self.nx = keyarg['nx'];
        else:
            self.nx = 200;

        if keyarg.has_key('ny'):
            self.ny = keyarg['ny'];
        else:
            self.ny = 200;


    def maplog(self,mat,low,high):
        """
        maplog(mat,low,high):
        This is a python implementation of Julian Stangls famous
        maplog script originally written in Matlab. The lower and higher 
        threshold are given as scales from the maximum intensity.
        Meaning that if the lower bound is 0.1 then all values 
        smaller then 0.1*i_max are set to this value.

        Input paramters:
        mat ......... the matrix where maplog should be applied to
        low ......... lower threshold
        high ........ upper threshold

        Return value:
        A matrix of the same shape as the input matrix where a logarithm
        of base 10 has been applied to. And where values lower then low are
        set to low and higher than high are set to high.
        
        """
        if low == 0.0 and high == 0.0:
            return mat;

        return numpy.log10(mat+1).clip(min=low,max=high);

    def mapgrid(self,xmat,ymat,zmat,**opts):
        """
        mapgrid(xmat,ymat,zmat,**opts):
        Grids a scattered intensity matrix to a regular one.
        Required input paramters:
        xmat ............. matrix with x values
        ymat ............. matrix with y values
        zmat ............. matrix with intensity (z) values
        optional arguments
        nx ............... number of points in x-direction
        ny ............... number of points in y-direction

        Return values:
        [xmat_new,ymat_new,zmat_new] ....A list with the reduced matrices
                                         for x, y and intensity values.        
        """

        #check for optional arguments
        if opts.has_key('nx'):
            xres = opts['nx'];
        else:
            xres = self.nx;

        if opts.has_key('ny'):
            yres = opts['ny'];
        else:
            yres = self.ny;

        g = gridder.Gridder2D(xres,yres)

        #run the external fortran program for gridding        
        #[zmat_new,xmat_new,ymat_new] = gridder.grid2dmap(xmat,ymat,zmat,xres,yres,axmat=True);
        g.GridData(xmat,ymat,zmat)
        gdmat = numpy.rot90(g.gdata,-1)
        gxmat = numpy.rot90(g.GetXMatrix(),-1)
        gymat = numpy.rot90(g.GetYMatrix(),-1)
        gdmat = g.gdata
        gxmat = g.GetXMatrix()
        gymat = g.GetYMatrix()
        
        return [gxmat,gymat,gdmat];
        
    def volgrid(self,xmat,ymat,zmat,datmat,**keyargs):
        """
        volgrid(xmat,ymat,zmat,datmat,**keyargs):
        The volgrid methods grids a volume data set. It returns the 
        data as a nxmxl array. 
        
        required input aruments:
              xmat ................. array with the x-values
              ymat ................. array with the y-values
              zmat ................. array with the z-values
              datmat ............... array with the data values
              
        optional keyword arguments:
            nx ................... number of points in x-direction
            ny ................... number of points in y-direction
            nz ................... number of points in z-direction
        """
        pass

    def mapplot(self,qx,qz,iq,**opts):
        """
        mapplot(qx,qz,iq):
        Use this function for easily producing nice plots of reciprocal space maps.
        Required input arguments:
        qx ..... matrix with inplane q values
        qz ..... matrix with qz values
        iq ..... matrix with intensity
        optional arguments:
        cc ..... number of color contours
        c ...... number of black contour lines
        title .. title of the plot
        cutof .. [lc,hc] lower and higher cutof
        
        The return value is a handle of the figure object.
        """

        #check for optional paramters
        if opts.has_key('cc'):
            colconts = opts['cc'];
        else:
            colconts = 40;

        if opts.has_key('c'):
            bwconts = opts['c'];
        else:
            bwconts = 0;

        if opts.has_key('title'):
            plot_title = opts['title'];
        else:
            plot_title = self.mapplot_title;

        if opts.has_key('cutof'):
            cf = opts['cutof'];
        else:
            cf = [min(numpy.ravel(iq)),max(numpy.ravel(iq))];            

        fig = pl.figure(figsize=(4.75,4));
        pl.axes([0.15,0.15,0.74,0.75]);

        if isinstance(qx,list):
            pl.hold(True);

            #determine the total minimum and maximum
            qx_min_tot = min(numpy.ravel(qx[0]));
            qx_max_tot = max(numpy.ravel(qx[0]));
            qz_min_tot = min(numpy.ravel(qz[0]));
            qz_max_tot = max(numpy.ravel(qz[0]));
            for i in range(1,len(qx)):
                if (min(numpy.ravel(qx[i]))<qx_min_tot):
                    qx_min_tot = min(numpy.ravel(qx[i]));
                if (max(numpy.ravel(qx[i]))>qx_max_tot):
                    qx_max_tot = max(numpy.ravel(qx[i]))
                if (min(numpy.ravel(qz[i]))<qz_min_tot):
                    qz_min_tot = min(numpy.ravel(qz[i]));
                if (max(numpy.ravel(qz[i]))>qz_max_tot):
                    qz_max_tot = max(numpy.ravel(qz[i]));
            
            for i in range(len(qx)):                
                pl.contourf(qx[i],qz[i],self.maplog(iq[i],cf[0],cf[1]),colconts);

            pl.axis([qx_min_tot,qx_max_tot,qz_min_tot,qz_max_tot]);
        else:            
            cplot= pl.contourf(qx,qz,self.maplog(iq,cf[0],cf[1]),colconts);


        #get level information from the cplot program
        levels = cplot.levels;
        noflevels = len(levels);
    
        cb = pl.colorbar(format="1e%4.2f",ticks=[levels[0],levels[int(noflevels/4.0)],levels[int(noflevels/2.0)],\
                            levels[int(3*noflevels/4.0)],levels[-1]]);
        if bwconts != 0:
            if isinstance(qx,list):
                pl.hold(True);
                for i in range(len(qx)):
                    pl.contour(qx[i],qz[i],self.maplog(iq[i],cf[0],cf[1]),bwconts,colors='black');
            else:
                cplot=pl.contour(qx,qz,self.maplog(iq,cf[0],cf[1]),bwconts,colors='black');

        #set the correct number of x-tick labels
        axis = pl.gca();
        xticks = axis.get_xticks();
        xmin = min(xticks);
        xmax = max(xticks);

        xtick_list = [];
        if xmin<0.0 and xmax>0.0:
            #in the case of a symmetric map
            xstep = max([abs(xmin),xmax])/2.0;
            xtick_list=[-2.0*xstep,-xstep,0.0,xstep,2.0*xstep];
        else:
            xstep = (xmax-xmin)/4.0;
            for i in range(5):
                xtick_list.append(xmin+i*xstep);

        axis.set_xticks(xtick_list);
        

        #set labels and colorbars
        pl.xlabel(self.mapplot_xlabel);
        pl.ylabel(self.mapplot_ylabel);
        pl.title(plot_title);
                
        pl.draw();
        #pl.axis('equal');
        
        return fig;
    

        
        
class hxrd(xraymethod):
    """
    This class handles data measured with HXRD (High resolution x-ray diffraction).
    """

    def __init__(self,det,**keyarg):
        xraymethod.__init__(self,det,keyarg);
        
        self.mapplot_xlabel = "qx"; #r'$q_{||}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_ylabel = "qz"; #r'$q_{z}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_title  = 'HXRD map';

    def q2ang(self,qx,qz,**keyargs):
        """
        q2ang(qx,qz,**keyargs):
        Calculate the scattering angles for a given q-space position.

        required input arguments:
        qx ..................... qx position
        qz ..................... qz position

        optional keyword arguments:
        dunit .................. ="rad"/"deg" output in radiants or degree

        """

        k0 = 2.*numpy.pi/self.wavelength
        rad2deg = 180./numpy.pi

        if keyargs.has_key("unit"):
            unit = keyargs["unit"]
        else:
            unit = "deg"

        Q = numpy.sqrt(qx**2+qz**2)
        tth = 2.*numpy.arcsin(Q/2./k0)
        om  = numpy.arcsin(qx/Q)+numpy.arcsin(Q/2./k0)

        if unit=="deg":
            tth = rad2deg*tth
            om  = rad2deg*om

        return [om,tth]

    def ang2q(self,om,tth,**keyarg):
        """
        Converts the measured data from angular space to reciprocal space.
        The required input parameters are:
        om ........ the omega matrix
        tth ....... the 2theta matrix

        optional arguments are
        dom ....... a shift of the omega values
        dtth ...... a shift of the 2theta values
        geom ...... -1 low incidence/high exit
                    1 (default) high incidence/low exit

        It returns 2 matrices [qx,qz]        
        """

        dom = 0.0;
        dtth = 0.0;

        k=2.0*numpy.pi/self.wavelength; #calculate the k vector
        deg2rad = numpy.pi/180.0;

        if keyarg.has_key("geom"):
            geom = keyarg["geom"]
        else:
            geom = 1.

        #check for optional parameters:
        if keyarg.has_key('dom'):
            dom = keyarg['dom'];

        if keyarg.has_key('dtth'):
            dtth = keyarg['dtth'];

        qx=2.0*k*numpy.sin((tth-dtth)*deg2rad/2)*\
            numpy.sin(geom*((om-dom)-(tth-dtth)/2)*deg2rad);
        qz=2.0*k*numpy.sin((tth-dtth)*deg2rad/2)*\
            numpy.cos(geom*((om-dom)-(tth-dtth)/2)*deg2rad);        

        return [qx,qz];    
        
        
    
class gid(xraymethod):
    """
    A class providing functions necessary for evaluating GID data.
    """    

    def __init__(self,det,ai,ao,**keyarg):
        xraymethod.__init__(self,det,keyarg);
        self.alpha_i = numpy.pi*ai/180.0;
        self.alpha_o = numpy.pi*ao/180.0;
        
        self.mapplot_xlabel = r'$q_{a}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_ylabel = r'$q_{r}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_title  = 'GID map';

    def ang2q(self,om,tth,**keyargs):
        """
        Calculates the radial, angular and z-components in q-space from the angular values
        Required input arguments are:
            om .................. a matrix with the omega values
            tth ................. a matrix with the 2th values    
        optional keyword arguments:
            dom ................. delta omega correction
            dth ................. delta theta correction
        Note: om and tth must we of same shape
        """
        
        if keyargs.has_key("dom"):
            dom = numpy.pi*keyargs["dom"]/180.0;
        else: 
            dom = 0.0;
        
        if keyargs.has_key("dth"):
            dth = numpy.pi*keyargs["dth"]/180.0;
        else:
            dth = 0.0;
        
        k = 2.0*numpy.pi/self.wavelength;
        om_rad = numpy.pi*om/180.0-dom;
        tth_rad = numpy.pi*tth/180.0-dth;
        qparallel = k*numpy.sqrt(numpy.cos(self.alpha_i)**2-2.0*numpy.cos(self.alpha_i)* \
                                   numpy.cos(self.alpha_o)*numpy.cos(tth_rad)+numpy.cos(self.alpha_o)**2);
                   
        beta = numpy.arcsin(k*numpy.cos(self.alpha_o)*numpy.sin(tth_rad)/qparallel);
        betap = numpy.pi/2.0 - om_rad - beta;
    
        qa = qparallel*numpy.sin(betap);
        qr = qparallel*numpy.cos(betap);
        qz = k*(numpy.sin(self.alpha_i)+numpy.sin(self.alpha_o));
    
        return [qa,qr,qz];
        

class gisaxs(xraymethod):
    def __init__(self,det,**keyargs):
        xraymethod.__init__(self,det,keyargs)

    def ang2q(self,ai,af,tth):
        """
        ang2q(ai,af,tth):
        Converts angular to q-space data for GISAXS measurements

        required input arguments:
        ai ................ angle of incidence
        af ................ exit angle
        tth ............... scattering angle

        return values
        [qx,qy,qz] ........ arrays with q-coordinates
        """

        k = numpy.pi*2.0/self.wavelength
        ai_tmp = numpy.pi*ai/180.
        af_tmp = numpy.pi*af/180.
        tth_tmp = numpy.pi*tth/180.
        qx = k*(numpy.cos(ai_tmp)-numpy.cos(af_tmp)*numpy.cos(tth_tmp))
        qy = k*numpy.cos(af_tmp)*numpy.sin(tth_tmp)
        qz = k*(numpy.sin(ai_tmp)+numpy.sin(af_tmp))

        return[qx,qy,qz]

class xrr(xraymethod):
    """
    xrr:
    A class providing methos for handling x-ray reflectivity data.
    """
    
    
    def __init__(self,det,**keyarg):
        xraymethod.__init__(self,det,keyarg);
        
        self.mapplot_xlabel = r'$q_{a}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_ylabel = r'$q_{r}\hspace{0.5}(1/A^{\circ})$';
        self.mapplot_title  = 'GID map';

    def ang2q(self,om,tth,**keyarg):
        """
        ang2q(om,tth):
        Converst measured data from angular space to q-space.
        
        Input arguments:
        om ........... an array holding the measured omega values
        tth .......... an array holding the measured 2th values
        optional arguments:
        dom ....... a shift of the omega values
        dtth ...... a shift of the 2theta values
        nqx ....... number of points in x-direction
        nqz ....... number of points in z-direction
        Return values:
         [qx,qz] ...... qx and qy values for the measured angles
        
        """

        dom = 0.0;
        dtth = 0.0;

        #calculate some general quantities

        k=2*numpy.pi/self.wavelength; #calculate the k vector
        deg2rad = numpy.pi/180.0;

        #check for optional parameters:
        if keyarg.has_key('dom'):
            dom = keyarg['dom'];

        if keyarg.has_key('dtth'):
            dtth = keyarg['dtth'];

        tth = tth - dtth;
        om  = om - dom;

        qx=2.0*k*numpy.sin(tth*deg2rad/2)*numpy.sin((om-tth/2)*deg2rad);
        qz=2.0*k*numpy.sin(tth*deg2rad/2)*numpy.cos((om-tth/2)*deg2rad);
        
        return [qx,qz];            






    
    

    
    
    
