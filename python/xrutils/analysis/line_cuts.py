# 2011/05/28 initial import of line cut functions, Dominik Kriegner/Tanja Etzelstorfer
import numpy

from .. import config
from .. import experiment

def getindex(x,y,xgrid,ygrid):
    """
    gives the indices of the point x,y in the grid given by xgrid ygrid
    xgrid,ygrid must be arrays containing equidistant points
    
    Parameters
    ----------
     x,y:   coordinates of the point of interest (float)
     xgrid,
     ygrid: grid coordinates in x and y direction (array)
     
    Returns
    -------
     ix,iy: index of the closest gridpoint (lower left) of the point (x,y) 
    """
    dx = xgrid[1]-xgrid[0]
    dy = ygrid[1]-ygrid[0]

    ix = int((x-xgrid[0])/dx)
    iy = int((y-ygrid[0])/dy)

    #check if index is in range of the given grid
    # to speed things up this is assumed to be the case
    #if(ix<0 or ix>xgrid.size): print "Warning: point (",x,",",y,") out of range in first coordinate!"
    #if(iy<0 or iy>ygrid.size): print "Warning: point (",x,",",y,") out of range in second coordinate!"

    return ix,iy

def get_qx_scan(qx,qz,intensity,qzpos,**kwargs):
    """
    extract qx line scan at position qzpos from a 
    gridded reciprocal space map by taking the closest line of the 
    intensity matrix, or summing up a given range along qz   

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    qzpos: position at which the line scan should be extracted
    **kwargs: possible keyword arguments:
        qrange: integration range perpendicular to scan direction
        qmin,qmax: minimum and maximum value of extracted scan axis
        bounds: flag to specify if the scan bounds of the extracted scan should be returned (default:False)       

    Returns
    -------
    qx,qxint: qx scan coordinates and intensities (bounds=False)
    qx,qxint,(qxb,qyb): qx scan coordinates and intensities + scan bounds for plotting

    Example
    -------
    >>> qxcut,qxcut_int = get_qx_scan(qx,qz,inten,5.0,qrange=0.03)
    """
    
    if qzpos < qz.min() or qzpos > qz.max():
        raise ValueError("given qzpos is not in the range of the given qz axis")
        
    if intensity.shape != (qx.size,qz.size):
        raise ValueError("shape of given intensity does not match to (qx.size,qz.size)")
    
    if kwargs.has_key('qmin'):
        qxmin = max(qx.min(),kwargs['qmin']) 
    else: qxmin = qx.min()
 
    if kwargs.has_key('qmax'):
        qxmax = min(qx.max(),kwargs['qmax']) 
    else: qxmax = qx.max()
    
    if kwargs.has_key('qrange'):
        qrange = kwargs['qrange']
    else: qrange = 0.

    if kwargs.has_key('bounds'):
        bounds = kwargs['bounds']
    else: bounds = False
     
    # find line corresponding to qzpos 
    ixmin,izmin = getindex(qxmin,qzpos-qrange/2.,qx,qz)
    ixmax,izmax = getindex(qxmax,qzpos+qrange/2.,qx,qz)
    
    # scan bounds for plotting if requested
    qxbounds = (numpy.array((qxmin,qxmax,qxmax,qxmin,qxmin)), \
      numpy.array((qzpos-qrange/2.,qzpos-qrange/2.,qzpos+qrange/2.,qzpos+qrange/2.,qzpos-qrange/2.)))
      
    if qrange > 0:
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_q[x,z]_scan: %d points used for integration" %(izmax-izmin+1))
        if bounds:
            return qx[ixmin:ixmax+1],intensity[ixmin:ixmax+1,izmin:izmax+1].sum(axis=1)/(izmax-izmin+1),qxbounds
        else:
            return qx[ixmin:ixmax+1],intensity[ixmin:ixmax+1,izmin:izmax+1].sum(axis=1)/(izmax-izmin+1)
    else:
        if bounds:
            return qx[ixmin:ixmax+1],intensity[ixmin:ixmax+1,izmin],qxbounds
        else:
            return qx[ixmin:ixmax+1],intensity[ixmin:ixmax+1,izmin]
        
def get_qz_scan(qx,qz,intensity,qxpos,**kwargs):
    """
    extract qz line scan at position qxpos from a 
    gridded reciprocal space map by taking the closest line of the 
    intensity matrix, or summing up a given range along qx   

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    qxpos: position at which the line scan should be extracted
    **kwargs: possible keyword arguments:
        qrange: integration range perpendicular to scan direction
        qmin,qmax: minimum and maximum value of extracted scan axis            

    Returns
    -------
    qz,qzint: qz scan coordinates and intensities 

    Example
    -------
    >>> qzcut,qzcut_int = get_qz_scan(qx,qz,inten,1.5,qrange=0.03)
    """
    
    if kwargs.has_key('bounds'):
        bounds = kwargs['bounds']
    else: bounds = False
    
    if bounds:
        qzc,qzcint,(qzb,qxb) = get_qx_scan(qz,qx,intensity.transpose(),qxpos,**kwargs) 
        return qzc,qzcint,(qxb,qzb)
    else:
        return get_qx_scan(qz,qx,intensity.transpose(),qxpos,**kwargs) 


def get_omega_scan_q(qx,qz,intensity,qxcenter,qzcenter,omrange,npoints,**kwargs):
    """
    extracts an omega scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    qxcenter: qx-position at which the omega scan should be extracted
    qzcenter: qz-position at which the omega scan should be extracted
    omrange: range of the omega scan to extract
    npoints: number of points of the omega scan
    **kwargs: possible keyword arguments:
        qrange: integration range perpendicular to scan direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative omega positions are returned (default: True)
        bounds: flag to specify if the scan bounds should be returned (default: False)

    Returns
    -------
    om,omint: omega scan coordinates and intensities (bounds=False)
    om,omint,(qxb,qzb): omega scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)
    
    Example
    -------
    >>> omcut, intcut = get_omega_scan(qx,qz,intensity,0.0,5.0,2.0,200)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    # angular coordinates of scan center
    dummy,omcenter,ttcenter = exp.Q2Ang(0,qxcenter,qzcenter,trans=False)

    return get_omega_scan_ang(qx,qz,intensity,omcenter,ttcenter,omrange,npoints,**kwargs)


def get_omega_scan_ang(qx,qz,intensity,omcenter,ttcenter,omrange,npoints,**kwargs):
    """
    extracts an omega scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    omcenter: omega-position at which the omega scan should be extracted
    ttcenter: 2theta-position at which the omega scan should be extracted
    omrange: range of the omega scan to extract
    npoints: number of points of the omega scan
    **kwargs: possible keyword arguments:
        qrange: integration range perpendicular to scan direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative omega positions are returned (default: True)
        bounds: flag to specify if the scan bounds should be returned (default: False)

    Returns
    -------
    om,omint: omega scan coordinates and intensities (bounds=False)
    om,omint,(qxb,qzb): omega scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)
    
    Example
    -------
    >>> omcut, intcut = get_omega_scan(qx,qz,intensity,0.0,5.0,2.0,200)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    if kwargs.has_key('relative'):
        relative = kwargs['relative']
    else: relative = True
    
    if kwargs.has_key('qrange'):
        qrange = kwargs['qrange']
    else: 
        qrange = 0.
    dummy,qxcenter,qzcenter = exp.Ang2Q(omcenter,ttcenter)
    qxcenter = qxcenter[0]; qzcenter = qzcenter[0]
    dom_m = exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2),trans=False)[1] - exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2)-qrange/2.,trans=False)[1]
    dom_p = exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2)+qrange/2.,trans=False)[1] - exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2),trans=False)[1]
    
    if kwargs.has_key('Nint'):
        nint = kwargs['Nint']
    else:
        nint = numpy.ceil(max(qrange/(qx[1]-qx[0]),qrange/(qz[1]-qz[0]) ))
        if nint == 0: nint = 1
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_omega_scan: using %d subscans for integration" % (nint))

    if kwargs.has_key('bounds'):
        bounds = kwargs['bounds']
    else: bounds = False
    
    # angles of central line scan
    omscan = omcenter-omrange/2. + omrange/(1.0*npoints)*numpy.arange(npoints)
    ttscan = numpy.ones(npoints)*ttcenter

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-2*dom_m,2*dom_p,nint)[numpy.newaxis,:]

    intOM1d = numpy.zeros(OMS.size)
    OMS1d = OMS.reshape(OMS.size)
    TTS1d = TTS.reshape(OMS.size)
    dummy,qxS,qzS = exp.Ang2Q(OMS1d,TTS1d)

    # determine omega scan intensities by look up in the gridded RSM
    for i in range(OMS.size):
        ix,iz = getindex(qxS[i],qzS[i],qx,qz)
        if (ix>=0 and ix<qx.size and iz>=0 and iz <= qz.size):
            intOM1d[i] = intensity[ix,iz]

    intOM = intOM1d.reshape((npoints,nint))
    intom = intOM.sum(axis=1)/float(nint)

    if relative:
        omscan = omscan - omcenter
        
    if bounds:
        qxb,qzb = get_omega_scan_bounds_ang(omcenter,ttcenter,omrange,npoints,**kwargs)
        return omscan,intom,(qxb,qzb)
    else:
        return omscan,intom


def get_omega_scan_bounds_ang(omcenter,ttcenter,omrange,npoints,**kwargs):
    """
    return reciprocal space boundaries of omega scan  

    Parameters
    ----------
    omcenter: omega-position at which the omega scan should be extracted
    ttcenter: 2theta-position at which the omega scan should be extracted
    omrange: range of the omega scan to extract
    npoints: number of points of the omega scan
    **kwargs: possible keyword arguments:
        qrange: integration range perpendicular to scan direction 
        lam: wavelength for use in the conversion to angular coordinates         

    Returns
    -------
    qx,qz: reciprocal space coordinates of the omega scan boundaries 

    Example
    -------
    >>> qxb,qzb = get_omega_scan_bounds_ang(1.0,4.0,2.4,240,qrange=0.1)
    """
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])
    
    if kwargs.has_key('qrange'):
        qrange = kwargs['qrange']
    else: 
        qrange = 0.
    dummy,qxcenter,qzcenter = exp.Ang2Q(omcenter,ttcenter)
    qxcenter = qxcenter[0]; qzcenter = qzcenter[0]
    dom_m = exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2),trans=False)[1] - exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2)-qrange/2.,trans=False)[1]
    dom_p = exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2)+qrange/2.,trans=False)[1] - exp.Q2Ang(0.,0.,numpy.sqrt(qxcenter**2+qzcenter**2),trans=False)[1]
    
    nint=2
       
    # angles of central line scan
    omscan = omcenter-omrange/2. + omrange/(1.0*npoints)*numpy.arange(npoints)
    ttscan = numpy.ones(npoints)*ttcenter

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-2*dom_m,2*dom_p,nint)[numpy.newaxis,:]
    OMSnew = numpy.zeros((npoints,nint))
    TTSnew = numpy.zeros((npoints,nint))

    # invert order of second half of angular coordinates
    OMSnew[:,0] = OMS[:,0]
    TTSnew[:,0] = TTS[:,0] 
    for i in range(npoints):
        OMSnew[i,1] = OMS[-1-i,1]
        TTSnew[i,1] = TTS[-1-i,1]

    OMS1d = OMSnew.transpose().flatten()
    TTS1d = TTSnew.transpose().flatten()
    dummy,qx,qz = exp.Ang2Q(OMS1d,TTS1d)

    return numpy.append(qx,qx[0]),numpy.append(qz,qz[0])


def get_radial_scan_q(qx,qz,intensity,qxcenter,qzcenter,ttrange,npoints,**kwargs):
    """
    extracts a radial scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    qxcenter: qx-position at which the radial scan should be extracted
    qzcenter: qz-position at which the radial scan should be extracted
    ttrange: two theta range of the radial scan to extract
    npoints: number of points of the radial scan
    **kwargs: possible keyword arguments:
        omrange: integration range perpendicular to scan direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative two theta positions are returned (default=True)
        bounds: flag to specify if the scan bounds should be returned (default: False)
        
    Returns
    -------
    om,tt,radint: omega,two theta scan coordinates and intensities (bounds=False)
    om,tt,radint,(qxb,qzb): radial scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)
    Example
    -------
    >>> omc,ttc,cut_int = get_radial_scan_q(qx,qz,intensity,0.0,5.0,1.0,100,omrange=0.01)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    # angular coordinates of scan center
    dummy,omcenter,ttcenter = exp.Q2Ang(0,qxcenter,qzcenter,trans=False)
    
    return get_radial_scan_ang(qx,qz,intensity,omcenter,ttcenter,ttrange,npoints,**kwargs)


def get_radial_scan_ang(qx,qz,intensity,omcenter,ttcenter,ttrange,npoints,**kwargs):
    """
    extracts a radial scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    omcenter: om-position at which the radial scan should be extracted
    ttcenter: tt-position at which the radial scan should be extracted
    ttrange: two theta range of the radial scan to extract
    npoints: number of points of the radial scan
    **kwargs: possible keyword arguments:
        omrange: integration range perpendicular to scan direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative two theta positions are returned (default=True)
        bounds: flag to specify if the scan bounds should be returned (default: False)
        
    Returns
    -------
    om,tt,radint: omega,two theta scan coordinates and intensities (bounds=False)
    om,tt,radint,(qxb,qzb): radial scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)

    Example
    -------
    >>> omc,ttc,cut_int = get_radial_scan_ang(qx,qz,intensity,32.0,64.0,30.0,800,omrange=0.2)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    if kwargs.has_key('relative'):
        relative = kwargs['relative']
    else: relative = True
    
    if kwargs.has_key('omrange'):
        omrange = kwargs['omrange']
    else: 
        omrange = 0.
    dom_m = omrange/2.
    dom_p = omrange/2.
    qrange = numpy.abs(exp.Ang2Q(omcenter-dom_m,ttcenter)[1] - exp.Ang2Q(omcenter+dom_p,ttcenter)[1])
    
    if kwargs.has_key('Nint'):
        nint = kwargs['Nint']
    else:
        nint = numpy.ceil(qrange/(qx[1]-qx[0]))
        if nint == 0: nint = 1
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_radial_scan: using %d subscans for integration" % (nint))

    if kwargs.has_key('bounds'):
        bounds = kwargs['bounds']
    else: bounds = False   
    
    # angles of central line scan
    omscan = omcenter-ttrange/4. + ttrange/(2.0*npoints)*numpy.arange(npoints)
    ttscan = ttcenter-ttrange/2. + ttrange/(1.0*npoints)*numpy.arange(npoints)

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) 

    intrad1d = numpy.zeros(OMS.size)
    OMS1d = OMS.reshape(OMS.size)
    TTS1d = TTS.reshape(OMS.size)
    dummy,qxS,qzS = exp.Ang2Q(OMS1d,TTS1d)

    # determine radial scan intensities by look up in the gridded RSM
    for i in range(OMS.size):
        ix,iz = getindex(qxS[i],qzS[i],qx,qz)
        if (ix>=0 and ix<qx.size and iz>=0 and iz <= qz.size):
            intrad1d[i] = intensity[ix,iz]

    intRAD = intrad1d.reshape((npoints,nint))
    intrad = intRAD.sum(axis=1)/float(nint)

    if relative:
        omscan = omscan - omcenter
        ttscan = ttscan - ttcenter

    if bounds:
        qxb,qzb = get_radial_scan_bounds_ang(omcenter,ttcenter,ttrange,npoints,**kwargs)
        return omscan,ttscan,intrad,(qxb,qzb)
    else:
        return omscan,ttscan,intrad 


def get_radial_scan_bounds_ang(omcenter,ttcenter,ttrange,npoints,**kwargs):
    """
    return reciprocal space boundaries of radial scan  

    Parameters
    ----------
    omcenter: om-position at which the radial scan should be extracted
    ttcenter: tt-position at which the radial scan should be extracted
    ttrange: two theta range of the radial scan to extract
    npoints: number of points of the radial scan
    **kwargs: possible keyword arguments:
        omrange: integration range perpendicular to scan direction
        lam: wavelength for use in the conversion to angular coordinates

    Returns
    -------
    qxrad,qzrad: reciprocal space boundaries of radial scan 

    Example
    -------
    >>>
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])
    
    if kwargs.has_key('omrange'):
        omrange = kwargs['omrange']
    else: 
        omrange = 0.
    dom_m = omrange/2.
    dom_p = omrange/2.

    nint = 2    

    # angles of central line scan
    omscan = omcenter-ttrange/4. + ttrange/(2.0*npoints)*numpy.arange(npoints)
    ttscan = ttcenter-ttrange/2. + ttrange/(1.0*npoints)*numpy.arange(npoints)

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) 
    OMSnew = numpy.zeros((npoints,nint))
    TTSnew = numpy.zeros((npoints,nint))

    # invert order of second half of angular coordinates
    OMSnew[:,0] = OMS[:,0]
    TTSnew[:,0] = TTS[:,0] 
    for i in range(npoints):
        OMSnew[i,1] = OMS[-1-i,1]
        TTSnew[i,1] = TTS[-1-i,1]

    OMS1d = OMSnew.transpose().flatten()
    TTS1d = TTSnew.transpose().flatten()
    dummy,qx,qz = exp.Ang2Q(OMS1d,TTS1d)

    return numpy.append(qx,qx[0]),numpy.append(qz,qz[0])


def get_ttheta_scan_q(qx,qz,intensity,qxcenter,qzcenter,ttrange,npoints,**kwargs):
    """
    extracts a twotheta scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    qxcenter: qx-position at which the 2theta scan should be extracted
    qzcenter: qz-position at which the 2theta scan should be extracted
    ttrange: two theta range of the scan to extract
    npoints: number of points of the radial scan
    **kwargs: possible keyword arguments:
        omrange: integration range in omega direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative two theta positions are returned (default=True)
        bounds: flag to specify if the scan bounds should be returned (default: False)
        
    Returns
    -------
    tt,ttint: two theta scan coordinates and intensities (bounds=False)
    om,tt,radint,(qxb,qzb): radial scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)
    
    Example
    -------
    >>> ttc,cut_int = get_ttheta_scan_q(qx,qz,intensity,0.0,4.0,4.4,440)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    # angular coordinates of scan center
    dummy,omcenter,ttcenter = exp.Q2Ang(0,qxcenter,qzcenter,trans=False)
    
    return get_ttheta_scan_ang(qx,qz,intensity,omcenter,ttcenter,ttrange,npoints,**kwargs)


def get_ttheta_scan_ang(qx,qz,intensity,omcenter,ttcenter,ttrange,npoints,**kwargs):
    """
    extracts a twotheta scan from a gridded reciprocal space map    

    Parameters
    ----------
    qx: equidistant array of qx momentum transfer
    qz: equidistant array of qz momentum transfer
    intensity: 2D array of gridded reciprocal space intensity with shape (qx.size,qz.size)
    omcenter: om-position at which the 2theta scan should be extracted
    ttcenter: tt-position at which the 2theta scan should be extracted
    ttrange: two theta range of the scan to extract
    npoints: number of points of the radial scan
    **kwargs: possible keyword arguments:
        omrange: integration range in omega direction
        Nint: number of subscans used for the integration (optionally)
        lam: wavelength for use in the conversion to angular coordinates
        relative: determines if absolute or relative two theta positions are returned (default=True)
        bounds: flag to specify if the scan bounds should be returned (default: False)
        
    Returns
    -------
    tt,ttint: two theta scan coordinates and intensities (bounds=False)
    tt,ttint,(qxb,qzb): 2theta scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)

    Example
    -------
    >>> ttc,cut_int = get_ttheta_scan_ang(qx,qz,intensity,32.0,64.0,4.0,400)
    """
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    if kwargs.has_key('relative'):
        relative = kwargs['relative']
    else: relative = True
    
    if kwargs.has_key('omrange'):
        omrange = kwargs['omrange']
    else: 
        omrange = 0.
    dom_m = omrange
    dom_p = omrange
    
    qrange = numpy.abs(exp.Ang2Q(omcenter-dom_m,ttcenter)[1] - exp.Ang2Q(omcenter+dom_p,ttcenter)[1])
    
    if kwargs.has_key('Nint'):
        nint = kwargs['Nint']
    else:
        nint = numpy.ceil(qrange/(qx[1]-qx[0]))
        if nint == 0: nint = 1
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.analysis.get_radial_scan: using %d subscans for integration" % (nint))

    if kwargs.has_key('bounds'):
        bounds = kwargs['bounds']
    else: bounds = False   
    
    # angles of central line scan
    omscan = omcenter*numpy.ones(npoints)
    ttscan = ttcenter-ttrange/2. + ttrange/(1.0*npoints)*numpy.arange(npoints)

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) 

    inttt1d = numpy.zeros(OMS.size)
    OMS1d = OMS.reshape(OMS.size)
    TTS1d = TTS.reshape(OMS.size)
    dummy,qxS,qzS = exp.Ang2Q(OMS1d,TTS1d)

    # determine radial scan intensities by look up in the gridded RSM
    for i in range(OMS.size):
        ix,iz = getindex(qxS[i],qzS[i],qx,qz)
        if (ix>=0 and ix<qx.size and iz>=0 and iz <= qz.size):
            inttt1d[i] = intensity[ix,iz]

    intTT = inttt1d.reshape((npoints,nint))
    inttt = intTT.sum(axis=1)/float(nint)

    if relative:
        ttscan = ttscan - ttcenter

    if bounds:
        qxb,qzb = get_ttheta_scan_bounds_ang(omcenter,ttcenter,ttrange,npoints,**kwargs)
        return ttscan,inttt,(qxb,qzb)
    else:
        return ttscan,inttt 
        

def get_ttheta_scan_bounds_ang(omcenter,ttcenter,ttrange,npoints,**kwargs):
    """
    return reciprocal space boundaries of 2theta scan  

    Parameters
    ----------
    omcenter: om-position at which the 2theta scan should be extracted
    ttcenter: tt-position at which the 2theta scan should be extracted
    ttrange: two theta range of the 2theta scan to extract
    npoints: number of points of the 2theta scan
    **kwargs: possible keyword arguments:
        omrange: integration range in omega direction
        lam: wavelength for use in the conversion to angular coordinates

    Returns
    -------
    qxtt,qztt: reciprocal space boundaries of 2theta scan (bounds=False)   
    tt,ttint,(qxb,qzb): 2theta scan coordinates and intensities + 
                        reciprocal space bounds of the extraced scan (bounds=True)

    Example
    -------
    >>>
    """   
    
    if kwargs.has_key('lam'):
        lam = kwargs['lam']
        exp = experiment.HXRD([1,0,0],[0,0,1],wl=lam)
    else: 
        exp = experiment.HXRD([1,0,0],[0,0,1])

    if kwargs.has_key('omrange'):
        omrange = kwargs['omrange']
    else: 
        omrange = 0.
    dom_m = omrange
    dom_p = omrange
    
    nint = 2    

    # angles of central line scan
    omscan = omcenter*numpy.ones(npoints)
    ttscan = ttcenter-ttrange/2. + ttrange/(1.0*npoints)*numpy.arange(npoints)

    # angles for subscans used for integration
    OMS = omscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) + numpy.linspace(-dom_m,dom_p,nint)[numpy.newaxis,:]
    TTS = ttscan[:,numpy.newaxis]*numpy.ones((npoints,nint)) 
    
    OMSnew = numpy.zeros((npoints,nint))
    TTSnew = numpy.zeros((npoints,nint))

    # invert order of second half of angular coordinates
    OMSnew[:,0] = OMS[:,0]
    TTSnew[:,0] = TTS[:,0] 
    for i in range(npoints):
        OMSnew[i,1] = OMS[-1-i,1]
        TTSnew[i,1] = TTS[-1-i,1]

    OMS1d = OMSnew.transpose().flatten()
    TTS1d = TTSnew.transpose().flatten()
    dummy,qx,qz = exp.Ang2Q(OMS1d,TTS1d)

    return numpy.append(qx,qx[0]),numpy.append(qz,qz[0])
    
