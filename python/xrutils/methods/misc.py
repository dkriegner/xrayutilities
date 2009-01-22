#this module holds some general macros for xray stuff that is not
#specific to a certain method. They help for evaluation of measured 
#data and to make some basic calculations necessary to perform a
#a scattering experiment.

import Numeric

#************************************************************
def scattang(scatt_vec,surf_vec,a,geom,lam):
    '''scattang: calculates the scattering angle of a reflection
       input parameters:
       scatt_vec=[h,k,l] ..... a list with the Miller indices of the
                               scattering vector
       surf_vec=[h,k,l] ...... a list with the Miller indices of the
                               crystallographic orientation of the surface
       a ..................... the lattice parameter
       geom .................. scattering geometry, the possible values are:
                               -1 .... high incidence/low exit
                                1 .... low incidence/high exit
       lam ................... wavelength of the used x-ray radiation in
                               Angstrom
       return value:
       the function returns a list where the first entry is the omega (sample)
       angle and the second the 2theta (detector) angle (both in degree).
    '''
    
    deg2rad = Numeric.pi/180;
    rad2deg = 180/Numeric.pi;

    #calculate the angle between the surface
    #vector and the scattering vector
    ang = _vec_ang(scatt_vec,surf_vec);

    dhkl = a/_vec_abs(scatt_vec);
    

    scatt_ang = Numeric.arcsin(lam/2/dhkl);

    if abs(ang)!=Numeric.pi/2:
        #not in GID
        oms = rad2deg*(scatt_ang+geom*ang);
        tth = rad2deg*(2*scatt_ang);
    else:
        #in GID (scattering vector and surface vector are 90 degree)
        oms = rad2deg*scatt_ang;
        tth = rad2deg*2*scatt_ang;

    print 'scatt/surf angle: %f' %(rad2deg*ang)
    print 'omega = %f' %(oms)
    print 'tth   = %f' %(tth)

    return [oms,tth];

#************************************************************
#converting between energy and wavelength
def lam2en(lam):    
    '''lam2en: convert a wavelength to an energy
       input parameter:
       lam ........... wavelength in Angstrom
       return value:
       energy in eV
    '''
    e=1.60219e-19;
    h=6.62602e-34;
    c=2.997925e8;
    Energy = 0;
    lam_m  = 0;

    #convert the wavelength to meter
    lam_m = lam*1e-10;
  
    #calculate the corresponding energy
    Energy = c*h/lam_m;
    #convert the energy to ev
    Energy = Energy/e;

    print 'energy: %f (ev)' %(Energy)

    return(Energy);

#************************************************************
def en2lam(Energy):    
    '''en2lam: converts an energy to a wavelength
       input parameter:
       energy .......... energy in eV
       return value:
       wavelength in Angstrom
    '''
    e=1.60219e-19;
    h=6.62602e-34;
    c=2.997925e8;
    lam = 0;
  
    #calculate the wavelength
    lam = c*h/Energy/e;
    #convert the wavelength to Angstrom
    lam = lam/1e-10;

    print 'wavelength: %f (Angstrom)' %(lam);
    
    return(lam);

#************************************************************
def getq(aLattice,hkl):
    '''getq: calculates the q value for a given lattice parameter
    and lattice plane
    input parameters:
    aLattice ......... lattice parameter in Angstrom
    hkl=[h,k,l] ...... Miller indices of the reciprocal lattice plane
    return value:
    a list q with the following compontents:
    q[0] = qx
    q[1] = qy
    q[2] = qz
    q[3] = |q|
    (the fourth component is the absolute value of the vector in
    reciprocal space)     
    '''
    qabs = 0;
    q    = [];
    
    if len(aLattice) != len(hkl):
        print 'list dimensions are not equal'
        return 0
    
    for i in range(len(aLattice)):
        q.append(2*Numeric.pi*hkl[i]/aLattice[i]);
        print 'q_%i = %f' %(i,q[i]);

    return(q);


#************************************************************
def geta(q,hkl):
    '''geta: calculates the lattice parameter from a set of
    reciprocal space coordinates
    input parameters:
    q=[] .... a list of variable length holding the
    q-space coordinates
    hkl=[] .. a list of variable length holding the
    miller indices (or their absolute values)
    corresponding to the q values
    NOTE: both lists must have the same length!!!
    EXAMPLE:
    We have qx and qz from a 2D RSM of a (224) diffraction.
    q = [qx,qz] and hkl=[sqrt(8)=sqrt(2^2+2^2),4]
    return value:
    a list with the same length as q holding the
    corresponding lattice parameters.
    '''
    aLattice = [];
    
    if len(q) != len(hkl):
        print 'list dimension not equal!'
        return 0
    
    for i in range(len(q)):
        aLattice.append(2*Numeric.pi*hkl[i]/q[i]);
        print 'lattice parameter %i = %f' %(i,aLattice[i])
        
    return(aLattice);

#************************************************************  
def tiltcorr(qxsym,qzsym,qxasym,qzasym,lam):
    '''tiltcorr: calculates the tilt corrected values of a symmetric
                 and a asymmetric diffraction

    '''

    k = 2*Numeric.pi/lam;
  
  
    qsym  = Numeric.sqrt(qxsym**2+qzsym**2);
    qasym = Numeric.sqrt(qxasym**2+qzasym**2);
    
    #calculate the tilt angle
    alpha = Numeric.arcsin(qxsym/qsym);
  
    thasym = Numeric.arcsin(qasym/(2*k));
    omasym = Numeric.arcsin(qxasym/qasym)+thasym;
    thsym  = Numeric.arcsin(qsym/(2*k));
    omsym  = Numeric.arcsin(qxsym/qsym)+thsym;
  
    #correct omega
    omasym = omasym - alpha;
    omsym  = omsym - alpha;

    #calclate the corrected q values
    qxsym_corr  = 2*k*Numeric.sin(thsym)*Numeric.sin(omsym-thsym);
    qzsym_corr  = 2*k*Numeric.sin(thsym)*Numeric.cos(omsym-thsym);
    qxasym_corr = 2*k*Numeric.sin(thasym)*Numeric.sin(omasym-thasym);
    qzasym_corr = 2*k*Numeric.sin(thasym)*Numeric.cos(omasym-thasym);
  
    #print the correted values
    print 'corrected position of symmetric peak:';
    print 'Qx = %f' %(qxsym_corr);
    print 'Qz = %f' %(qzsym_corr);
    print 'corrected position of asymmetric peak:';
    print 'Qx = %f' %(qxasym_corr);
    print 'Qz = %f' %(qzasym_corr);
    print 'tilt = %f' %(180*alpha/Numeric.pi);

    #return values to the calling instance
    return ([qxsym_corr,qzsym_corr,qxasym_corr,qzasym_corr,180*alpha/Numeric.pi]);


#------------------------------------------------------------
#FUNCTIONS FOR INTERNAL USE IN THIS MODULE ONLY
#------------------------------------------------------------
def _vec_abs(vector):
    absval = 0;

    for i in range(len(vector)):
        absval = absval+vector[i]**2;

    #take square root
    absval = Numeric.sqrt(absval);

    return absval;

def _vec_scal_prod(vector1,vector2):
    prod = 0;

    for i in range(len(vector1)):
        prod = prod + vector1[i]*vector2[i];    

    return prod;

def _vec_ang(vector1,vector2):
    deg2rad = Numeric.pi/180;

    vec1_abs = _vec_abs(vector1);
    vec2_abs = _vec_abs(vector2);

    vec_prod = _vec_scal_prod(vector1,vector2);

    print vec_prod/vec1_abs/vec2_abs
    try:
    	ang = Numeric.arccos(vec_prod/vec1_abs/vec2_abs);
    except:
    	ang = 0.0;

    return(ang);
