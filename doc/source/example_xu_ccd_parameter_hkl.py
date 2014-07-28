"""
example script to show the detector parameter determination for area detectors from images recorded in the primary beam
and at known symmetric coplanar Bragg reflections of a reference crystal
"""

import xrayutilities as xu
import os
import numpy

Si = xu.materials.Si

datadir = 'data'
specfile = "si_align.spec"

en=15000 #eV
wl = xu.en2lam(en)
imgdir = os.path.join(datadir,"si_align_") # data path for CCD files
filetmp = "si_align_12_%04d.edf.gz"

qconv = xu.QConversion(['z+','y-'],['z+','y-'],[1,0,0]) 
hxrd = xu.HXRD(Si.Q(1,1,-2),Si.Q(1,1,1),wl=wl,qconv=qconv)

## manually selected images

s = xu.io.SPECFile(specfile,path=datadir)
for num in [61,62,63,20,21,26,27,28]:
    s[num].ReadData()
    try:
        imagenrs = numpy.append(imagenrs,s[num].data['ccd_n'])
    except:
        imagenrs = s[num].data['ccd_n']

# avoid images which do not have to full beam on the detector as well as other which show signal due to cosmic radiation
avoid_images = [37,57,62,63,65,87,99,106,110,111,126,130,175,181,183,185,204,206,207,208,211,212,233,237,261,275,290]

images = []
ang1 = [] # outer detector angle
ang2 = [] # inner detector angle
sang = [] # sample rocking angle
hkls = [] # Miller indices of the reference reflections

def hotpixelkill(ccd):
    """
    function to remove hot pixels from CCD frames
    ADD REMOVE VALUES IF NEEDED!
    """
    ccd[304,97] = 0
    ccd[303,96] = 0
    return ccd

# read images and angular positions from the data file
# this might differ for data taken at different beamlines since
# they way how motor positions are stored is not always consistent
for imgnr in numpy.sort(list(set(imagenrs)-set(avoid_images))[::4]):
    filename = os.path.join(imgdir,filetmp%imgnr)
    edf = xu.io.EDFFile(filename)
    ccd = hotpixelkill(edf.data)
    images.append(ccd)
    ang1.append(float(edf.header['motor_pos'].split()[4]))
    ang2.append(float(edf.header['motor_pos'].split()[3]))
    sang.append(float(edf.header['motor_pos'].split()[1]))
    if imgnr > 1293.:
        hkls.append((0,0,0))
    elif imgnr < 139: 
        hkls.append((0,0,numpy.sqrt(27))) #(3,3,3))
    else: 
        hkls.append((0,0,numpy.sqrt(75))) #(5,5,5))

# call the fit for the detector parameters 
# detector arm rotations and primary beam direction need to be given 
# in total 8 detector parameters + 2 additional parameters for the reference crystal orientation and the wavelength are fitted, 
# however the 4 misalignment parameters of the detector and the 3 other parameters can be fixed
# the fixable parameters are detector tilt azimuth, the detector tilt angle, the detector rotation around the primary beam, the outer angle offset
# sample tilt, sample tilt azimuth and the x-ray wavelength
param,eps = xu.analysis.area_detector_calib_hkl(sang,ang1,ang2,images,hkls,hxrd,Si,['z+','y-'],'x+',start=(45,1.69,-0.55,-1.0,1.3,60.,wl),fix=(False,False,False,False,False,False,False),plot=True)

# Following is an example of the output of the summary of the area_detector_calib_hkl function
#total time needed for fit: 624.51sec
#fitted parameters: epsilon: 9.9159e-08 (2,['Parameter convergence']) 
#param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset,sampletilt,stazimuth,wavelength)
#param: 367.12 349.27 6.8187e-05 6.8405e-05 131.4 2.87 -0.390 -0.061 1.201 318.44 0.8254
#please check the resulting data (consider setting plot=True)
#detector rotation axis / primary beam direction (given by user): ['z+', 'y-'] / x+
#detector pixel directions / distance: z- y+ / 1
#	detector initialization with: init_area('z-','y+',cch1=367.12,cch2=349.27,Nch1=516,Nch2=516, pwidth1=6.8187e-05,pwidth2=6.8405e-05,distance=1.,detrot=-0.390,tiltazimuth=131.4,tilt=2.867)
#AND ALWAYS USE an (additional) OFFSET of -0.0611deg in the OUTER DETECTOR ANGLE!

#param,eps = xu.analysis.area_detector_calib(ang1,ang2,images,['z+','y-'],'x+',start=(45,0,0,0),fix=(False,False,False,False),plot=True,wl=wl)

