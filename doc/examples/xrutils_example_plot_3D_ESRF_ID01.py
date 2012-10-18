import numpy
import xrutils as xu
import matplotlib as mpl
import matplotlib.pyplot as plt
from mayavi import mlab
import os

en=7050.0 #x-ray energy in eV 
home = "data" # data path (root)
datadir = os.path.join(home,"bfo_") # data path for CCD files
specdir = home # location of spec file
cch = [342.5,157.3] #center channel of the Maxipix detector: vertical(del)/horizontal(nu) z-y+
chpdeg = [215.5, 215.5] # channel per degree for the detector
nav = [2,2] # reduce data: number of pixels to average in each detector direction
roi = [0,516,0,516] # region of interest
sample = "bfo" # sample name -> used as spec file name
ccdfiletmp = os.path.join(datadir,"bfo_12_%05d.edf.gz") # template for the CCD file names

qconv = xu.experiment.QConversion(['z+','y-'],['z+','y-'],[1,0,0]) # 2S+2D goniometer (simplified ID01 goniometer, sample mu,phi detector nu,del
# convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)

hxrd = xu.HXRD([1,0,0],[0,0,1],en=en,qconv=qconv)
hxrd.Ang2Q.init_area('z-','y+',cch1=cch[0],cch2=cch[1],Nch1=516,Nch2=516, chpdeg1=chpdeg[0],chpdeg2=chpdeg[1],Nav=nav,roi=roi)

h5file = os.path.join(specdir,sample+".h5")
#read spec file and save to HDF5 (needs to be done only once)
try: s
except NameError: s = xu.io.SPECFile(sample+".spec",path=specdir)
else: s.Update()
s.Save2HDF5(h5file)

# number of points to be used during the gridding
nx, ny, nz = 119,120,121

def hotpixelkill(ccd):
    """
    function to remove hot pixels from CCD frames
    """
    ccd[44,159] = 0
    ccd[45,159] = 0
    ccd[43,160] = 0
    ccd[46,160] = 0
    ccd[44,161] = 0
    ccd[45,161] = 0
    ccd[304,95] = 0
    return ccd

def gridmap(scannr,nx,ny,nz):
    """
    read ccd frames and grid them in reciprocal space
    angular coordinates are taken from the spec file
    """

    [mu,phi,nu,delta],sdata = xu.io.geth5_scan(h5file,scannr,'Mu','Phi','Nu','Delta') # read scan data from HDF5 file
    ccdn = sdata['ccd_n'] # extract CCD frame numbers stored in the data
    
    for idx in range(len(ccdn)):
        i = ccdn[idx]
        # read ccd image from EDF file
        e = xu.io.EDFFile(ccdfiletmp%i)
        ccdraw = e.data
        ccd = hotpixelkill(ccdraw)
        ####
        # here a darkfield correction would be done
        ####
        # reduce data size
        CCD = xu.blockAverage2D(ccd, nav[0],nav[1], roi=roi) 
        
        if i==ccdn[0]: 
            # now the size of the data is known -> create data array
            intensity = numpy.zeros( (len(ccdn),) + CCD.shape )

        intensity[i-ccdn[0],:,:] = CCD
    
    # transform scan angles to reciprocal space coordinates for all detector pixels
    qx,qy,qz = hxrd.Ang2Q.area(mu,phi,nu,delta)

    # convert data to rectangular grid in reciprocal space
    gridder = xu.Gridder3D(nx,ny,nz)
    gridder(qx,qy,qz,intensity)
    
    return gridder.xaxis,gridder.yaxis,gridder.zaxis,gridder.gdata,gridder

qx,qy,qz,gint,gridder = gridmap(SPECSCANNR,nx,ny,nz) # SPECSCANNR NEEDS TO BE INSERTED

# plot 3D map using mlab
QX,QY,QZ = numpy.mgrid[qx.min():qx.max():1j*nx,qy.min():qy.max():1j*ny,qz.min():qz.max():1j*nz]
INT = xu.maplog(gint,5.5,1)

# plot 3D data using Mayavi
mlab.figure()
mlab.contour3d(QX,QY,QZ,INT,contours=15,opacity=0.5)
mlab.colorbar(title="log(int)",orientation="vertical")
mlab.axes(nb_labels=5,xlabel='Qx',ylabel='Qy',zlabel='Qz')
#mlab.close(all=True)

# plot 2D sums using matplotlib
plt.figure()
plt.contourf(qx,qy,xu.maplog(gint.sum(axis=2),5,0).T,50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
#plt.savefig(os.path.join("pics","XY_sum.png"))

plt.figure()
plt.contourf(qx,qz,xu.maplog(gint.sum(axis=1),5,0).T,50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QZ ($1/\AA$)")
plt.colorbar()
#plt.savefig(os.path.join("pics","XZ_sum.png"))

plt.figure()
plt.contourf(qy,qz,xu.maplog(gint.sum(axis=0),5,0).T,50)
plt.xlabel("QY ($1/\AA$)")
plt.ylabel("QZ ($1/\AA$)")
plt.colorbar()
#plt.savefig(os.path.join("pics","YZ_sum.png"))

