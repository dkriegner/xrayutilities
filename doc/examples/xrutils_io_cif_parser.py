import xrutils as xu
import numpy
#import matplotlib.pyplot as plt

# parse cif file to get unit cell structure
cif_cal = xu.io.CIFFile("cif_files/Calcite.cif")

#create material
Calcite = xu.materials.Material("Calcite",cif_cal.Lattice(),numpy.zeros((6,6),dtype=numpy.double))

#experiment class according to Nina
expcal = xu.HXRD(Calcite.Q(-2,1,9),Calcite.Q(1,-1,4))

powder_cal = xu.Powder(Calcite)
powder_cal.PowderIntensity()
th,inte = powder_cal.Convolute(0.002,0.02)

print(powder_cal)

#plt.figure()
#plt.clf()
#plt.bar(powder_cal.ang*2,powder_cal.data,align='center')
#plt.plot(th*2,inte,'k-',lw=2)
#plt.xlabel("2Theta (deg)")
#plt.ylabel("Intensity")

