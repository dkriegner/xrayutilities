import xrutils as xu
import numpy
import matplotlib.pyplot as plt

# defining material and experimental setup
InAs = xu.materials.InAs
energy= numpy.linspace(500,20000,5000) # 500 - 20000 eV

F = InAs.StructureFactorForEnergy(InAs.Q(1,1,1),energy)

plt.figure(); plt.clf()
plt.plot(energy,F.real,'k-',label='Re(F)')
plt.plot(energy,F.imag,'r-',label='Imag(F)')
plt.xlabel("Energy (eV)"); plt.ylabel("F"); plt.legend()
