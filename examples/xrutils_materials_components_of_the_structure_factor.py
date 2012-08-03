# f = f0(|Q|) + f1(en) + j * f2(en)
import xrutils as xu
import numpy

Fe = xu.materials.elements.Fe # iron atom
Q = numpy.array([0,0,1.9],dtype=numpy.double)
en = 10000 # energy in eV

print("Iron (Fe): E: %9.1f eV" % en)
print("f0: %8.4g" % Fe.f0(numpy.linalg.norm(Q)))
print("f1: %8.4g" % Fe.f1(en))
print("f2: %8.4g" % Fe.f2(en))
