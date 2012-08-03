import xrutils as xu
import numpy

# defining a ZincBlendeLattice with two types of atoms and lattice constant a
def ZincBlendeLattice(aa,ab,a):
    #create lattice base
    lb = xu.materials.LatticeBase()
    lb.append(aa,[0,0,0])
    lb.append(aa,[0.5,0.5,0])
    lb.append(aa,[0.5,0,0.5])
    lb.append(aa,[0,0.5,0.5])
    lb.append(ab,[0.25,0.25,0.25])
    lb.append(ab,[0.75,0.75,0.25])
    lb.append(ab,[0.75,0.25,0.75])
    lb.append(ab,[0.25,0.75,0.75])

    #create lattice vectors
    a1 = [a,0,0]
    a2 = [0,a,0]
    a3 = [0,0,a]

    l = xu.materials.Lattice(a1,a2,a3,base=lb)
    return l

# defining InP, no elastic properties are given,
# helper functions exist to create the (6,6) elastic tensor for cubic materials
InP  = xu.materials.Material("InP",ZincBlendeLattice(xu.materials.elements.In, xu.materials.elements.P,5.8687), numpy.zeros((6,6),dtype=numpy.double))
# InP is of course already included in the xu.materials module
