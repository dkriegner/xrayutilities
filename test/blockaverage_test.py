#!/usr/bin/env python
import xrayutilities as xu
import numpy

a = numpy.linspace(0,10,11)
a = numpy.require(a,dtype=numpy.double,requirements=['C_CONTIGUOUS'])
b = xu.cxrayutilities.block_average1d(a,2)
c = xu.blockAverage1D(a,2)
print(b)
print(c)

a = numpy.linspace(1,10,10)[numpy.newaxis,:]*numpy.ones((20,10))
print a.shape
print a
a = numpy.require(a,dtype=numpy.double,requirements=['C_CONTIGUOUS'])
b = xu.cxrayutilities.block_average2d(a,5,2)
c = xu.blockAverage2D(a,5,2)
print(b)
print(c)
print b.shape

a = numpy.linspace(1,10,10)[numpy.newaxis,:]*numpy.ones((10,10))
print a.shape
print a
a = numpy.require(a,dtype=numpy.double,requirements=['C_CONTIGUOUS'])
b = xu.cxrayutilities.block_average_PSD(a,3)
c = xu.blockAveragePSD(a,3)
print(b)
print(c)
print b.shape
