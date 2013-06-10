import xrayutilities.cxrayutilities as cu
import numpy

#2D gridder parameters
xmin = 0
xmax = 10
ymin = xmin
ymax = xmax

x = numpy.arange(xmin,xmax,dtype="float64")
y = x
nx = x.size
ny = y.size

data = numpy.random.random_sample(x.size)
out = numpy.zeros((nx,ny),dtype="float64")

cu.gridder2d(x,y,data,nx,ny,xmin,xmax,ymin,ymax,out)
print out
