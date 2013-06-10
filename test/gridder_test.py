import xrayutilities.cxrayutilities as cu
import numpy

#2D gridder parameters
nx = 10
ny = 10
xmin = 0
xmax = 10
ymin = 0
ymax = 10

x = numpy.arange(0,10,dtype="float64")
y = x
data = numpy.random.random_sample(x.size)
out = numpy.zeros((nx,ny),dtype="float64")

cu.gridder2d(x,y,data,nx,ny,xmin,xmax,ymin,ymax,out)
print out
