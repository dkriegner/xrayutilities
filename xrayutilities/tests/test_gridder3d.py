import xrayutilities as xu
import numpy
import unittest

class TestGridder3D(unittest.TestCase):

    def setUp(self):
        self.nx = 10
        self.ny = 19 # do not change this here unless you fix also the tests cases
        self.nz = 10
        self.xmin = 1
        self.xmax = 10
        self.x = numpy.linspace(self.xmin,self.xmax,num=self.nx)
        self.y = self.x.copy()
        self.z = self.x.copy()
        self.data = numpy.random.rand(self.nx)
        self.gridder = xu.Gridder3D(self.nx,self.ny,self.nz)
        self.gridder(self.x,self.y,self.z,self.data)

    def test_gridder3d_xaxis(self):
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.nx) 
        # test values of xaxis
        for i in range(self.nx):
            self.assertAlmostEqual(self.gridder.xaxis[i], self.x[i], places=12)
    
    def test_gridder3d_yaxis(self):
        # test length of yaxis
        self.assertEqual(len(self.gridder.yaxis), self.ny) 
        # test end values of yaxis
        self.assertAlmostEqual(self.gridder.yaxis[0], self.y[0], places=12)
        self.assertAlmostEqual(self.gridder.yaxis[-1], self.y[-1], places=12)
        self.assertAlmostEqual(self.gridder.yaxis[1] - self.gridder.yaxis[0], (self.xmax-self.xmin)/float(self.ny-1) , places=12)
    
    def test_gridder3d_zaxis(self):
        # test length of yaxis
        self.assertEqual(len(self.gridder.zaxis), self.nz) 
        # test end values of yaxis
        self.assertAlmostEqual(self.gridder.zaxis[0], self.z[0], places=12)
        self.assertAlmostEqual(self.gridder.zaxis[-1], self.z[-1], places=12)
        self.assertAlmostEqual(self.gridder.zaxis[1] - self.gridder.zaxis[0], (self.xmax-self.xmin)/float(self.nz-1) , places=12)
    
    def test_gridder3d_data(self):
        # test shape of data
        self.assertEqual(self.gridder.gdata.shape[0], self.nx) 
        self.assertEqual(self.gridder.gdata.shape[1], self.ny) 
        # test values of data
        aj,ak,al = numpy.indices((self.nx,self.ny,self.nz))
        aj,ak,al = numpy.ravel(aj),numpy.ravel(ak),numpy.ravel(al) 
        for i in range(self.nx*self.ny*self.nz):
            j,k,l = (aj[i],ak[i],al[i])
            if k==2*j and l==j: self.assertAlmostEqual(self.gridder.gdata[j,k,l], self.data[j], places=12)
            else: self.assertEqual(self.gridder.gdata[j,k,l], 0.)

if __name__ == '__main__':
        unittest.main()
