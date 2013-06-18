import xrayutilities as xu
import numpy
import unittest

class TestGridder1D(unittest.TestCase):

    def setUp(self):
        self.nx = 10
        self.ny = 19 # do not change this here unless you fix also the tests cases
        self.xmin = 1
        self.xmax = 10
        self.x = numpy.linspace(self.xmin,self.xmax,num=self.nx)
        self.y = self.x.copy()
        self.data = numpy.random.rand(self.nx)
        self.gridder = xu.Gridder2D(self.nx,self.ny)
        self.gridder(self.x,self.y,self.data)

    def test_gridder2d_xaxis(self):
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.nx) 
        # test values of xaxis
        for i in range(self.nx):
            self.assertAlmostEqual(self.gridder.xaxis[i], self.x[i], places=12)
    
    def test_gridder2d_yaxis(self):
        # test length of yaxis
        self.assertEqual(len(self.gridder.yaxis), self.ny) 
        # test end values of yaxis
        self.assertAlmostEqual(self.gridder.yaxis[0], self.y[0], places=12)
        self.assertAlmostEqual(self.gridder.yaxis[-1], self.y[-1], places=12)
        self.assertAlmostEqual(self.gridder.yaxis[1] - self.gridder.yaxis[0], (self.xmax-self.xmin)/float(self.ny-1) , places=12)
    
    def test_gridder2d_data(self):
        # test shape of data
        self.assertEqual(self.gridder.gdata.shape[0], self.nx) 
        self.assertEqual(self.gridder.gdata.shape[1], self.ny) 
        # test values of data
        for i in range(self.nx):
            self.assertAlmostEqual(self.gridder.gdata[i,2*i], self.data[i], places=12)

if __name__ == '__main__':
        unittest.main()
