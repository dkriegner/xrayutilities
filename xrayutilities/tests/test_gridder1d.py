import xrayutilities as xu
import numpy
import unittest

class TestGridder1D(unittest.TestCase):

    def setUp(self):
        self.num = 10
        self.xmin = 1
        self.xmax = 10
        self.x = numpy.linspace(self.xmin,self.xmax,num=self.num)
        self.data = numpy.random.rand(self.num)
        self.gridder = xu.Gridder1D(self.num)
        self.gridder(self.x,self.data)

    def test_gridder1d_axis(self):
        hist,bins = numpy.histogram(self.x,bins=self.num)
        x = (bins[0:-1]+bins[1:])/2.
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.num) 
        # test values of xaxis
        for i in range(self.num):
            self.assertAlmostEqual(self.gridder.xaxis[i], x[i], places=12)
    
    def test_gridder1d_data(self):
        # test length of data
        self.assertEqual(len(self.gridder.data), self.num) 
        # test values of data
        for i in range(self.num):
            self.assertAlmostEqual(self.gridder.data[i], self.data[i], places=12)

if __name__ == '__main__':
        unittest.main()
