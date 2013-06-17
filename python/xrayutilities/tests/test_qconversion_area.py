import xrayutilities as xu
import numpy
import unittest

class TestQConversion(unittest.TestCase):

    def setUp(self):
        self.mat = xu.materials.Si
        self.hxrd = xu.HXRD(self.mat.Q(1,1,0),self.mat.Q(0,0,1))
        self.nch = (9,13)
        self.hxrd.Ang2Q.init_area('z+','x+',4,6,self.nch[0],self.nch[1],1.0,50e-6,50e-6)
        self.hklsym = (0,0,4)
        self.hklasym = (2,2,4)

    def test_qconversion_area(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        qout = self.hxrd.Ang2HKL(ang[0],ang[3],mat=self.mat,dettype='area')
        self.assertEqual(qout[0].shape, self.nch)
        for i in range(3):
            q = qout[i]
            self.assertAlmostEqual(numpy.average(q), self.hklsym[i], places=6) 

    def test_qconversion_area_asym(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklasym))
        qout = self.hxrd.Ang2HKL(ang[0],ang[3],mat=self.mat,dettype='area')
        self.assertEqual(qout[0].shape, self.nch)
        for i in range(3):
            q = qout[i]
            self.assertAlmostEqual(numpy.average(q), self.hklasym[i], places=6)

if __name__ == '__main__':
        unittest.main()
