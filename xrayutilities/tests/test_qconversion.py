import xrayutilities as xu
import numpy
import unittest

class TestQConversion(unittest.TestCase):

    def setUp(self):
        self.mat = xu.materials.Si
        self.hxrd = xu.HXRD(self.mat.Q(1,1,0),self.mat.Q(0,0,1))
        self.hklsym = (0,0,4)
        self.hklasym = (2,2,4)

    def test_qconversion_point(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        qout = self.hxrd.Ang2HKL(ang[0],ang[3],mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hklsym[i], places=10)

    def test_qconversion_point_asym(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklasym))
        qout = self.hxrd.Ang2HKL(ang[0],ang[3],mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hklasym[i], places=10)
    
    def test_qconversion_energy(self):
        ang1 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        ang2 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym)/2.)
        qout = self.hxrd.Ang2HKL((ang1[0],ang2[0]),(ang1[3],ang2[3]),en=(self.hxrd.energy,2*self.hxrd.energy),mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i][0], self.hklsym[i], places=10)
            self.assertAlmostEqual(qout[i][1], self.hklsym[i], places=10)


if __name__ == '__main__':
        unittest.main()
