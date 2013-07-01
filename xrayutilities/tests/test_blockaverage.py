import xrayutilities as xu
import numpy
import unittest

class TestBlockAverageFunctions(unittest.TestCase):

    def setUp(self):
        self.seq = numpy.random.rand(11)
        self.n = 3
        self.seq2d = numpy.random.rand(10,15)
        self.n2d = (3,4)

    def test_blockav1d(self):
        out = xu.blockAverage1D(self.seq, self.n)
        self.assertEqual(out[0], numpy.average(self.seq[0:self.n]))
        self.assertEqual(out.size, numpy.ceil(self.seq.size/float(self.n)))

    def test_blockav2d(self):
        out = xu.blockAverage2D(self.seq2d, self.n2d[0], self.n2d[1])
        self.assertEqual(out[0,0], numpy.average(self.seq2d[0:self.n2d[0],0:self.n2d[1]]))
        self.assertEqual(out.shape, (numpy.ceil(self.seq2d.shape[0]/float(self.n2d[0])),numpy.ceil(self.seq2d.shape[1]/float(self.n2d[1]))))
        
    def test_blockav_psd(self):
        out = xu.blockAveragePSD(self.seq2d, self.n)
        self.assertEqual(out[0,0], numpy.average(self.seq2d[0,0:self.n]))
        self.assertEqual(out.shape, (self.seq2d.shape[0],numpy.ceil(self.seq2d.shape[1]/float(self.n))))

if __name__ == '__main__':
        unittest.main()
