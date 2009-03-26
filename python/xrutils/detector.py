#module handling detectors

import numpy

class Absorber(object):
    def __init__(self,value,factor):
        self.value = value
        self.factor = factor
        self.etime = 1.




def FilterFromReferenceData():
    """
    FilterFromReferenceData(self,value,
    """
    pass

def FilterFromConstant(value,factor):
    pass

class Detector(object):
    def __init__(self,name,description):
        self.name = name
        self.description = description
        self._exp_time = 1.

    def ApplyFilter(self,data):
        pass

    def ApplyMonitor(self,data,mon):
        return data/mon
        

    def ApplyExpTime(self,data):
        return data/self._exp_time
    
    def _set_exp_time(self,t):
        if isinstance(t,str) or isinstance(t,int) or isinstance(t,float):
            self._exp_time = float(t)
        else:
            raise TypeError,"exposure time must be a string, integer or float"

    def _get_exp_time(self):
        return self._exp_time

    exptime = property(_get_exp_time,_set_exp_time)



class PSD(Detector):
    def __init__(self,channels,center,chdeg,dir):
        Detector.__init__(self,"PSD","position sensitive detector")
        self.nchannels = channels
        self.center    = center
        self.chdeg     = chdeg
        self.direction = -1
        self.roi       = [0,channels-1]
        self.filter    = None

    def Channel2Angle(self,cval):
        """
        Channel2Angle(cval):
        This method returns an array of angular values around a 
        certain center value cval according to the detector setup.
        If cval is an array by itself the method returns an array of shape
        (n,nchannels) where n is the number of elements in cval.

        required input arguments:
        cval ................ single float or double array with center value

        return value:
        double array of shape (nchannels) or (ncval,nchannels) 
        """
        pass

    def ApplyMonitor(self,data,mon):
        """
        ApplyMonitor(data,mon):
        Normalizes data to monitor counter values. If the data is a one
        dimensional array and mon a single float value
        """
        if len(data.shape)==1:
            return Detector.ApplyMonitor(data,mon)
        else:
            return data/mon[:,numpy.newaxis]

    def ApplyFilter(self,data,**keyargs):
        pass




  

    
class CCD(Detector):
    def __init__(self,xchans,xcenter,xchdeg,xdir,
                      ychans,ycenter,ychdeg,ydir):
        Detector.__init__(self,"CCD","2D CCD detector")
        self.nxchannels = xchans
        self.xcenter    = xcenter
        self.xchdeg     = xchdeg
        self.xdir       = xdir

        self.nychannels = ychans
        self.ycenter    = ycenter
        self.ychdeg     = ychdeg
        self.ydir       = ydir

        self.xroi = [0,xchans-1]
        self.yroi = [0,ychans-1]

    def Channel2Angle(self,
        


