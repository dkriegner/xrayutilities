"""
Panalytical XML (www.XRDML.com) data file parser

based on the native python xml.dom.minidom module. 
want to keep the number of dependancies as small as possible
"""

from xml.dom import minidom
import numpy

class XRDMLMeasurement(object):
    """
    class to handle scans in a XRDML datafile
    """
    def __init__(self,measurement):
        """
        initialization routine for a XRDML measurement which parses are all
        scans within this measurement.
        """

        slist = measurement.getElementsByTagName("scan") # get scans in <xrdMeasurement>

        self.ddict = {}
        is_scalar = 0
        
        #loop over all scan entries - scan points
        for s in slist:
            points = s.getElementsByTagName("dataPoints")[0]
            
            # add count time to output data
            countTime = points.getElementsByTagName("commonCountingTime")[0].childNodes[0].nodeValue
            if not self.ddict.has_key("countTime"): 
                self.ddict["countTime"] = [] 
            self.ddict["countTime"].append(float(countTime))

            #check for intensities first to get number of points in scan
            det = points.getElementsByTagName("intensities")[0]
            data = det.childNodes[0]
            # count time normalization; output is counts/sec
            data_list = (numpy.fromstring(data.nodeValue,sep=" ")/float(countTime)).tolist()
            nofpoints = len(data_list)
            if not self.ddict.has_key("detector"): 
                self.ddict["detector"] = [] 
            self.ddict["detector"].append(data_list)
            # if present read beamAttenuationFactors
            # they are already corrected in the data file, but may be interesting
            attfact = points.getElementsByTagName("beamAttenuationFactors")
            if len(attfact)!=0:
                data = attfact[0].childNodes[0]
                data_list = numpy.fromstring(data.nodeValue,sep=" ").tolist()
                nofpoints = len(data_list)
                if not self.ddict.has_key("beamAttenuationFactors"): 
                    self.ddict["beamAttenuationFactors"] = [] 
                self.ddict["beamAttenuationFactors"].append(data_list)
            
            #read the axes position
            pos = points.getElementsByTagName("positions")
            for p in pos:
                #read axis name and unit
                aname = p.getAttribute("axis")
                aunit = p.getAttribute("unit")

                #read axis data
                l = p.getElementsByTagName("listPositions")
                s = p.getElementsByTagName("startPosition")
                e = p.getElementsByTagName("endPosition")
                if len(l)!=0: # listPositions
                    l = l[0]
                    data = l.childNodes[0]
                    data_list = numpy.fromstring(data.nodeValue,sep=" ").tolist()
                elif len(s)!=0: # start endPosition
                    data_list = numpy.linspace(float(s[0].childNodes[0].nodeValue),float(e[0].childNodes[0].nodeValue),nofpoints).tolist()
                else: # commonPosition
                    l = p.getElementsByTagName("commonPosition")
                    l = l[0]
                    data = l.childNodes[0]
                    data_list = numpy.fromstring(data.nodeValue,sep=" ").tolist()
                    is_scalar = 1

                #print(data_list)
                #have to append the data to the data dictionary
                if not self.ddict.has_key(aname):
                    self.ddict[aname] = []

                if not is_scalar:
                    self.ddict[aname].append(data_list)
                else:
                    self.ddict[aname].append(data_list[0])
                    is_scalar = 0
                        
        #finally all scan data needs to be converted to numpy arrays
        for k in self.ddict.keys():
            self.ddict[k] = numpy.array(self.ddict[k])

        #flatten output if only one scan was present
        if len(slist) == 1:
            for k in self.ddict.keys():
                self.ddict[k] = numpy.ravel(self.ddict[k])

    def __getitem__(self,key):
        return self.ddict[key]

    def __str__(self):
        ostr = "XRDML Measurement\n"
        for k in self.ddict.keys():
            ostr += "%s with %s points\n" %(k,str(self.ddict[k].shape))

        return ostr


class XRDMLFile(object):
    """
    class to handle XRDML data files. The class is supplied with a file
    name and uses the XRDMLScan class to parse the xrdMeasurement in the
    file
    """
    def __init__(self,fname):
        """
        initialization routine supplied with a filename
        the file is automatically parsed and the data are available
        in the "scan" object. If more <xrdMeasurement> tags are present, which
        should not be the case, their data is present in the "scans" object.

        Parameter
        ---------
         fname:     filename of the XRDML file

        """
        self.filename = fname
        d = minidom.parse(fname)
        root = d.childNodes[0]
            
        slist = root.getElementsByTagName("xrdMeasurement")

        #determine the number of scans in the file
        self.nscans = len(slist)
        self.scans = []
        for s in slist:
            self.scans.append(XRDMLMeasurement(s))

        if self.nscans == 1:
            self.scan = self.scans[0]
    
    def __str__(self):
        ostr = "XRDML File: %s\n" %self.filename
        for s in self.scans:
            ostr += s.__str__()

        return ostr
