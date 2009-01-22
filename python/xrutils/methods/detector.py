#a module for handling data of a 1d detector

import Numeric
import numpy

class Detector(object):
    def __init__(self):
        self.filters = 0

    def SetFilterValues(self,f):
        if isinstance(f,list):
            self.filters = numpy.array(f,dtype=numpy.double)
        elif isinstance(f,numpy.ndarray):
            self.filters = f
        else:
            raise TypeError,"Filter values must be a list or an ndarray!"


class PSD:
    name = "";
    dim  = 0;
    dim_names = [];
    channels = [];
    ccs = [];
    ch_degs = [];
    directions = [];

    def __init__(self,name,dim,dim_names,chan,ccs,ch_degs,directions):
        """
        The Detector object provides a class for handling detectors.
        The initial parameters are:
        name .......... a name of the detector object
        dim ........... number of dimension of the detector
        dimnames ...... angular names the dimensions are parallel to
        channels ...... a sequence with the number of channels per dimension
        ccs ........... a sequence with the center channels per dimension
        ch_degs ....... a sequence with the channels per degree
        directions .... a sequence with the directions per dimension
                        1.0 ..... higher channels for higher angles
                       -1.0 ..... lower channels for higher angles 
        """
        self.name = name;
        self.dim  = dim;
        self.dim_names = dim_names;
        self.channels = chan;
        self.ccs = ccs;
        self.ch_degs = ch_degs;
        self.directions = directions;
        

    def get_angles(self,angs,dims):
        """
        This method returns the angular values for every detector point.
        It returns a list with the dimension of the detector where every
        entry in the list is a matrix whos shape is determined by the
        channel numbers of the detector. The matrices contain the angular
        data for every detector dimension.
        """
        ang_list = [];
        ang_axis = [];
        index_list = [];

        #check if the angular list is ok
        if len(angs)!=self.dim:
            print 'number of angles doesn\'t fit the detector dimension'
            return None

        for i in range(self.dim):
            ang_list.append(Numeric.ones(self.channels,Numeric.Float));        
            ang_axis = Numeric.arrayrange(0,self.channels[i]);            
            ang_axis = ang_axis.astype(Numeric.Float);
            #build the index list
            for j in range(self.dim):
                if j==i:
                    index_list.append(self.channels[i]);
                else:
                    index_list.append(1);

            #now reshape the angular axis:
            ang_axis = Numeric.reshape(ang_axis,index_list);
            ang_list[i] = ang_list[i]*ang_axis;
            ang_list[i] = angs[i]+self.directions[i]*\
                          (ang_list[i]-self.ccs[i])/self.ch_degs[i]
            index_list = []
            
        return Numeric.array(ang_list).astype(Numeric.Float)
    
                    

    def realing_data(self,data):
        """
        The realing data methods provides a way of linearising the
        data from a more dimensional detector.
        If one for instance has an 2 dimensional detector with m x n
        channels it will take the data as an m x n array and convert it
        to an array of shape n*m x 1. This method is usefull if you have
        for instance a mxn array with angular values and one needs a
        n*m x 1 array of these data.
        """
        nofp = 1;

        #determine the total number of points
        for i in range(self.dim):
            nofp = nofp*data.shape[i];

        #reshape the array
        new_data = Numeric.zeros(data.shape,Numeric.Float);
        new_data = data;

        new_data = Numeric.reshape(new_data,(nofp,1));

        return new_data;
        
class CCD(Detector):
    def __init__(self,nx,ny,cx,cy,cdx,cdy):
        self.nx = nx
        self.ny = ny
        self.cenx = cx
        self.ceny = cy
        self.cdegx = cdx
        self.cdegy = cdy



            
        
