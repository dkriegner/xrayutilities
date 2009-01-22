#a module for handling data of a 1d detector

import Numeric

class Detector:
    name = "";
    dim  = 0;
    channels = [];
    ccs = [];
    ch_degs = [];
    directions = [];

    def __init__(self,name,dim,chan,ccs,ch_degs,directions):
        """
        The Detector object provides a class for handling detectors.
        The initial parameters are:
        name .......... a name of the detector object
        dim ........... number of dimension of the detector
        channels ...... a sequence with the number of channels per dimension
        ccs ........... a sequence with the center channels per dimension
        ch_degs ....... a sequence with the channels per degree
        directions .... a sequence with the directions per dimension
        """
        self.name = name;
        self.dim  = dim;
        self.channels = chan;
        self.ccs = ccs;
        self.ch_degs = ch_degs;
        self.directions = directions;
        

    def get_angles(self,angs):
        """
        This method returns the angular values for every detector point.
        It returns a list with the dimension of the detector where every
        entry in the list is a matrix whos shape is determined by the
        channel numbers of the detector. The matrices contain the angular
        data for every detector dimension.
        """
        ang_list = [];
        ang_array = [];

        #check if the angular list is ok
        if len(angs)!=self.dim:
            print 'number of angles doesn\'t fit the detector dimension'
            return None

        for i in range(self.dim):
            ang_list.append(Numeric.zeros(self.channels,Numeric.Float));

        for i in range(self.dim):
            ang_array.append(angs[i]+self.directions[i]*\
            (range(self.channels[i])-self.ccs[i])/self.ch_degs[i])

        #have to build the matrices
        
            

    def realing_data(self,data):
        """
        The realing data methods provides a way of linearising the
        data from a more dimensional detector.
        If one for instance has an 2 dimensional detector with m x n
        channels it will take the data as an m x n array and convert it
        to an array of shape n*m x 1.
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

        
        

        

#create a dictionary holding all the data for a 1d detector (a Braun PSD for instance)
#a single optional argument can be given:
#roi = [start,stop] defining a region of interest
def createdet_1d(name,description,nof_chans,cc,ch_deg,dir,par_axis,**opt_args):
    det_dict = {'Name':name,
                'description':description,
                'channels':nof_chans,
                'cc':float(cc),
                'ch_deg':float(ch_deg),                
                'direction':float(dir),
                'roi':[0,nof_chans-1],
                'grid volume': nof_chans,
                'parallel gon. axis':par_axis};
    
    if opt_args.has_key('roi'):
        det_dict['roi'];
    
    return(det_dict);

#calulcates the axis for a single center position of the detector
#this function returns a single value if center_value is a single value
def get_grid_det_1d(center_value,det_dict,**opt_args):    

    if isinstance(center_value,float):
        #if the center value is a single float value
        nof_points = 1;
        cen_val = Numeric.array([center_value],Numeric.Float);
        cen_val = Numeric.reshape(cen_val,(1,1));
    else:
        #otherwise center_values is assumed to be a numeric array of shape
        #(n,1)
        nof_points = center_value.shape[0];

    #check if a roi is given as an optional option:
    if opt_args.has_key('uroi'):
        roi_start = det_dict['roi'][0]+opt_args['uroi'][0];
        roi_stop  = roi_start+opt_args['uroi'][1];
        eff_channels = abs(roi_stop-roi_start);
    else:
        roi_start = det_dict['roi'][0];
        roi_stop  = det_dict['roi'][1];
        eff_channels = det_dict['channels'];
    #from this it follows that the roi options overrides the detectors own roi setup

    #generate the data grid
    grid_data = Numeric.zeros((nof_points*(roi_stop-roi_start+1),1),Numeric.Float);

    #iterate over all numbers of points
    for i in range(nof_points):
        #loop over all detector channels:
        for j in range(roi_start,roi_stop+1):
            grid_data[j+i*eff_channels,0]=center_value[i,0]+det_dict['direction']* \
                                           (j-det_dict['cc'])/det_dict['ch_deg']

    return grid_data;

    
def real_data(data_array,det_dict,**opt_args):
    #realign data according to the detector object

    nof_points = data_array.shape[0];
    roi_start = 0;
    roi_stop  = 0;
    eff_channels = 0;

    #check if roi is set
    if opt_args.has_key('roi'):
        #if the roi option is set:
        roi_start = opt_args['uroi'][0];
        roi_stop  = opt_args['uroi'][1];
        eff_channels = roi_stop - roi_start+1;
    else:
        roi_start = det_dict['roi'][0];
        roi_stop  = det_dict['roi'][1];
        eff_channels = det_dict['channels'];

    #check the dimension of the data_array:
    if data_array.shape[1]>1:
        #in this case data is assumed to be a matrix with detector data

        #check is a user defined roi is set:
        if roi_start != 0:
            #cut the roi region from the original data
            data_buffer = data_array[:,roi_start:roi_stop];
        else:
            data_buffer = data_array;        

        #reshape the data matrix of the detector    
        new_data = Numeric.reshape(data_buffer,(nof_points*eff_channels,1))
        
    elif data_array.shape[1]==1:
        #data is assumed to be some arbitrary value

        #generate an output array:
        new_data = Numeric.zeros((nof_points*eff_channels,1),Numeric.Float);
        
        for i in range(nof_points):
            new_data[i*eff_channels:(i+1)*eff_channels,0]=data_array[i,0]

    return new_data;
            
        
