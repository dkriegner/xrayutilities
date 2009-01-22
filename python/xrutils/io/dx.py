#a module to write OpenDX input files in an object oriented approach

class DXFile(object):
    """
    Class DXFile:
        This class is the basement for the dx module. It represents a
        complete DX file and acts as a container for all other
        objects residing within this files. I.g. this are fields.

    Required input arguments for the constructor:
        filename ............... name of the DX file

    Optional keyword arguments:
        seq .................... yes or no, this flag determines
                                 wether a file contains a sequence or not.
	path ................... path where to store the file.
        
    """

    def __init__(self,filename,**keyargs):

        #parse the keyword arguments:
        if keyargs.has_key("seq"):
            #file contains a sequence
            self.seq_flag = 1;
        else:
            self.seq_flag = 0;

        if keyargs.has_key("path"):
            if keyargs["path"][-1]=='/':
                self.DXFilePath = keyargs["path"]
            else:
                self.DXFilePath = keyargs["path"]+"/";
        else:
            self.DXFilePath = "./";

        self.DXFileName = filename;

        #try to open the file
        try:
            self.DXFile_FID = open(self.DXFilePath+self.DXFileName);
        except:
            print "Error opening file: %s" &(self.DXFilePath+self.DXFileName);

        

    def __str__(self):
        pass

class DXGrid(object):

    def __init__(self,**keyargs):
        pass

class DXConnections(object):

    def __init__(self,**keyargs):
        pass

class DXData(object):
    def __init__(self,**keyargs):
        pass
    
