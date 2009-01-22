#an experimental data file

class DataFile(object):
    """
    class DataFile
    A data file object holds
    basic information about a data file that is somewhere
    stored on the file system. However, it knows nothing
    about the real internals of the file (format and so on).
    It stores only information like the path, filename,
    and certain offsets within the file.
    """

    def __init__(self,path,filename):
        if path[-1] == "/":
        	self.path = path;
        else:
        	self.path = path + "/";
        
        self.filename = filename;
        self.fullfilename = self.path + self.filename;
        self.fid = None;
        
	def open(self):
		"""
		open():
		Open the file in the file object.
		"""
		try:
			self.fid = open(self.fullfilename,"r");
		except:
			print "error opening file %s" %(self.fullfilename);
			self.fid = None;
			return None
			
		#after opening the file we have to find EOF offset with 
		#respect to the beginning of the file
			
	def close(self):
		try:
			self.fid.close();
		except:
			self.fid = None;
			return None;
			
			
