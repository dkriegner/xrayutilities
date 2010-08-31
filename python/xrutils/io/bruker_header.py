#module for reading Bruker CCD data (written on BW2)

#import usefull modules
import tables
import numpy # replace Numeric; changes not tested
import re

#compile some usefull regular expressions
blank_split = re.compile(r"\S+") #used for splitting a list of numbers seperated by single blanks 
                                  #in a string 
blank_remov = re.compile(r"\s+") #removeing multiple blanks in a string

#table definition for a V11 header table
class V11_header_table(tables.IsDescription):
	FNAME    = tables.StringCol(100) #the frame number of the file belonging to this header
	FORMAT   = tables.IntCol()
	VERSION  = tables.IntCol()
	HDRBLKS  = tables.IntCol()
	TYPE     = tables.StringCol(72)
	SITE     = tables.StringCol(72)
	MODEL    = tables.StringCol(72)
	USER     = tables.StringCol(72)
	SAMPLE   = tables.StringCol(72)
	SETNAME  = tables.StringCol(72)
	RUN      = tables.IntCol()
	SAMPNUM  = tables.IntCol()
	TITLE    = tables.StringCol(72*8)
	NCOUNTS  = tables.IntCol(shape=(1,2))
	NOVERFL  = tables.IntCol(shape=(1,3))
	MINIMUM  = tables.IntCol()
	MAXIMUM  = tables.IntCol()
	NONTIME  = tables.IntCol()
	NLATE    = tables.IntCol()
	FILENAM  = tables.StringCol(72)
	CREATED  = tables.StringCol(72)
	CUMULAT  = tables.FloatCol()
	ELAPSDR  = tables.FloatCol()
	ELAPSDA  = tables.FloatCol()
	OSCILLA  = tables.IntCol()
	NSTEPS   = tables.IntCol()
	RANGE    = tables.FloatCol()
	START    = tables.FloatCol()
	INCREME  = tables.FloatCol()
	NUMBER   = tables.IntCol()
	NFRAMES  = tables.IntCol()
	ANGLES   = tables.FloatCol(shape=(1,4))
	NOVER64  = tables.IntCol(shape=(1,3))
	NPIXELB  = tables.IntCol(shape=(1,2))
	NROWS    = tables.IntCol()
	NCOLS    = tables.IntCol()
	WORDORD  = tables.IntCol()
	LONGORD  = tables.IntCol()
	TARGET   = tables.StringCol(72)
	SOURCEK  = tables.FloatCol()
	SOURCEM  = tables.FloatCol()
	FILTER   = tables.StringCol(72)
	CELL     = tables.FloatCol(shape=(2,6))
	MATRIX   = tables.FloatCol(shape=(2,9))
	LOWTEMP  = tables.IntCol(shape=(1,3))
	ZOOM     = tables.FloatCol(shape=(1,3))
	CENTER   = tables.FloatCol(shape=(1,4))
	DISTANC  = tables.FloatCol(shape=(1,2)) #has to be observed
	TRAILER  = tables.IntCol()
	COMPRES  = tables.StringCol(72)
	LINEAR   = tables.StringCol(72)
	PHD      = tables.FloatCol(shape=(1,2))
	PREAMP   = tables.FloatCol(shape=(1,2)) #has to be observed
	CORRECT  = tables.StringCol(72)
	WARPFIL  = tables.StringCol(72)
	WAVELEN  = tables.FloatCol(shape=(1,4))
	MAXXY    = tables.FloatCol(shape=(1,2))
	AXIS     = tables.IntCol()
	ENDING   = tables.FloatCol(shape=(1,4))
	DETPAR   = tables.FloatCol(shape=(2,6))
	LUT      = tables.StringCol(72)
	DISPLIM  = tables.FloatCol(shape=(1,2))
	PROGRAM  = tables.StringCol(72)
	ROTATE   = tables.IntCol()
	BITMASK  = tables.StringCol(72)
	OCTMASK  = tables.IntCol(shape=(2,8))
	ESDCELL  = tables.FloatCol(shape=(2,6))
	DETTYPE  = tables.StringCol(72)
	NEXP     = tables.IntCol(shape=(1,5))
	CCDPARM  = tables.FloatCol(shape=(1,5))
	CHEM     = tables.StringCol(72)
	MORPH    = tables.StringCol(72)
	CCOLOR   = tables.StringCol(72)
	CSIZE    = tables.StringCol(72)
	DNSMET   = tables.StringCol(72)
	DARK     = tables.StringCol(72)
	AUTORNG  = tables.FloatCol(shape=(1,5))
	ZEROADJ  = tables.FloatCol(shape=(1,4))
	XTRANS   = tables.FloatCol(shape=(1,3))
	HKL_XY   = tables.FloatCol(shape=(1,5))
	AXES2    = tables.FloatCol(shape=(1,4))
	ENDINGS2 = tables.FloatCol(shape=(1,4))
	FILTER2  = tables.FloatCol(shape=(1,2))

def GetIntArray(string):
	"""
	GetIntArray(string):
	extracts a list of integer values from a string and converts
	it to a integer numpy array.
	input arguments:
		string .............. the string
	return value:
		ia .................. a list with integer values
	"""
	string  = blank_remov.sub(' ',string.strip())
	strlist = blank_split.findall(string)
	
	for i in range(len(strlist)):
		strlist[i] = int(strlist[i])
	
	ia = numpy.array(strlist).astype(numpy.int)
	return ia

def GetFloatArray(string):
	"""
	GetFLoatArray(string):
	extracts a list of float values from a string and converts
	it to a float numpy array.
	input arguments:
		string .............. the string
	return value:
		fa .................. a list with integer values
	"""
	string  = blank_remov.sub(' ',string.strip())
	strlist = blank_split.findall(string)
	
	for i in range(len(strlist)):
		strlist[i] = float(strlist[i])
	
	fa = numpy.array(strlist).astype(numpy.float)
	return fa

def GetFloatMatrix(strlist):
	"""
	GetFloatMatrix(strlist)
	Builds a float matrix out of the values from a string list. 
	The matrix is represented by a numpy array of shape (nxm) 
	where n is the number of strings in the list and m is the number 
	of values in the strings (it has to be the same for all strings).
	input arguments:
		strlist .................. list with strings
	return value:
	    fm ....................... matrix with float values
	"""

	al = []

	for string in strlist:
		al.append(GetFloatArray(string))

	n = len(al)
	m = al[0].shape[0]
	fm = numpy.zeros((n,m),dtype=numpy.float)
	for i in range(len(al)):
		fm[i,:] = al[i][:]

	return fm

def GetIntMatrix(strlist):
	"""
	GetIntMatrix(strlist)
	Builds a integer matrix out of the values from a string list. 
	The matrix is represented by a numpy array of shape (nxm) 
	where n is the number of strings in the list and m is the number 
	of values in the strings (it has to be the same for all strings).
	input arguments:
		strlist .................. list with strings
	return value:
	    fi ....................... matrix with integer values
	"""

	al = []

	for string in strlist:
		al.append(GetIntArray(string))

	n = len(al)
	m = al[0].shape[0]
	fi = numpy.zeros((n,m),dtype=numpy.int)
	for i in range(len(al)):
		fi[i,:] = al[i][:]

	return fi

def read_header(fid,h5table,name):
	"""
	read_header(fid,h5table)
	Read the header information of a frame from the CCD file and store it 
	to a HDF5 table. 
	Input arguments:
		fid .................. Python file object to the CCD file
		h5table .............. HDF5 table for the data. 
		name ................. name of the array the header record belongs to
	"""

	hdrblk_counter = 0
	maxblks = 0
	itemname = []
	itemvalue = []

	while True:
		try:    
			#read all the data in two tables (as strings)            
			itemname.append(fid.read(8))
			itemvalue.append(fid.read(72))
	
			if itemname[hdrblk_counter]=="FILTER2:":
				#print "reached end of header"
				#perform some dummy read
				fid.read(80)
				break

			hdrblk_counter = hdrblk_counter + 1
		except:
			print "error reading header data from file"
			return

	#set the variables of the header class
	h5table.row['FNAME'] = name
	h5table.row['FORMAT']  = int(itemvalue[0])
	h5table.row['VERSION'] = int(itemvalue[1])
	h5table.row['HDRBLKS'] = int(itemvalue[2])
	h5table.row['TYPE']    = itemvalue[3]
	h5table.row['SITE']    = itemvalue[4]
	h5table.row['MODEL']   = itemvalue[5]
	h5table.row['USER']    = itemvalue[6]
	h5table.row['SAMPLE']  = itemvalue[7]
	h5table.row['SETNAME'] = itemvalue[8]
	h5table.row['RUN']     = int(itemvalue[9])
	h5table.row['SAMPNUM'] = int(itemvalue[10])
	#read 8 lines of title
	h5table.row['TITLE']   = ""+\
					itemvalue[11]+"\n"+itemvalue[12]+"\n"+itemvalue[13]+"\n"+\
					itemvalue[14]+"\n"+itemvalue[15]+"\n"+itemvalue[16]+"\n"+\
					itemvalue[17]+"\n"+itemvalue[18]
	#set the total number of counts
	h5table.row['NCOUNTS'] = GetIntArray(itemvalue[19])
	h5table.row['NOVERFL'] = GetIntArray(itemvalue[20])
	h5table.row['MINIMUM'] = int(itemvalue[21])
	h5table.row['MAXIMUM'] = int(itemvalue[22])
	h5table.row['NONTIME'] = int(itemvalue[23])
	h5table.row['NLATE']   = int(itemvalue[24])
	h5table.row['FILENAM'] = itemvalue[25].strip()
	h5table.row['CREATED'] = itemvalue[26].strip()
	h5table.row['CUMULAT'] = float(itemvalue[27])
	h5table.row['ELAPSDR'] = float(itemvalue[28])
	h5table.row['ELAPSDA'] = float(itemvalue[29])
	h5table.row['OSCILLA'] = int(itemvalue[30])
	h5table.row['NSTEPS']  = int(itemvalue[31])
	h5table.row['RANGE']   = float(itemvalue[32])
	h5table.row['START']   = float(itemvalue[33])
	h5table.row['INCREME'] = float(itemvalue[34])
	h5table.row['NUMBER']  = int(itemvalue[35])
	h5table.row['NFRAMES'] = int(itemvalue[36])
	h5table.row['ANGLES']  = GetFloatArray(itemvalue[37])
	h5table.row['NOVER64'] = GetIntArray(itemvalue[38])
	h5table.row['NPIXELB'] = GetIntArray(itemvalue[39])
	h5table.row['NROWS']   = int(itemvalue[40])
	h5table.row['NCOLS']   = int(itemvalue[41])
	h5table.row['WORDORD'] = int(itemvalue[42])
	h5table.row['LONGORD'] = int(itemvalue[43])
	h5table.row['TARGET']  = itemvalue[44].strip()
	h5table.row['SOURCEK'] = float(itemvalue[45])
	h5table.row['SOURCEM'] = float(itemvalue[46])
	h5table.row['FILTER']  = itemvalue[47].strip()
	h5table.row['CELL']    = GetFloatArray(itemvalue[48]+itemvalue[49])
	h5table.row['MATRIX']  = GetFloatArray(itemvalue[50]+itemvalue[51])
	h5table.row['LOWTEMP'] = GetIntArray(itemvalue[52])
	h5table.row['ZOOM']    = GetFloatArray(itemvalue[53])
	h5table.row['CENTER']  = GetFloatArray(itemvalue[54])
	h5table.row['DISTANC'] = GetFloatArray(itemvalue[55])
	h5table.row['TRAILER'] = int(itemvalue[56])
	h5table.row['COMPRES'] = itemvalue[57].strip()
	h5table.row['LINEAR']  = itemvalue[58].strip()
	h5table.row['PHD']     = GetFloatArray(itemvalue[59])
	h5table.row['PREAMP']  = GetFloatArray(itemvalue[60])
	h5table.row['CORRECT'] = itemvalue[61].strip()
	h5table.row['WARPFIL'] = itemvalue[62].strip()
	h5table.row['WAVELEN'] = GetFloatArray(itemvalue[63])
	h5table.row['MAXXY']   = GetFloatArray(itemvalue[64])
	h5table.row['AXIS']    = int(itemvalue[65])
	h5table.row['ENDING']  = GetFloatArray(itemvalue[66])
	h5table.row['DETPAR']  = GetFloatArray(itemvalue[67]+itemvalue[68])
	h5table.row['LUT']     = itemvalue[69].strip()
	h5table.row['DISPLIM'] = GetFloatArray(itemvalue[70])
	h5table.row['PROGRAM'] = itemvalue[71].strip()
	h5table.row['ROTATE']  = int(itemvalue[72])
	h5table.row['BITMASK'] = itemvalue[73].strip()
	h5table.row['OCTMASK'] = GetIntArray(itemvalue[74]+itemvalue[75])
	h5table.row['ESDCELL'] = GetFloatArray(itemvalue[76]+itemvalue[77])
	h5table.row['DETTYPE'] = itemvalue[78].strip()
	h5table.row['NEXP']    = GetIntArray(itemvalue[79])
	h5table.row['CCDPARM'] = GetFloatArray(itemvalue[80])
	h5table.row['CHEM']    = itemvalue[81].strip()
	h5table.row['MORPH']   = itemvalue[82].strip()
	h5table.row['CCOLOR']  = itemvalue[83].strip()
	h5table.row['CSIZE']   = itemvalue[84].strip()
	h5table.row['DNSMET']  = itemvalue[85].strip()
	h5table.row['DARK']    = itemvalue[86].strip()
	h5table.row['AUTORNG'] = GetFloatArray(itemvalue[87])
	h5table.row['ZEROADJ'] = GetFloatArray(itemvalue[88])
	h5table.row['XTRANS']  = GetFloatArray(itemvalue[89])
	h5table.row['HKL_XY']  = GetFloatArray(itemvalue[90])
	h5table.row['AXES2']   = GetFloatArray(itemvalue[91])
	h5table.row['ENDINGS2'] = GetFloatArray(itemvalue[92])
	h5table.row['FILTER2']  = GetFloatArray(itemvalue[93])
	#finish the data
	h5table.row.append()
	h5table.flush()
	
	   
