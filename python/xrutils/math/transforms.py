import numpy

class Transform(object):
    def __init__(self,matrix):
        self.matrix = matrix
        try:
            self.imatrix = numpy.linalg.inv(matrix)
        except:
            print "matrix cannot be inverted - seems to be singular"
            self.imatrix = None

    def __call__(self,*args):
        olist = []
        for a in args:
            if isinstance(a,list):
                p = numpy.array(a,dtype=numpy.double)
            elif isinstance(a,numpy.ndarray):
                p = a
            else:
                raise TypeError,"Argument must be a list or numpy array!"

            #matrix product in pure array notation
            if len(p.shape)==1:
                #argument is a vector
                print "transform a vector ..."
                b = (self.matrix*p[numpy.newaxis,:]).sum(axis=1)
                olist.append(b)
            elif len(p.shape)==2 and p.shape[0]==3 and p.shape[1]==3:
                #argument is a matrix
                print "transform a matrix ..."
                b = numpy.zeros(p.shape,dtype=numpy.double)
                b2 = numpy.zeros(p.shape,dtype=numpy.double)
                for i in range(3):
                    for j in range(3):
                        b[i,j] = (self.matrix[i,:]*p[:,j]).sum()

                #perform multiplication with the inverse matrix
                for i in range(3):
                    for j in range(3):
                        b2[i,j] = (b[i,:]*self.imatrix[:,j]).sum()

                olist.append(b2)

    
        if len(args) == 1:
            return olist[0]
        else:
            return olist
    def __str__(self):
        ostr = ""
        ostr += "Transformation matrix:\n"
        ostr += "%f %f %f\n" %(self.matrix[0,0],self.matrix[0,1],self.matrix[0,2])
        ostr += "%f %f %f\n" %(self.matrix[1,0],self.matrix[1,1],self.matrix[1,2])
        ostr += "%f %f %f\n" %(self.matrix[2,0],self.matrix[2,1],self.matrix[2,2])

        return ostr

def CoordinateTransform(v1,v2,v3):
    """
    CoordinateTransform(v1,v2,v3):
    Create a Transformation object which transforms a point into a new 
    coordinate frame. The new frame is determined by the three vectors
    v1, v2 and v3.

    required input arguments:
    v1 ............. list or numpy array with new base vector 1
    v2 ............. list or numpy array with new base vector 2 
    v2 ............. list or numpy array with new base vector 3

    return value:
    An instance of a Transform class
    """

    if isinstance(v1,list):
        e1 = numpy.array(v1,dtype=numpy.double)
    elif isinstance(v1,numpy.ndarray):
        e1 = v1
    else:
        raise TypeError,"vector must be a list or numpy array"
    
    if isinstance(v2,list):
        e2 = numpy.array(v2,dtype=numpy.double)
    elif isinstance(v2,numpy.ndarray):
        e2 = v2
    else:
        raise TypeError,"vector must be a list or numpy array"
    
    if isinstance(v3,list):
        e3 = numpy.array(v3,dtype=numpy.double)
    elif isinstance(v3,numpy.ndarray):
        e3 = v3
    else:
        raise TypeError,"vector must be a list or numpy array"

    #normalize base vectors
    e1 = e1/numpy.sqrt((e1**2).sum())
    e2 = e2/numpy.sqrt((e2**2).sum())
    e3 = e3/numpy.sqrt((e3**2).sum())

    #assemble the transformation matrix
    m = numpy.array([e1,e2,e3])
    
    return Transform(m)
