#module with vector operations

import numpy

from .. import config

def VecNorm(v):
    """
    VecNorm(v):
    Calculate the norm of a vector.

    required input arguments:
    v .......... vector as list or numpy array

    return value:
    float holding the vector norm
    """
    #{{{1
    if isinstance(v,list):
        vtmp = numpy.array(v,dtype=numpy.double)
    elif isinstance(v,numpy.ndarray):
        vtmp = v.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")

    return numpy.sqrt((vtmp**2).sum())
    #}}}1

def VecUnit(v):
    """
    VecUnit(v):
    Calculate the unit vector of v.

    required input arguments:
    v ........... vector as list or numpy array

    return value:
    numpy array with the unit vector
    """
    #{{{1
    if isinstance(v,list):
        vtmp = numpy.array(v,dtype=numpy.double)
    elif isinstance(v,numpy.ndarray):
        vtmp = v.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy arra")

    return vtmp/VecNorm(vtmp)
    #}}}1

def VecDot(v1,v2):
    """
    VecDot(v1,v2):
    Calculate the vector dot product.

    required input arguments:
    v1 .............. vector as numpy array or list
    v2 .............. vector as numpy array or list

    return value:
    float value 
    """
    #{{{1
    if isinstance(v1,list):
        v1tmp = numpy.array(v1,dtype=numpy.double)
    elif isinstance(v1,numpy.ndarray):
        v1tmp = v1.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")
    
    if isinstance(v2,list):
        v2tmp = numpy.array(v2,dtype=numpy.double)
    elif isinstance(v2,numpy.ndarray):
        v2tmp = v2.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")

    return (v1tmp*v2tmp).sum()
    #}}}1


def VecAngle(v1,v2,deg=False):
    """
    VecAngle(v1,v2,deg=false):
    calculate the angle between two vectors. The following
    formula is used
    v1.v2 = |v1||v2|cos(alpha)

    alpha = acos((v1.v2)/|v1|/|v2|)

    required input arguments:
    v1 .............. vector as numpy array or list
    v2 .............. vector as numpy array or list

    optional keyword arguments:
    deg ............. (default: false) return result in degree
                      otherwise in radiants

    return value:
    float value with the angle inclined by the two vectors
    """
    #{{{1
    u1 = VecNorm(v1)
    u2 = VecNorm(v2)
    if(config.VERBOSITY >= config.DEBUG):
        print("XU.math.VecAngle: norm of the vectors: %8.5g %8.5g" %(u1,u2)) 

    alpha = numpy.arccos(VecDot(v1,v2)/u1/u2)
    if deg:
        alpha = 180.*alpha/numpy.pi

    return alpha
    #}}}1

