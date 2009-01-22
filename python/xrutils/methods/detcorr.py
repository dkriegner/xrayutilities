import Numeric

def detrowcorr(mca,**keyargs):
    """
    detrowcorr(mca):
    Correct the rows of a detector matrix. If the sum of 
    a row is significantly lower than the sourounding rows
    the row is replaced by an average of the sourounding rows.

    required arguments:
    mca ............... the detector matrix

    optional keyword arguments:
    threshold ......... default is 10% -> if the row sum is 10% lower than the sourounding.
    range ............. range of rows to be corrected
    """
    
    for i in range(mca.shape[0]):
	#loop over all rows


def detcolcorr(mca):
    pass    
