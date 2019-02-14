#loads files with name 'filename' from start to end
import cPickle as pickle
import numpy as np
#from numba import jit
#
#@jit(nopython=True)
def load_quantity(filename,start,end):
    with open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %start, 'rb') as fileObject:
        A = pickle.load(fileObject)[np.newaxis]
    
    
    for j in range(start+1,end+1):
        with open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %j, 'rb') as fileObject:
            B = pickle.load(fileObject)
        
        A = np.concatenate((A,B[np.newaxis]),axis=0)
        
    return np.transpose(A,axes=(1,2,0))