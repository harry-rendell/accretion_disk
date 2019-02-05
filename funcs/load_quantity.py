#loads files with name 'filename' from start to end
import pickle
import numpy as np
#from numba import jit
#
#@jit(nopython=True)
def load_quantity(filename,start,end):
    fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %start, 'rb')
    A = pickle.load(fileObject)[np.newaxis]
    fileObject.close()
    
    for j in range(start+1,end+1):
        fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %j, 'rb')
        B = pickle.load(fileObject)
        fileObject.close()
        A = np.concatenate((A,B[np.newaxis]),axis=0)
        
    return np.transpose(A,axes=(1,2,0))