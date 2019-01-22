#loads files with name 'filename' from start to end
import pickle
import numpy as np

def load_quantity_reduced(self,filename,start,end):
    fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %start, 'rb')
    A = pickle.load(fileObject)
    A = (abs(A).sum(1) * self.C * self.x1 ** (1.5)).sum()
    fileObject.close()
    
    for j in range(start+1,end+1):
        
        fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'+filename+'/'+filename+'_0%i' %j, 'rb')
        B = pickle.load(fileObject)
        B = (abs(B).sum(1) * self.C * self.x1 ** (1.5)).sum()
        fileObject.close()
        A = np.append(A,B)
        
    return A
