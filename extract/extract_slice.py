import pyPLUTO as pp
import numpy as np
import pickle

wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
filename = 'abc'
fileObject = open('/data/hslr2' + filename,'wb')

n = 20;
offset = 300;
theta = 0; #which theta to slice at 
q = np.zeros((616,248,n)) 

for i in range(n):
    D = pp.pload(offset+i,w_dir=wdir)
    q[:,:,i] = D.prs[:,:,theta] #quantity to extract
pickle.dump(q,fileObject)
fileObject.close()