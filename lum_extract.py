import pyPLUTO as pp
import numpy as np
import pickle

#a = 0.004342463406340986
#b = 1.7939330580529451

n_r = 328
n_th = 84
n_phi = 128

d_th = 1.0/n_th
d_phi = 1.0/n_phi

wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
#wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data/'

def load_coords(wdir):
    D = pp.pload(0,w_dir=wdir)
    x1 = D.x1[24:352]
    x2 = D.x2[39:209]
    x3 = D.x3
    return x1,x2,x3

def load(i,wdir,offset):
    D = pp.pload(i+offset,w_dir=wdir)
    q = D.bx1[24:352,39:209,:]*D.bx3[24:352,39:209,:]
    return q

def integrate(q):
    q_lower = np.tensordot(q[:,:85,:],sine[:85],axes=(1,0)).sum(1)
    q_upper = np.tensordot(q[:,85:,:],sine[85:],axes=(1,0)).sum(1)
    
    dV = np.zeros((616,3))
    A = (1.0-2.0/x1)
    B = (1.0-3.0/x1)
    dV = np.array([ q_lower, q_upper, q_lower+q_upper ]) * (A / B) * x1 ** (1.5)
    L = dV.sum(1)
    return L

x1,x2,x3 = load_coords(wdir)

sine = np.sin(x2)

n = 1024
offset = 239

L = np.zeros((3,n))
for i in range(n):
    q = load(i,wdir,offset)
    L[:,i] = integrate(q)

fileObject = open('/data/hslr2/lumin','wb')
#fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data/lumin','wb')
pickle.dump(L,fileObject)
fileObject.close()

