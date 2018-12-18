import pyPLUTO as pp
import numpy as np
import pickle
import os

a = 230.65848172665787
b = -389.74685109951344

n_r = 328
n_th = 84
n_phi = 128

d_th = 1.0/n_th
d_phi = 1.0/n_phi

wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'

def load_coords(wdir):
    D = pp.pload(0,w_dir=wdir)
    x1 = D.x1[24:352]
    x2 = D.x2[124:208]
    x3 = D.x3
    return x1,x2,x3

def load(i,wdir):

    offset=0;
#    fileObject = open(wdir + 'data.' + '%04d' %(i+offset),'wb')
    D = pp.pload(i+offset,w_dir=wdir)
    q = D.bx1[24:352,124:208,:]*D.bx3[24:352,124:208,:]
    return q

#FAST 192ms -> vectorised
def integrate(q):
    q2 = q.sum(2)
    dL = np.zeros((328,84))
    dS = np.zeros(328)
    for n_r in range(293):
        R = ( a * np.log( x1[n_r] / np.sin(x2) ) + b ).astype(int)
        dz = x1[n_r] * (np.sin(x2)**-2.0)*d_th
        dL[R,:] = q2[R,:]*dz
        
        A = (1.0-2.0/x1[n_r])
        B = (1.0-3.0/x1[n_r])
        dS = dL.sum(1) * -1.5 * (A / B) * x1[n_r] ** (-3.0/2.0)
        
    dV = np.sum(dS)
    return dV

x1,x2,x3 = load_coords(wdir)

#filename = 'maxwell2'
#os.mkdir(filename)
n=5

L = np.zeros(n)
for i in range(n):
    q = load(i,wdir)
    L[i] = integrate(q)

fileObject = open('/data/hslr2/lum','wb')
pickle.dump(L,fileObject)
fileObject.close()
