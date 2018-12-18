import pyPLUTO as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt

a = 0.004342463406340986
#b = 1.7939330580529451
h=1; k=1;

n_r = 328
n_th = 84
n_phi = 128
isigma = 60/(np.pi**2)
d_th = 1.0/n_th
d_phi = 1.0/n_phi



# wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

def load_coords(wdir):
    D = pp.pload(300,w_dir=wdir)
    x1 = D.x1[24:352]
    x2 = D.x2[39:209]
    x3 = D.x3
    return x1,x2,x3

def load(i,wdir,offset):
    D = pp.pload(i+offset,w_dir=wdir)
    q = D.bx1[24:352,39:209,:]*D.bx3[24:352,39:209,:]
    return q

def temperature_multi(q_rtp):
    q_lower = np.tensordot(q[:,:85,:],sine[:85],axes=(1,0)).sum(1)
    q_upper = np.tensordot(q[:,85:,:],sine[85:],axes=(1,0)).sum(1)
    
    dT_r = np.zeros((328,3))
    dT_r = abs(np.array([ q_lower, q_upper, (q_lower+q_upper)*0.5 ]) * (A / B) * (x1 ** (-0.5)) * isigma)**0.25

    return dT_r

def temperature(q_rtp):
    q_r = np.tensordot(q_rtp,sine,axes=(1,0)).sum(1)
    dT_r = abs( q_r * (A / B) * (x1 ** (-0.5)) * isigma)**0.25
    return dT_r

#x1,x2,x3 = load_coords(wdir)
#sine = np.sin(x2)
#q = load(0,wdir,300)

A = (1.0-2.0/x1)
B = (1.0-3.0/x1)

x1_log = np.log(x1)

n = 1024
offset = 239

dT = np.log(temperature(q))

plt.plot(x1_log,dT,linewidth=0.5)

#r34 = x1**(-0.75)*2.8+0.3
#fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/output/temp','wb')
#pickle.dump(dT,fileObject)
#fileObject.close()

