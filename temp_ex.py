import pyPLUTO as pp
import numpy as np
import pickle

isigma = 60/(np.pi**2)

r_min = 24; r_max = 616;
t_min = 39; t_max = 209;

wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
#wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

def load_coords(wdir):
    D = pp.pload(300,w_dir=wdir)
    x1 = D.x1[r_min:r_max]; x1_log = np.log(x1)
    x2 = D.x2[t_min:t_max]
    x3 = D.x3
    A = (1.0-2.0/x1)
    B = (1.0-3.0/x1)
    return x1,x1_log,x2,x3,A,B

def load(i,wdir,offset):
    D = pp.pload(i+offset,w_dir=wdir)
    q = D.bx1[r_min:r_max,t_min:t_max,:]*D.bx3[r_min:r_max,t_min:t_max,:]
    return q

def temperature(q_rtp):
    
#    q_lower = np.tensordot(q[:,:85,:],sine[:85],axes=(1,0)).sum(1)
#    q_upper = np.tensordot(q[:,85:,:],sine[85:],axes=(1,0)).sum(1)
#    dT_r = abs(np.array([ q_lower, q_upper, (q_lower+q_upper)*0.5 ]) * (A / B) * (x1 ** (-0.5)) * isigma)**0.25
    
    q_r = np.tensordot(q_rtp,sine,axes=(1,0)).sum(1)
    dT_r = abs( q_r * (A / B) * (x1 ** (-0.5)) * isigma)**0.25
    return dT_r

x1,x1_log,x2,x3,A,B = load_coords(wdir)
sine = np.sin(x2) 


offset = 239
n = 1024
filename = 'dT_extended'
dT = np.zeros(((r_max-r_min),n))

for i in range(n):
    j = offset+i
    q = load(i,wdir,offset)
    dT[:,i] = temperature(q)
    
fileObject = open('/data/hslr2/' + filename,'wb')
pickle.dump(dT,fileObject)
fileObject.close()