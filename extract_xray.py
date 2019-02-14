import pyPLUTO as pp
import numpy as np
import os

filename = 'csv_M_dot'
os.mkdir(filename)
wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
#wdir  = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
n = 10
offset=239

def load_coords(wdir):
    D = pp.pload(300,w_dir=wdir)
    return D.x1[24]*(np.sin(D.x2))

A = load_coords(wdir)

M_dot = np.zeros(1024)

for i in range(n):
    j = offset+i
    D = pp.pload(j,w_dir=wdir)
    M_dot[i] = (D.vx1[24]*D.rho[24]*A[:,np.newaxis]).sum((0,1))


np.savetxt('/data/hslr2/' + filename +'/'+ filename + '_0%i' %j,M_dot)