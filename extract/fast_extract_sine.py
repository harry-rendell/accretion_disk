import pyPLUTO as pp
import numpy as np
import os


r_min = 24; r_max = 616;
t_min = 39; t_max = 209;

filename = 'csv_ms'
os.mkdir(filename)
#wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
wdir  = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
n = 10
offset=239

def load_coords(wdir):
    D = pp.pload(300,w_dir=wdir)
    x2 = D.x2[t_min:t_max]
    return np.sin(x2)

sine = load_coords(wdir)

for i in range(n):
    j = offset+i
    D = pp.pload(j,w_dir=wdir)
    ms = D.bx1[r_min:r_max,t_min:t_max,:]*D.bx3[r_min:r_max,t_min:t_max,:]
    q_rp = np.tensordot(ms,sine,axes=(1,0))
﻿   np.savetxt('/data/hslr2/' + filename +'/'+ filename + '_0%i' %j,q_rp,delimiter = ',')