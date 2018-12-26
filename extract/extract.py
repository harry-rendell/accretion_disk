import pyPLUTO as pp
import numpy as np
import pickle
import os

# =============================================================================
# DATA EXTRACTION
# =============================================================================

#when run on muon[4,5,6] server, this script starts by loading dbl file =offset, and loads n
#subsequent dbl files.

#the selected quantity is extracted from each dbl file and saved into a new folder as
#its own file.

#ZMaxwell = ZBr*ZBPh
#ZBtot = ZBr + ZBTh + ZBPh
#Zprstot = Zprs + 0.5*ZBtot**2

r_min = 24; r_max = 616;
t_min = 39; t_max = 209;

filename = 'maxwell_stress_rp'
os.mkdir(filename)
wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
n = 150
offset=239

for i in range(n):
    j = offset+i
    fileObject = open('/data/hslr2/' + filename +'/'+ filename + '_0%i' %j,'wb')
    D = pp.pload(j,w_dir=wdir)
    q_rp = np.tensordot( D.bx1[r_min:r_max,t_min:t_max,:]*D.bx3[r_min:r_max,t_min:t_max,:] ,sine,axes=(1,0))/170
    pickle.dump(q_rp,fileObject)
    fileObject.close()