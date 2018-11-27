import pyPLUTO as pp
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

filename = 'maxwell'
os.mkdir(filename)
wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
n = 150
offset=0
for i in range(n):
    j = offset+i
    fileObject = open('/data/hslr2/' + filename +'/'+ filename + '_0%i' %j,'wb')
    D = pp.pload(j,w_dir=wdir)
    pickle.dump(D.bx1[24:352,124:208,:]*D.bx3[24:352,124:208,:],fileObject)
    fileObject.close()