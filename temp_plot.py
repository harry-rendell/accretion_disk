import pyPLUTO as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt

a = 0.004335413952759297
#b = 1.7939330580529451
h=1; k=1;

r_min = 24; r_max = 616;
t_min = 39; t_max = 209;

isigma = 60/(np.pi**2)




# wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

def load_coords(wdir):
    D = pp.pload(300,w_dir=wdir)
    x1 = D.x1[r_min:r_max]; x1_log = np.log(x1)
    x2 = D.x2[t_min:t_max]
    x3 = D.x3
    A = (1.0-2.0/x1)
    B = (1.0-3.0/x1)
    return x1,x1_log,x2,x3,A,B

def pickl(filename):
    file = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/' + filename, 'rb')
    quantity = pickle.load(file)
    file.close()
    return quantity

# =============================================================================
# UNCOMMENT TO LOAD
# =============================================================================
#x1,x1_log,x2,x3,A,B = load_coords(wdir)
#dT = pickl('dT_extended')
def plot(n):
    
    for i in range(n):
        plt.plot(x1_log,dT[:,i],linewidth=0.4)
    
    plt.plot(x1_log,np.mean(dT,axis=1),linewidth=1,linestyle='--')
    
    # =============================================================================
    # POLYFIT
    # =============================================================================
    l,m=np.polyfit(x1_log,np.mean(dT,axis=1),1)
    plt.plot(x1_log,l*x1_log+m)
    
    plt.show()
    return l

power = plot(10)

