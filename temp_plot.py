import pyPLUTO as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from func.load_quantity import load_quantity
from func.load_coords import load_coords

a = 0.004335413952759297
#b = 1.7939330580529451
h=1; k=1;

r_min = 24; r_max = 616;
t_min = 39; t_max = 209;

isigma = 60/(np.pi**2)


dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25


# wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

x1,x2,x3,A,B = load_coords(wdir)
q_rp = load_quantity('maxwell_stress_rp',239,239) #start-end inclusive

# =============================================================================
# UNCOMMENT TO LOAD
# =============================================================================
#x1,x1_log,x2,x3,A,B = load_coords(wdir)
#dT = pickl('dT_extended_50')
def plot(dT_rp,n):
    
    dT_r = np.mean(abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25,axis=1)

    for i in range(n):
        plt.plot(x1_log,dT_r[:,i],linewidth=0.4)
    
    plt.plot(x1_log,np.mean(dT_r,axis=(1)),linewidth=1,linestyle='--')
    
    # =============================================================================
    # POLYFIT
    # =============================================================================
    l,m=np.polyfit(x1_log,np.mean(dT_r,axis=1),1)
    plt.plot(x1_log,l*x1_log+m)
    
    plt.show()
    return l

power = plot(q_rp,18)

