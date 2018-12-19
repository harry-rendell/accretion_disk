import pyPLUTO as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt

a1 = 0.004335413952759297
a2 = 0.009126342080710765
h=1; k=1;
isigma = 60/(np.pi**2)

#r_min = 24; r_max = 616; r_mid = 352;
#t_min = 39; t_max = 209;

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

#dT = pickl('dT')
#x1,x1_log,x2,x3,A,B = load_coords(wdir)
    
# =============================================================================
# NEED TO INCLUDE AREA OF ANNULUS
# =============================================================================

def spectrum(dT_r,nu,r_min,r_max):
#    B_ = (( (x1**2) * 1.0/(np.exp(np.divide.outer(nu,dT))-1) ).sum(1) )*(nu**3) 
    if r_max > 328:
        logB_inner =  (a1*(x1[r_min:328]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_r[r_min:328]))-1) )).transpose((1,0)) *(nu**3) 
        logB_outer =  (a2*(x1[328:r_max]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_r[328:r_max]))-1) )).transpose((1,0)) *(nu**3) 
        logB = np.log((np.concatenate((logB_inner,logB_outer),axis=0)).sum(0))
    if r_max <= 328:
        logB =  np.log(( (a1*(x1[r_min:r_max]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_r[r_min:r_max]))-1) )).transpose((1,0)) *(nu**3) ).sum(0))
    return logB

def ref_spectrum(T,nu,r_min,r_max):
    dT_const = np.ones(r_max-r_min)*T
    if r_max > 328:
        logB_inner =  (a1*(x1[:328]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_const[:328]))-1) )).transpose((1,0)) *(nu**3) 
        logB_outer =  (a2*(x1[328:r_max]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_const[328:r_max]))-1) )).transpose((1,0)) *(nu**3) 
        logB = np.log((np.concatenate((logB_inner,logB_outer),axis=0)).sum(0))
    if r_max <= 328:
        logB =  np.log(( (a1*(x1[:r_max]**2) * (1.0/(np.exp(np.divide.outer(nu,dT_const[:r_max]))-1) )).transpose((1,0)) *(nu**3) ).sum(0))
    return logB


def plot():
    N = 50
    nu = np.logspace(-3,1,N)
    nu_log = np.log(nu)
    
    for i in range(10):
        plt.plot( nu_log, spectrum(dT[:,i*10],nu,0,164),color = 'b',label = '616',lw=0.4)
        plt.plot( nu_log, spectrum(dT[:,i*10],nu,164,328),color = 'r',label = '327',lw=0.4)
#        plt.plot( nu_log, spectrum(dT[:,i*10],nu,0,328),color = 'k',label = '327',lw=0.4)
#    plt.legend()
    plt.plot(nu_log,ref_spectrum(np.mean(dT,axis=1),nu,0,328),color='k',lw=0.5)
#    plt.xlim([-10,4])
#    plt.ylim([-10000,1000])

    plt.show()

plot()



