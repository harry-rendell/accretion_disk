import pyPLUTO as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt

a = 0.004342463406340986
#b = 1.7939330580529451
h=1; k=1;
isigma = 60/(np.pi**2)


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


#x1,x2,x3 = load_coords(wdir)
#sine = np.sin(x2)
#q = load(0,wdir,300)

A = (1.0-2.0/x1)
B = (1.0-3.0/x1)


def spectrum_q(q,nu):
    q_ = np.tensordot(q,sine,axes=(1,0)).sum(1)
    dT_const = np.ones(328)
#    dT_const = (x1**-0.75)*5
    dT = abs(q_ * (A / B) * (x1 ** -0.5) * isigma)**0.25
    
    B_ = (( (x1**2) * 1.0/(np.exp(np.divide.outer(nu,dT))-1) ).sum(1) )*(nu**3)
    
    B_const = (( (x1**2) * 1.0/(np.exp(np.divide.outer(nu,dT_const))-1) ).sum(1) )*(nu**3)

    return B_,B_const

def spectrum(dT,nu):

    B_ = (( (x1**2) * 1.0/(np.exp(np.divide.outer(nu,dT))-1) ).sum(1) )*(nu**3)    
    B_const = (( (x1**2) * 1.0/(np.exp(np.divide.outer(nu,dT_const))-1) ).sum(1) )*(nu**3)
    return B_,B_const

n = 1024
offset = 239

def plot():
    N = 50
    B_tot2 = []; B_const2 = []
    nu_range= np.linspace(0.1,50,N)
    
    B_tot,B_const = spectrum(q,nu_range)
#    B_const = spectrum(q,nu_range)

    
    
#    plt.plot(nu_range,B_tot[0],nu_range,B_tot[1],nu_range,B_tot[2],nu_range,B_tot_test)
    plt.plot(np.log(nu_range),np.log(B_tot),color = 'r',marker='+')
    plt.plot(np.log(nu_range),np.log(B_const),marker = '.',color='y')
    plt.xlim([-2.5,4])
    plt.ylim([-5,12])
#    plt.plot(np.log(nu),np.log(B_const),color='k')
#    plt.plot(np.log(nu_range),np.log(B_tot_const),marker='.')
    plt.show()

plot()

#fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/output/temp','wb')
#pickle.dump(dT,fileObject)
#fileObject.close()

