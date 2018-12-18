import numpy as np
import pickle
import matplotlib.pyplot as plt

a = 230.65848172665787
b = -389.74685109951344

n_r = 352
n_th = 168
d_th = 1.0/n_th

def load(filename,n):
    q = np.zeros((352,168,128,5))
    file2 = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/coords', 'rb')
    x1,x2,x3 = pickle.load(file2)
    x1 = x1[24:352]
    x2 = x2[40:208]
    wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/'
    offset = 300
    for i in range(n):
        file = open(wdir + filename +'/'+ filename + '_%i' %(i+offset), 'rb')
        q[:,:,:,i] = pickle.load(file)
        
    return q,x1,x2,x3

#SLOW 507ms
def integrate2(n):
    q2 = q.sum(2)
    dL = np.zeros((328,168))
    dS = np.zeros(328)
    for n_r in range(341):
        for n_t in range(168):
            R = int( a * np.log( x1[n_r] / np.sin(x2[n_t]) ) + b )
            dz = x1[n_r] * (1.0/np.sin(x2[n_t])**2.0)*d_th
            dL[R,n_t] = q2[R,n_t,n]*dz
        
        A = (1.0-2.0/x1[n_r])
        B = (1.0-3.0/x1[n_r])
        dS = dL.sum(1) * -1.5 * (A / B) * x1[n_r] ** (-3.0/2.0)
        
    dV = np.sum(dS)
    return dV

#FAST 192ms -> vectorised
def integrate(n):
    q2 = q.sum(2)
    dL = np.zeros((328,168))
    dS = np.zeros(328)
    for n_r in range(317):
        R = ( a * np.log( x1[n_r] / np.sin(x2) ) + b ).astype(int)
        dz = x1[n_r] * (np.sin(x2)**-2.0)*d_th
        dL[R,:] = q2[R,:,n]*dz
        
        A = (1.0-2.0/x1[n_r])
        B = (1.0-3.0/x1[n_r])
        dS = dL.sum(1) * -1.5 * (A / B) * x1[n_r] ** (-3.0/2.0)
        
    dV = np.sum(dS)
    return dV

q,x1,x2,x3 = load('maxwell',5)


#sum_over_r = np.zeros(5)
#sum_over_r2 = np.zeros(5)
#for i in range(5):
#    sum_over_r[i] = integrate(i)
#    sum_over_r2[i] = integrate2(i)
#
#time = np.linspace(300e-4,305*30.7812e-4,5)
#plt.plot(time,sum_over_r/np.average(sum_over_r))
#plt.plot(time,sum_over_r2/np.average(sum_over_r2))
##plt.xticks([0,0.25,0.5,0.75,1])
#plt.xlabel(r'Time $(GM/c^3) \times 10^4$')
#plt.ylabel(r'L')