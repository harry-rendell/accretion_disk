import numpy as np
import pickle

a = 230.65848172665787
b = -389.74685109951344

n_r = 328
n_th = 84
d_th = 1.0/n_th

def load(filename,n):
    q = np.zeros((328,84,128,150))
    file2 = open('/data/hslr2/coords', 'rb')
    x1,x2,x3 = pickle.load(file2)
    x1 = x1[24:352]
    x2 = x2[124:208]
    wdir = '/data/hslr2/'
    for i in range(n):
        file = open(wdir + filename +'/'+ filename + '_0%i' %i, 'rb')
        q[:,:,:,i] = pickle.load(file)
        
    return q,x1,x2,x3

#FAST 192ms -> vectorised
def integrate(n):
    q2 = q.sum(2)
    dL = np.zeros((328,84))
    dS = np.zeros(328)
    for n_r in range(293):
        R = ( a * np.log( x1[n_r] / np.sin(x2) ) + b ).astype(int)
        dz = x1[n_r] * (np.sin(x2)**-2.0)*d_th
        dL[R,:] = q2[R,:,n]*dz
        
        A = (1.0-2.0/x1[n_r])
        B = (1.0-3.0/x1[n_r])
        dS = dL.sum(1) * -1.5 * (A / B) * x1[n_r] ** (-3.0/2.0)
        
    dV = np.sum(dS)
    return dV

n=150

q,x1,x2,x3 = load('maxwell',n)
L = np.zeros(n)
for i in range(n):
    L[i] = integrate(i)

fileObject = open('/data/hslr2/Luminosity','wb')
pickle.dump(L,fileObject)
fileObject.close()

#time = np.linspace(300e-4,305*30.7812e-4,5)
#plt.plot(time,sum_over_r/np.average(sum_over_r))
#plt.plot(time,sum_over_r2/np.average(sum_over_r2))
##plt.xticks([0,0.25,0.5,0.75,1])
#plt.xlabel(r'Time $(GM/c^3) \times 10^4$')
#plt.ylabel(r'L')