import matplotlib
from scipy.optimize import curve_fit
from matplotlib import ticker
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import pickle

#file2 = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/coords', 'rb')
#x1,x2,x3 = pickle.load(file2)

#gives checkerboardpattern
#q2 = np.zeros((Dr,DTh))
#q2[::2,1::2] = 1; q2[1::2,::2]=-1;

Dr = 616;
DTh = 248;

def vert_quick(r0,m):
    q = np.zeros((Dr,DTh))
    for i in range(m,DTh-m):
        R1 = int(230.65848172665787*np.log(r0/np.sin(x2[i]))-389.74685109951344)
        R2 = int(109.57292540168723*np.log(r0/np.sin(x2[i]))+71.18439340413698)
        if R1 < 487:
            q[R1,i]=1
            
        else:
            q[R2,i]=1
    
         
    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_rlim(0,r0+5)
    ax.set_theta_offset(-np.pi/2)
    ax.set_thetalim(x2[0],x2[-1])
    ax.grid(color='b', linestyle='-', linewidth=0.1)
    X1 = np.linspace(0,616,616)
    C1 = ax.pcolormesh(x2, x1, q,vmin=0, vmax=1,cmap=plt.cm.binary)
    plt.show(C1)
    return q
    
q = vert_quick(25,0)

#plt.plot(x1)

#Y = np.linspace(0,616,616)
#X = np.log(x1)

#m1,c1 = np.polyfit(X[100:400],Y[100:400],1)
#m2,c2 = np.polyfit(X[500:],Y[500:],1)

##plt.plot(X,m1*X+c1,X,m2*X+c2,X,Y)

#X = np.linspace(0,616,616)
#Y1 = np.exp(X*0.0043354139527592836+1.6897139363007663)
#Y2 = np.exp(X*0.009126342080710765-0.6496531250140436)
#plt.plot(X,x1,X,Y1,X,Y2)

