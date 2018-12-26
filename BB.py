#import numpy as np
#import matplotlib.pyplot as plt
#from func.load_coords import load_coords
#from func.load_quantity import load_quantity
from func.animation_cartesian import animation_cartesian
from func.animation_polar import animation_polar
isigma = 60/(np.pi**2)

# wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
    
# =============================================================================
# NEED TO INCLUDE AREA OF ANNULUS
# =============================================================================

q_rp = load_quantity('maxwell_stress_rp',669,679) #start-end inclusive
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
x1,x2,x3,A,B = load_coords(wdir)


def spectrum(q_rp,nu,R_min,R_max):
    
    R_mid = 326;
    
    a1 = 0.004335413952759297 #dimensionless
    a2 = 0.009126342080710765 #dimensionless
    
    dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis]*isigma)**0.25
    
    #seperate into chunks, easier to read
    factor_nu = (nu**3)[:,np.newaxis,np.newaxis]
    coeff = 2*128/(x1[R_max-1]**2-x1[R_min]**2)
#    coeff = 1

    if (R_max <= R_mid) & (R_min < R_mid):
        factor_r = (x1[R_min:R_max]**2)[np.newaxis,:,np.newaxis]
        factor_dist = (np.exp(np.divide.outer(nu,dT_rp[R_min:R_max,:]))-1)**-1
        B_tot = (a1 * coeff * factor_nu * factor_r * factor_dist).sum((1,2))
        
    elif (R_max > R_mid) & (R_min < R_mid):
        factor_r1 = (x1[R_min:R_mid]**2)[np.newaxis,:,np.newaxis]
        factor_dist1 = (np.exp(np.divide.outer(nu,dT_rp[R_min:R_mid,:]))-1)**-1
        B_inner = (a1 * coeff * factor_nu * factor_r1 * factor_dist1)

        factor_r2 = (x1[R_mid:R_max]**2)[np.newaxis,:,np.newaxis]
        factor_dist2 = (np.exp(np.divide.outer(nu,dT_rp[R_mid:R_max,:]))-1)**-1
        B_outer = (a2 * factor_nu * factor_r2 * factor_dist2)
        
        B_tot = np.concatenate((B_inner,B_outer),axis=1).sum((1,2))
        
    elif (R_max > R_mid) & (R_min > R_mid):
        factor_r = (x1[R_min:R_max]**2)[np.newaxis,:,np.newaxis]
        factor_dist =  (np.exp(np.divide.outer(nu,dT_rp[R_min:R_max,:]))-1)**-1
        B_tot = (a2 * factor_nu * factor_r * factor_dist).sum((1,2))
    
    else:
        print('error, invalid input of values')
    
    return B_tot

def single_spectrum(T,nu,R_min=0,R_max=592):
        factor_nu = (nu**3)
        factor_dist = (( np.exp(nu/T)-1 ) )**-1
        B_outer = (factor_nu * factor_dist)
        return np.log(B_outer)

def plot(n,title,x_label,y_label):    

    
    N = 50
#    nu = np.linspace(1e-2,3,N)
    nu = np.logspace(-2,1,N)/5.391
#    nu_log = np.log(nu)
    
    dT_const = np.ones((592,128))*1
    fig,ax=plt.subplots()
    for i in range(n):
#        plt.plot(nu_log, spectrum(dT[:,:,i*10],nu,0,592),color = 'b',label = '616',lw=0.4)
        ax.plot(nu, spectrum(q_rp[:,:,0],nu,0,302),color = 'r',label = 'inner',lw=0.4)
        ax.plot(nu, spectrum(q_rp[:,:,3],nu,0,302),color = 'r',label = 'inner',lw=0.4)
        ax.plot(nu, spectrum(q_rp[:,:,7],nu,0,302),color = 'r',label = 'inner',lw=0.4)

#        ax.plot(nu, spectrum(q_rp[:,:,i],nu,302,592),color = 'k',label = 'outer',lw=0.4)

#        plt.plot(nu_log, spectrum(q_rp[:,:,i],nu,0,303),color = 'g',label = 'inner',lw=0.4)
#    ax.legend()
#    plt.plot(nu_log,spectrum(dT_const,nu,0,592),color='k',lw=0.5)
#    plt.plot(nu_log,single_spectrum(0.08,nu),color='r',lw=0.5,ls = '-')
#    plt.xlim([-10,4])
#    plt.ylim([-10000,1000])
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    ax.set_yscale('log'); ax.set_xscale('log')
    
    plt.show()

#plot(10,r'Spectrum of accretion disk, $6r_g-25r_g$ ',r'Frequency, $c^3/GM$',r'$B(\nu)$')

#animation_cartesian(x1,q_rp[:,0,:],0,592,'TEST2',frames=11,n_fps=5,pt=0.3,y_lims=[-0.01,0])
animation_polar(x1,x3,q_rp,0,328,0,128,'TEST3',10,'lin',v_min=-0.015,v_max=0.0005)
