import numpy as np
import matplotlib.pyplot as plt
#from func.load_coords import load_coords
#from func.load_quantity import load_quantity

isigma = 60/(np.pi**2)

# wdir = '/data/csr12/Theory/AccretionDisks/HoggReynolds2018_Dynamo/hr_01/run_files/'
#wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
    
# =============================================================================
# NEED TO INCLUDE AREA OF ANNULUS
# =============================================================================

#q_rp = load_quantity('maxwell_stress_rp',669,679) #start-end inclusive
#x1,x2,x3,A,B = load_coords(wdir)

def spectrum(q,nu,R_min=0,R_max=328):
    
    R_mid = 328;
    
    a1 = 0.004335413952759297 #dimensionless
    a2 = 0.009126342080710765 #dimensionless
    
    dT_rp = abs( 1.5 * q/170 * (x1**(-0.5) * (A / B))[:,np.newaxis]*isigma)**0.25
    
    #seperate into chunks, easier to read
    factor_nu = (nu**3)
#    coeff = 2*128/(x1[R_max-1]**2-x1[R_min]**2)
#    coeff = 1

    if (R_max <= R_mid) & (R_min < R_mid):
#        factor_r = (x1[R_min:R_max]**2)[:,np.newaxis]
        factor_dist = (np.exp(nu*dT_rp[R_min:R_max,:]**-1)-1)**-1
        B_tot = (factor_nu * factor_dist).sum(1) # * coeff * a1 * factor_r
        
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

def plot(n,nu,title,x_label,y_label,R_min=0,R_max=328):    
    
    dT_const = np.ones((592,128))*1
    fig,ax=plt.subplots()
    for i in range(n):
        ax.plot(x1[R_min:R_max], spectrum(q_rp[:,:,i],nu,R_min,R_max),color = 'r',label = 'inner',lw=0.3)
#    ax.legend()
#    plt.plot(nu_log,spectrum(dT_const,nu,0,592),color='k',lw=0.5)
#    plt.plot(nu_log,single_spectrum(0.08,nu),color='r',lw=0.5,ls = '-')
#    plt.xlim([]); plt.ylim([])
#    
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
#    ax.set_yscale('log'); ax.set_xscale('log')
    
    plt.show()

plot(10,0.1,r'Spectrum of accretion disk, $\nu=0.1$ $c^3/GM$',r'Radius, $r_g$',r'$B(r_g)$')


