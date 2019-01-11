import numpy as np

def spectrum(q_rp,nu,R_min,R_max,x1,A,B):
    
    isigma = 60/(np.pi**2)
    
    R_mid = 326;
    
    a1 = 0.004335413952759297 #dimensionless
    a2 = 0.009126342080710765 #dimensionless
    
    dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis]*isigma)**0.25
    
    #seperate into chunks, easier to read
    factor_nu = (nu**3)[:,np.newaxis,np.newaxis,np.newaxis]
    coeff = 2*128/(x1[R_max-1]**2-x1[R_min]**2)
#    coeff = 1

    if (R_max <= R_mid) & (R_min < R_mid):
        factor_r = (x1[R_min:R_max]**2)[np.newaxis,:,np.newaxis,np.newaxis]
        factor_dist = (np.exp(np.divide.outer(nu,dT_rp[R_min:R_max,:,:]))-1)**-1
        B_tot = (a1 * coeff * factor_nu * factor_r * factor_dist)
        
    elif (R_max > R_mid) & (R_min < R_mid):
        factor_r1 = (x1[R_min:R_mid]**2)[np.newaxis,:,np.newaxis,np.newaxis]
        factor_dist1 = (np.exp(np.divide.outer(nu,dT_rp[R_min:R_mid,:,:]))-1)**-1
        B_inner = (a1 * coeff * factor_nu * factor_r1 * factor_dist1)

        factor_r2 = (x1[R_mid:R_max]**2)[np.newaxis,:,np.newaxis,np.newaxis]
        factor_dist2 = (np.exp(np.divide.outer(nu,dT_rp[R_mid:R_max,:,:]))-1)**-1
        B_outer = (a2 * factor_nu * factor_r2 * factor_dist2)
        
        B_tot = np.concatenate((B_inner,B_outer),axis=1)
        
    elif (R_max > R_mid) & (R_min > R_mid):
        factor_r = (x1[R_min:R_max]**2)[np.newaxis,:,np.newaxis,np.newaxis]
        factor_dist =  (np.exp(np.divide.outer(nu,dT_rp[R_min:R_max,:,:]))-1)**-1
        B_tot = (a2 * factor_nu * factor_r * factor_dist)
    
    else:
        print('error, invalid input of values')
    
    return B_tot

def single_spectrum(T,nu,R_min=0,R_max=592):
#        factor_nu = (nu**3)
#        factor_dist = ( np.exp(nu/T)-1 )**-1
#        B_outer = (factor_nu * factor_dist)
        return nu**3*(np.exp(nu/T)-1)**-1