import numpy as np

def spectrum(self,nu,R_min,R_max):
    
    isigma = 60/(np.pi**2)
    
    a1 = 0.004335413952759297 #dimensionless
    
    dT_rp = abs( 1.5 * self.q_rp/170 * (self.x1**(-0.5) * (self.C))[:,np.newaxis,np.newaxis]*isigma)**0.25
    
    #seperate into chunks, easier to read
    factor_nu = (nu**3)[:,np.newaxis,np.newaxis,np.newaxis]
#    coeff = 2*128 #/(x1[R_max-1]**2-x1[R_min]**2)
#    coeff = 1
    factor_r = (self.x1[R_min:R_max]**2)[np.newaxis,:,np.newaxis,np.newaxis]
    factor_dist = (np.exp(np.divide.outer(nu,dT_rp[R_min:R_max,:,:]))-1)**-1
    B_tot = (a1 * factor_nu * factor_r * factor_dist)

    return B_tot