import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import pickle

#file = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/q/prs_20', 'rb')
#q = pickle.load(file)
#file2 = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/coords', 'rb')
#x1,x2,x3 = pickle.load(file2)

# =============================================================================
# SLICE OR TIME EVOLUTION ANIMATION
# =============================================================================

#q must be in form q = q(r,theta,n) where an animation will be made created with
#each frame having an increasing n.

def create_animation(q,rmin,rmax,phimin=0,phimax=128):
    N = 50
    nu = np.logspace(-2,0.2,N)
    nu_log = np.log(nu)
    
    fig,ax = plt.subplots(1, 1, figsize=(6,4))

    factor_nu = (nu**3)[:,np.newaxis,np.newaxis]
    factor_r = (2*np.pi*a1*(x1[rmin:rmax]**2))[np.newaxis,:,np.newaxis]

    def animate(i): 
        ax.clear()
        ax.set_xlim([0,1.6])
        ax.set_ylim([0,180])
        factor_dist =  ( ( np.exp(np.divide.outer(nu,dT_rp[rmin:rmax,:,i]))-1 ) )**-1
        B_tot = (factor_nu * factor_r * factor_dist).sum((1,2))
        cont = ax.plot(nu, B_tot)
        return cont
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate,frames=18,interval=200)
    #plt.axis('off')
    anim.save('timeEvo_B_zoom_spike.mp4', writer=writer)
    
dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25

    
create_animation(dT_rp,0,500)