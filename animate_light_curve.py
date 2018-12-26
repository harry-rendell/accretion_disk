import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from func.load_coords import load_coords
from func.load_quantity import load_quantity

# =============================================================================
# SLICE OR TIME EVOLUTION ANIMATION
# =============================================================================

#q must be in form q = q(r,theta,n) where an animation will be made created with
#each frame having an increasing n.

isigma = 60/(np.pi**2)

wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
#x1,x2,x3,A,B = load_coords(wdir)
#q_rp = load_quantity('maxwell_stress_rp',239,400) #start-end inclusive
#dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25


def create_animation(title,x_label,y_label,q,rmin,rmax,phimin=0,phimax=128):
    x1_log = np.log10(x1)
    fig,ax = plt.subplots(1, 1, figsize=(6,4))
    
    def animate(i): 
        ax.clear()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_ylim([-2.2,-0.9])
        ax.set_xlim([0.75,2.2])
        
#        ax.set_ylim([0,0.1])
#        ax.set_xlim([0,145])
#        
#        ax.set_ylim([-0.0015,0.0005])
#        ax.set_xlim([30,120])
        
        
        
        c = ax.plot(x1_log[rmin:rmax],np.log10(q[rmin:rmax,0,i]),lw=0.4,label=(i))
        ax.legend()
#        c = ax.plot(x1[rmin:rmax],(q[rmin:rmax,0,i]),lw=0.4)
        return c
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=-1)
    anim = animation.FuncAnimation(fig, animate,frames=162,interval=1000)
    anim.save('lightcurve_long_log_test.mp4', writer=writer)
    
    
create_animation(r'Temp variation with time',r'log$(r_g)$',r'log$(T)$',dT_rp,0,592)