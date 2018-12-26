import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from func.load_coords import load_coords

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
    wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
    x1,x2,x3,A,B = load_coords(wdir)
    
    x,y = np.meshgrid(x3[phimin:phimax],x1[:rmax])
    fig,ax = plt.subplots(1, 1, figsize=(6,4), subplot_kw={'projection': 'polar'})
    ax.set_thetalim(x3[phimin],x3[phimax-1])
    ax.set_rlim(x1[rmin],x1[rmax-1])
    ax.set_rorigin(-x1[rmin])
    
    q0 = q[:rmax,phimin:phimax,0]
    cont = ax.pcolormesh(x, y, q0,norm=colors.LogNorm(vmin=0.002, vmax=0.13),cmap=plt.cm.nipy_spectral)
#    cont = ax.pcolormesh(x, y, q0,vmin=0.02, vmax=0.13,cmap=plt.cm.nipy_spectral)
    plt.colorbar(cont)

    def animate(i): 
        z = q[:rmax,phimin:phimax,i]
#        cont = plt.contourf(x, y, z,cmap=plt.cm.nipy_spectral) #slower
        plt.title(i)
        cont = ax.pcolormesh(x, y, z,norm=colors.LogNorm(vmin=0.02, vmax=0.13),cmap=plt.cm.nipy_spectral)
#       imshow?
#        cont = ax.pcolormesh(x, y, z,vmin=0.02, vmax=0.13,cmap=plt.cm.nipy_spectral)
#        maxwell limits: ,vmin=-3e-4,vmax=2e-4,
        return cont
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate,frames=18,interval=200)
    #plt.axis('off')
    anim.save('timeEvo_T_zoom_spike.mp4', writer=writer)
    
dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25

    
#create_animation(dT_rp,0,500)











