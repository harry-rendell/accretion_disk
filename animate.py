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

def create_animation(q,Dr,Dtheta,Dphi):

    thetamin = 123-Dtheta
    thetamax = 125+Dtheta
    
    x,y = np.meshgrid(x2[thetamin:thetamax],x1[:Dr])
    fig,ax = plt.subplots(1, 1, figsize=(6,4), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(-np.pi/2)
    ax.set_thetalim(x2[thetamin],x2[thetamax-1])
#    ax.set_rlim(0,x1[Dr])
    
    q0 = q[:Dr,thetamin:thetamax,0]
    cont = ax.pcolormesh(x, y, q0,norm=colors.LogNorm(vmin=1e-4, vmax=1e-3),cmap=plt.cm.nipy_spectral)
#        cont = ax.pcolormesh(x, y, z,vmin=,vmax=,cmap=plt.cm.nipy_spectral)
    plt.colorbar(cont)
    
    def animate(i): 
        z = q[:Dr,thetamin:thetamax,i]
#        cont = plt.contourf(x, y, z,cmap=plt.cm.nipy_spectral) #slower
        plt.title(i)
        cont = ax.pcolormesh(x, y, z,norm=colors.LogNorm(vmin=1e-4, vmax=1e-3),cmap=plt.cm.nipy_spectral)
#        cont = ax.pcolormesh(x, y, z,vmin=,vmax=,cmap=plt.cm.nipy_spectral)
#        maxwell limits: ,vmin=-3e-4,vmax=2e-4,
        return cont
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate,frames=Dphi,interval=200)
    #plt.axis('off')
    anim.save('timeEvo_prs_20_zoom.mp4', writer=writer)
    
#create_animation(400,80,20)