import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


def rotating_frame_animation_polar(fig,ax,x,y,z,x_min,x_max,y_min,y_max,save_as,n_frames,log_or_lin,v_min,v_max):
    
    x = x[x_min:x_max]
    y = y[y_min:y_max]
    z = z[x_min:x_max,y_min:y_max,:]
    
    X,Y = np.meshgrid(y,x)
    
    ax.set_thetalim(y[0],y[-1])
    ax.set_rlim(x[0],x[-1]) 
#    ax.set_rorigin(-x1[0])
    ax.set_rorigin(0)
    
    z0 = z[:,:,0]
    Dt = 30.7812

#    for R in range(x_max-1):
#        for t in range(n_frames): 
#            Dphi = x[R]**-0.5/(x[R]-2) * t * Dt #169.99084990353364
#            index = int(round((Dphi-0.004090615434360149)/(0.008181230868723452)))
#            
#            z[R,:,t] = np.roll(z[R,:,t],-index,axis=0)
    
    for t in range(n_frames): 
        Dphi =  0.005 * t * Dt 
        index = int(round((Dphi-0.004090615434360149)/(0.008181230868723452)))
        z[:,:,t] = np.roll(z[:,:,t],-index,axis=1)
    
    if log_or_lin == 'log':
        
        cont = ax.pcolormesh(X, Y, z0 ,norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.nipy_spectral)
        plt.colorbar(cont)
        
        def animate(i):
#            ax.cla()
            cont = ax.pcolormesh(X, Y, z[:,:,i],norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.nipy_spectral,label=i)
#            ax.legend()
            return cont
    
    elif log_or_lin == 'lin':
        
        cont = ax.pcolormesh(X, Y, z0,vmin=v_min, vmax=v_max,cmap=plt.cm.nipy_spectral)
        plt.colorbar(cont)
        
        def animate(i):
#            ax.cla()
            cont = ax.pcolormesh(X, Y, z[:,:,i],vmin=v_min,vmax=v_max,cmap=plt.cm.nipy_spectral,label=i)
#            ax.legend()
            return cont
    
    else:
        print('error, choose log or lin')
    
    
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=5000)

    anim = animation.FuncAnimation(fig, animate,frames=n_frames,interval=200)
    anim.save(save_as +'.mp4', writer=writer)

