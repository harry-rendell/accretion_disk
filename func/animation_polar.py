import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


def animation_polar(x,y,z,x_min,x_max,y_min,y_max,save_as,n_frames,log_or_lin,v_min,v_max):
    
    x = x[x_min:x_max]
    y = y[y_min:y_max]
    z = z[x_min:x_max,y_min:y_max,:]
    
    X,Y = np.meshgrid(y,x)
    
    fig,ax = plt.subplots(1, 1, figsize=(6,4), subplot_kw={'projection': 'polar'})
    ax.set_thetalim(y[0],y[-1])
    ax.set_rlim(x[0],x[-1]) 
#    ax.set_rorigin(-x1[rmin])
    ax.set_rorigin(0)
    
    z0 = z[:,:,0]
    
    
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
    writer = Writer(fps=5, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate,frames=n_frames,interval=200)
    anim.save(save_as +'.mp4', writer=writer)
        