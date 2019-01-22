import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def rotating_frame_animation_polar(i, ax, x, y, z, x_min, x_max, y_min, y_max, log_or_lin, v_min, v_max):
    
    x = x[x_min:x_max]
    y = y[y_min:y_max]
    z = z[x_min:x_max,y_min:y_max,:]
    
    X,Y = np.meshgrid(y,x)
    
    ax.set_thetalim(y[0],y[-1])
    ax.set_rlim(x[0],x[-1]) 
    ax.set_rorigin(0)
    
#    z0 = z[:,:,0]
#    Dt = 30.7812

    if log_or_lin == 'log':

        plot = ax.pcolormesh(X, Y, z[:,:,i],norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.nipy_spectral,label=i)
#            ax.legend()
        return plot
    
    elif log_or_lin == 'lin':
        
        plot = ax.pcolormesh(X, Y, z[:,:,i],vmin=v_min,vmax=v_max,cmap=plt.cm.nipy_spectral,label=i)
#            ax.legend()
        return plot
    
    else:
        print('error, choose log or lin')
