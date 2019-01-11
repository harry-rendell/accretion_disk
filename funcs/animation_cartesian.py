import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animation_cartesian(fig,ax,x,y,x_min,x_max,save_as,n_frames,n_fps,pt,y_lims=None):
    
    x = x[x_min:x_max]
    y = y[x_min:x_max,:]
    
    ymax = np.max(y); ymin = np.min(y)
    
    def animate(i): 
        ax.clear()
        if y_lims == None:
            ax.set_ylim([ymin,ymax])
        elif y_lims != None:
            ax.set_ylim(y_lims)
        
        cont = ax.plot(x, y[...,i],label=i,lw=pt)
        ax.legend()
        return cont
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=n_fps, bitrate=5000)

    anim = animation.FuncAnimation(fig, animate,frames=n_frames,interval=200)
    anim.save(save_as +'.mp4', writer=writer,dpi=600)