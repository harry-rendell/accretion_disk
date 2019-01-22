import matplotlib.pyplot as plt
import numpy as np

def animation_cartesian(i,ax,x,y,x_min,x_max,pt,y_lims=None):
    

    
    x = x[x_min:x_max]
    y = y[x_min:x_max,:]
    
    ymax = np.max(y); ymin = np.min(y) 
    
    ax.clear()
    if y_lims == None:
        ax.set_ylim([ymin,ymax])
    elif y_lims != None:
        ax.set_ylim(y_lims)
    
    ax.plot(x,y[...,i  ],color = 'b',lw=0.3*pt)
    ax.plot(x,y[...,i+1],color = 'b',lw=0.6*pt)
    ax.plot(x,y[...,i+2],color = 'b',lw=pt)
    #show another graph with difference?    
   
#    ax.legend()

#    return plot