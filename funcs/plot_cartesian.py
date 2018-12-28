import numpy as np
import matplotlib.pyplot as plt

def plot_cartesian(x,y,plot_list,x_min,x_max,pt,log_or_lin,title='',x_label='',y_label='',fit=None):
    
#    y = np.mean(y,axis=1)
    
    if len(np.shape(y)) == 1:
        y = y[:,np.newaxis]
    
    x = x[x_min:x_max]
    y = y[x_min:x_max]
    
    fig,ax = plt.subplots(1, 1, figsize=(6,4))    

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_title(title)
    if log_or_lin == 'log':
        x = np.log(x)
        y = np.log(y)
#        ax.set_yscale('log'); ax.set_xscale('log')
    
    for i in plot_list:
        ax.plot(x,y[...,i],linewidth=pt)

    if fit == 'fit':
        l,m=np.polyfit(x,np.mean(y,axis=(1,2)),1)
        ax.plot(x,l*x+m)
        print 'power: ',l