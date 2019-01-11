import numpy as np
import matplotlib.pyplot as plt

def plot_cartesian(fig,ax,x,y,plot_list,x_min,x_max,pt,log_or_lin,title='',x_label='',y_label='',labels=np.empty(0),fit=None):
    
#    y = np.mean(y,axis=1)
    
    if len(np.shape(y)) == 1:
        y = y[:,np.newaxis]
    
#    x = x[x_min:x_max]
#    y = y[x_min:x_max]
    
#    fig,ax = plt.subplots(1, 1, figsize=(6,4))    

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_title(title)
    if log_or_lin == 'log':
        x = np.log10(x)
        y = np.log10(y)
#        ax.set_yscale('log'); ax.set_xscale('log')
    
    for i in plot_list:
        ax.plot(x,y[...,i],linewidth=pt,label = r'$\nu = $%.2f'%labels[i])
    
#    if np.array_equal(labels,np.empty(0)) != True :
    ax.legend(loc=4)
    
    if fit == 'fit':
        l,m=np.polyfit(x,np.mean(y,axis=(1,2)),1)
        ax.plot(x,l*x+m)
        print 'power: ',l


