import numpy as np
import matplotlib.pyplot as plt

def plot_cartesian(fig,ax,x,y,plot_list,pt,log_or_lin,labels=np.empty(0),fit=None):
    
    
    if len(np.shape(y)) == 1:
        y = y[:,np.newaxis]

        
    if log_or_lin == 'log':
        x = np.log10(x)
        y = np.log10(y)
#        ax.set_yscale('log'); ax.set_xscale('log')
        
    
    for i in plot_list:
        ax.plot(x,y[...,i],linewidth=pt,label = r'$\nu = $%.2f'%labels[i],color=plt.cm.jet_r(40*i))
    
#    if np.array_equal(labels,np.empty(0)) != True :
#    ax.legend(loc='lower right')
    
    if fit == 'fit':
        l,m=np.polyfit(x,np.mean(y,axis=tuple(range(1, y.ndim))),1)
        ax.plot(x,l*x+m)
        print 'power: ',l


