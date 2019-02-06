import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation



def animation_polar(x,y,z,save_as,n_frames,log_or_lin,v_min,v_max):
    
    class MidpointNormalize(colors.Normalize):
		"""
		Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
		e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
		"""
		def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
			self.midpoint = midpoint
			colors.Normalize.__init__(self, vmin, vmax, clip)

		def __call__(self, value, clip=None):
			x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
			return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
#    x = x[x_min:x_max]
#    y = y[y_min:y_max]
#    z = z[x_min:x_max,y_min:y_max,:]
    
    X,Y = np.meshgrid(y,x)
    
    fig,ax = plt.subplots(1, 1, figsize=(6,4), subplot_kw={'projection': 'polar'})
    ax.set_thetalim(y[0],y[-1])
    ax.set_rlim(x[0],x[-1]) 
    ax.set_theta_offset(-np.pi/2)
    ax.set_rorigin(0)
    
    z0 = z[:,:,0]
    
    
# =============================================================================
#     
# =============================================================================
    
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position+38),ax.get_rmax()/(1.7),r'$r_g$',ha='center',va='center')
#    ax.text(np.radians(label_position+46),ax.get_rmax()/(2),r'$r_g$',ha='center',va='center')

#    ,rotation=label_position-51

    ax.set_ylabel(r'$\theta$',labelpad = 30,rotation = 0)
    ax.yaxis.set_label_position("right")

    for label in ax.get_xticklabels():
        label.set_rotation(51)

    if log_or_lin == 'log':
        C = ax.pcolormesh(y, x, z0, norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.Reds)
        cbar = plt.colorbar(C,pad=0.13)
        cbar.set_label(r'Maxwell Stress, $B_r B_{\phi}$')
        def animate(i):
            C = ax.pcolormesh(y, x, z[:,:,i], norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.Reds)
            return C
    
    elif log_or_lin == 'lin':
        C = ax.pcolormesh(y,x,z0,
                          vmin=v_min,vmax=v_max,cmap=plt.cm.seismic,norm=MidpointNormalize(midpoint=0,vmin=v_min, vmax=v_max))
        cbar = plt.colorbar(C,pad=0.13)
        cbar.set_label(r'Maxwell Stress, $B_r B_{\phi}$')
        def animate(i):
            C = ax.pcolormesh(y,x,z[:,:,i],
                          vmin=v_min,vmax=v_max,cmap=plt.cm.seismic,norm=MidpointNormalize(midpoint=0,vmin=v_min, vmax=v_max))
            return C

#        cm.nipy_spectral
#        cm.seismic
#        cm.jet
    
    
#    if log_or_lin == 'log':
#        
#        cont = ax.pcolormesh(X, Y, z0 ,norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.nipy_spectral)
#        plt.colorbar(cont)
#        def animate(i):
##            ax.cla()
#            cont = ax.pcolormesh(X, Y, z[:,:,i],norm=colors.LogNorm(vmin=v_min, vmax=v_max),cmap=plt.cm.nipy_spectral,label=i)
##            ax.legend()
#            return cont
#    
#    elif log_or_lin == 'lin':
#        
#        cont = ax.pcolormesh(X, Y, z0,vmin=v_min, vmax=v_max,cmap=plt.cm.nipy_spectral)
#        plt.colorbar(cont)
#        def animate(i):
##            ax.cla()
#            cont = ax.pcolormesh(X, Y, z[:,:,i],vmin=v_min,vmax=v_max,cmap=plt.cm.nipy_spectral,label=i)
##            ax.legend()
#            return cont
    
    else:
        print('error, choose log or lin')
    
    
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=16, bitrate=5000)

    anim = animation.FuncAnimation(fig,animate,frames=n_frames,interval=200)
    anim.save(save_as +'.mp4', writer=writer,dpi=250)
        
#animation_polar(x1,x2,q,'MS_correctlabel',128,'lin',v_min=-5e-4,v_max=2e-4)
