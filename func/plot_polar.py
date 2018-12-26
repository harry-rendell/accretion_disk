 import pyPLUTO as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle

#hfont = {'fontname':'Latin Modern Roman'}

# =============================================================================
# READING DATA
# =============================================================================

#file = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/lum', 'rb')
#L = pickle.load(file)

#file2 = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/coords', 'rb')
#x1,x2,x3 = pickle.load(file2)

#wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data/'
#D = pp.pload(300,w_dir=wdir)
#x1 = D.x1; x2 = D.x2; x3 = D.x3
#q = (D.bx1*D.bx1 + D.bx2*D.bx2 + D.bx3*D.bx3)/2

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

# =============================================================================
# PLOT FUNCTION
# =============================================================================
def plot(q,n,title,log_or_lin,colormin,colormax,rmin=0,rmax=352,thetamin=0,thetamax=248):
    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    ax.set_theta_offset(-np.pi/2)
    ax.set_thetalim(x2[thetamin],x2[thetamax-1])
    ax.set_rlim(x1[rmin],x1[rmax-1])
    ax.set_rorigin(-x1[rmin])

    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position+46),ax.get_rmax()/(1.7),r'$r_g$',ha='center',va='center')
#    ax.text(np.radians(label_position+46),ax.get_rmax()/(2),r'$r_g$',ha='center',va='center')

#    ,rotation=label_position-51

    ax.set_ylabel(r'$\theta$',labelpad = 30,rotation = 0)
    ax.yaxis.set_label_position("right")

    for label in ax.get_xticklabels():
        label.set_rotation(51)

    ax.set_title(title)
    if log_or_lin == 'log':
        C = ax.pcolormesh(x2[thetamin:thetamax], x1[rmin:rmax], q[rmin:rmax,thetamin:thetamax,n],
                          norm=colors.LogNorm(vmin=colormin, vmax=colormax),cmap=plt.cm.Reds)
    elif log_or_lin == 'lin':
        C = ax.pcolormesh(x2[thetamin:thetamax],x1[rmin:rmax], q[rmin:rmax,thetamin:thetamax,n],
                          vmin=colormin,vmax=colormax,cmap=plt.cm.Reds,norm=MidpointNormalize(midpoint=0,vmin=np.min(q), vmax=np.max(q)))
#        cm.nipy_spectral
#        cm.seismic
#        cm.jet
    cbar = plt.colorbar(C,pad=0.13)
    cbar.set_label(r'Magnetic Pressure, $\frac{1}{2}B^2$')
#    return C,fig,ax

# =============================================================================
# PLOTS plot(quantity,slice,title,linear or log, colormin,colormax,rmin,rmax,thetamin,thetamax,)
# =============================================================================
#plot(q,0,r"",        'log',1e-3,1,0,616)
#plot(q,0,r"",       'log',5e-5,1e-3,0,400,124-70,124+70)
#plot(q,0,"Radial B",       'lin',-0.005,0.005,300,352)
#plot(q,0,"Toroidal B",     'lin',-0.02,0.02,300,352)
#plot(q,0,"Azimuthal B",    'lin',-5e-3,5e-3,300,352)
#plot(q,0,"", 'lin',np.min(q),np.max(q),0,352,124-70,124+70)
#plot(q,0,"", 'log',1e-5,1e-3,0,400,124-70,124+70)
#plot(q,0,"Velocity", 'lin',0,1,0,352)


#ax.set_theta_zero_location('W', offset=10)


#Dr = 500 Dtheta = 64 good range
#    Dr      = 616 max
#    Dtheta  = 123 max
    
plt.savefig("mag_prs.png",dpi = 900,bbox_inches="tight")
#plt.close()