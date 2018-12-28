import numpy as np
from func.load_coords import load_coords
from func.load_quantity import load_quantity
from func.animation_cartesian import animation_cartesian
from func.animation_polar import animation_polar

isigma = 60/(np.pi**2)
a1 = 0.004335413952759297 #dimensionless
a2 = 0.009126342080710765 #dimensionless

# =============================================================================
# LOAD
# =============================================================================
q_rp = load_quantity('maxwell_stress_rp',669,679) #start-end inclusive
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
x1,x2,x3,A,B = load_coords(wdir)

# =============================================================================
# ANIMATIONS
# =============================================================================
animation_cartesian(x1,q_rp[:,0,:],0,592,'TEST2',frames=11,n_fps=5,pt=0.3,y_lims=[-0.01,0])
animation_polar(x1,x3,q_rp,0,328,0,128,'TEST3',10,'lin',v_min=-0.015,v_max=0.0005)

# =============================================================================
# PLOT TEMP
# =============================================================================
dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25
animation_cartesian(nu,B_tot,0,328,'TEST3',18,5,0.3)
plot_cartesian(x1,dT_rp[:,:,:],range(10)*3,0,328,0.3,'log',r'Temperature variation',r'$r_g$',r'Temp','fit')


# =============================================================================
# PLOT BB
# =============================================================================
isigma = 60/(np.pi**2)
nu = np.logspace(-2,1,50)


plot_cartesian(nu,spectrum(q_rp[:,:,:3],nu,0,328).sum((1,2)), (0,1,2), 0, 328, 0.3, 'log',r'Spectrum of accretion disk, $6r_g-25r_g$',r'Frequency, $c^3/GM$',r'$B(\nu)$')
#leave out index 4 to iterate over phi
#leave out index 3 to iterate over r
#otherwise, summed over




# =============================================================================
# PLOTS plot(quantity,slice,title,linear or log, colormin,colormax,rmin,rmax,thetamin,thetamax,)
# =============================================================================
#plot_cartesian(q,0,r"",        'log',1e-3,1,0,616)
#plot_cartesian(q,0,r"",       'log',5e-5,1e-3,0,400,124-70,124+70)
#plot_cartesian(q,0,"Radial B",       'lin',-0.005,0.005,300,352)
#plot_cartesian(q,0,"Toroidal B",     'lin',-0.02,0.02,300,352)
#plot_cartesian(q,0,"Azimuthal B",    'lin',-5e-3,5e-3,300,352)
#plot_cartesian(q,0,"", 'lin',np.min(q),np.max(q),0,352,124-70,124+70)
#plot_cartesian(q,0,"", 'log',1e-5,1e-3,0,400,124-70,124+70)
#plot_cartesian(q,0,"Velocity", 'lin',0,1,0,352)

# =============================================================================
# SAVE FIG
# =============================================================================
#plt.savefig("mag_prs.png",dpi = 900,bbox_inches="tight")