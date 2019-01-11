import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from funcs.load_coords import load_coords
from funcs.load_quantity import load_quantity
from funcs.animation_cartesian import animation_cartesian
from funcs.animation_polar import animation_polar
from funcs.plot_cartesian import plot_cartesian
from funcs.plot_polar import plot_polar
from funcs.rotating_frame_animation_polar import rotating_frame_animation_polar
from spectrum import spectrum, single_spectrum

#a1 = 0.004335413952759297 #dimensionless
#a2 = 0.009126342080710765 #dimensionless

# =============================================================================
# LOAD
# =============================================================================
start = 669; end = start + 9;
q_rp = load_quantity('maxwell_stress_rp',start,end) #start-end inclusive
wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'
x1,x2,x3,A,B = load_coords(wdir)

# =============================================================================
# ANIMATIONS
# =============================================================================
animation_cartesian(x1,q_rp[:,0,:],0,592,'TEST2',frames=11,n_fps=5,pt=0.3,y_lims=[-0.01,0])
#animation_polar(x1,x3,q_rp,0,328,0,128,'TEST3',10,'lin',v_min=-0.015,v_max=0.0005)
plot_cartesian(x1,np.mean(bx1s[24:,:,:],axis=(1,2)), (0,), 0, 592, 0.3, 'lin')
plot_cartesian(x1,np.mean(bx1[24:,:,:],axis=(1,2)), (0,), 0, 592, 0.3, 'lin')
# =============================================================================
# PLOT TEMP
# =============================================================================
#dT_rp = abs( 1.5 * q_rp/170 * (x1**(-0.5) * (A / B))[:,np.newaxis,np.newaxis] * isigma)**0.25
#animation_cartesian(nu,B_tot,0,328,'TEST3',18,5,0.3)
#plot_cartesian(x1,dT_rp[:,:,:],range(10)*3,0,328,0.3,'log',r'Temperature variation',r'$r_g$',r'Temp','fit')

plt.plot(np.log(dx1))
# =============================================================================
# PLOT BB
# =============================================================================
N = 10
#nu = np.logspace(-0.7,0,N) #-0.45
#nu = np.linspace(10**(-0.45),10**(0.04),N)
nu = np.linspace(10**(-1.5),10**(-0.5),N)
t = np.linspace(start,end,end-start+1)*30.7812e-4

y = spectrum(q_rp[:,:,:],nu,0,328,x1,A,B).sum((1,2)).transpose(1,0)

for i in range(N):
    y[:,i] = y[:,i]-np.mean(y[:,i])

fig,ax = plt.subplots(1, 1, figsize=(6,4))    
plot_cartesian(fig,ax,t,y, range(N), 0, 0, 0.3, 'lin',r'Spectral radiance above $\nu_{max}$',r'$Time$ ($GM/c^3$)',r'$B_{\nu}-\bar{B}_{\nu}$',nu)
#r'$Time$ ($GM/c^3$)',r'$B(\nu)-\bar{B}(\nu)$'
for i in range(N):
    plt.plot(t[np.argmin(y[:,i])],np.min(y[:,i]),marker='x',ms=3)


#plot_cartesian(fig,ax,nu,spectrum(q_rp[:,:,:],nu,328,592,x1,A,B).sum((1,2)), (0,5,10), 0, 0, 0.3, 'log',r'Spectral radiance',r'log$_{10}$[ $\nu$ ($c^3/GM$) ]',r'log$_{10}$[ B$(\nu)$ ]')
#plot_cartesian(fig,ax,nu,single_spectrum(0.1,nu)*2,(0,), 0, 0, 0.3, 'log',r'Spectral radiance variation over the range $6r_g-25r_g$',r'Frequency, $c^3/GM$',r'$B(\nu)$')

#y = spectrum(q_rp[:,:,:],nu,0,328,x1,A,B).sum((1,2))

#avg = spectrum(q_rp[:,:,:],np.array([1]),0,328,x1,A,B).sum((1,2)).transpose(1,0)

#residual = y - avg
#plot_cartesian(nu,y, (0,5,10), 0, 328, 0.3, 'log',r'Spectrum of accretion disk, $6r_g-25r_g$',r'Frequency, $c^3/GM$',r'$B(\nu)$')
#nu = np.linspace(0.0001,5,10)

fig = plt.figure(figsize=(6,4))
ax1 = plt.subplot(211,projection = 'polar')
ax2 = plt.subplot(212)
rotating_frame_animation_polar(fig,ax1,x1,x3,abs(q_rp),0,328,0,128,'test_polar',10,'log',v_min=1e-5,v_max=1e-2)
animation_cartesian(fig,ax2,x1,q_rp.sum(1),0,328,'test_cartesian',10,5,0.3)

#animation_cartesian(fig,ax,x,y,x_min,x_max,save_as,n_frames,n_fps,pt,y_lims=None):


#plot_cartesian(nu,y, range(10), 0, 592, 0.3, 'log',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Time, $GM/c^3$',r'$L$',t)

y1 = y[:,0]; y2 = y[:,1]
#w1, psd1  = scipy.signal.welch(l[:,0])
#w2, psd2 = scipy.signal.welch(m[:,0])
#plt.plot(w1,psd1,lw=0.3)
#plt.plot(w2,psd2,lw=0.3)

w1,cf1 = signal.coherence(y[:,0],y[:,1],nperseg=64)
w2,cf2 = signal.coherence(y[:,0],y[:,2],nperseg=64)
w3,cf3 = signal.coherence(y[:,0],y[:,3],nperseg=64)
plt.plot(w1,cf1,w2,cf2,w3,cf3)
#leave out index 4 to iterate over phi
#leave out index 3 to iterate over r
#otherwise, summed over



#plot_cartesian((x1)**0.5/(x1-2),np.mean(q_rp,axis=1),10*range(10),0,592,0.3,'lin') #showing that keplerian velocity is obeyed
x=range(592)
lnv = 0.5*np.log(x1) - np.log(x1-2)+5.13574
plt.plot(np.log(x1),lnv)



l,m=np.polyfit(X,Y,1)
plt.plot(x,l*x+m)
plt.plot(x,x3)

X = np.log(x1[:328])
Y = np.log(np.mean(v3[:328],axis=(1,2)))
plt.plot(X,Y)

m,c = np.polyfit(X,Y,1)
plt.plot(X,m*X+c)

m = -0.7239448341438455
c = 0.7587790794206271
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
plt.savefig("Time_Lag_at__693_2.png",dpi = 600,bbox_inches="tight")






















