import numpy as np
import pickle
from scipy import signal
import matplotlib.pyplot as plt
from funcs.load_coords import load_coords
from funcs.load_quantity import load_quantity
from funcs.load_quantity_reduced import load_quantity_reduced
from funcs.plot_cartesian import plot_cartesian
from funcs.animation_generic import animation_generic
from spectrum import spectrum, single_spectrum

wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

''' use *args to pass self.x1 etc'''

class main():
    

    def load(self,name,start,end,reduced='yes'):
        
        fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/coords')
        coords = pickle.load(fileObject)
        fileObject.close()
        self.x1, self.x2, self.x3, self.C = coords
        
        if reduced == 'no':
            self.q_rp = load_quantity(name,start,end)
        elif reduced == 'yes':
            self.q_rp = load_quantity_reduced(self,name,start,end)

        
        self.start = start
        self.end = end
        self.t = np.linspace(start,end,end-start+1)*30.7812e-4
        
        self.dF = abs((self.q_rp/170.0) * (self.C * self.x1 ** (-0.5))[:,np.newaxis,np.newaxis]) #Local flux
        self.dT = (abs(self.dF)**0.25)                                                      #Local temperature                        
        
    def BB_spectrum(self,nu):
        self.spectrum = spectrum(self,nu,0,328).sum((1,2)).transpose(1,0)

#        for i in range(N):
#            ax.plot(self.t[np.argmax(y[:,i])],np.max(y[:,i]),marker='x',ms=3,color=color)
            
    def multi_animation(self):
        animation_generic(self.x1, self.x3, abs(self.q_rp), self.x1, abs(self.q_rp).sum(1), n_frames = 98, n_fps = 7, save_as = 'generic_test_100_450')
        
        
    def spectrogram(self):
        x = abs(self.q_rp).sum(1,2)
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        signal.spectrogram(x)
    
    
    def cartesian_nu(self,nu):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        y = spectrum(self,nu,0,328).sum((1,2))
        plot_cartesian(fig,ax,nu,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'log',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Freq, $c^3/GM$',r'$L$',self.t)
        
#        plot_cartesian(fig,ax,x,y,plot_list,x_min,x_max,pt,log_or_lin,title='',x_label='',y_label='',labels=np.empty(0),fit=None):

    def cartesian_t(self,nu,norm='norm'):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        y = spectrum(self,nu,0,328).sum((1,2)).transpose(1,0)
        
        plot_cartesian(fig,ax,self.t,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'lin',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Time, $GM/c^3$',r'$L$',nu)

    def temperature(self,skip):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        
        
#        plot_cartesian(fig,ax,self.t,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'lin',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Time, $GM/c^3$',r'$L$',nu)

        plot_cartesian(fig,ax,self.x1,self.dT[:,:,::skip].sum(1),range(self.dT_rp.shape[-1]),0,328,0.3,'log',r'Temperature variation',r'log$(r_g)$',r'log(T)',self.t[::skip],'fit',color=None)

#        Labels are wrong, should be t not nu
        
    def luminosity(self,fig,ax,R_min,R_max):
#        load_quantity_reduced
        a1 = 0.004335413952759297 #dimensionless
#        a2 = 0.009126342080710765 #dimensionless
        
        L = (abs(self.q_rp.sum(1)) * (a1 * self.x1 ** 1.5 * self.C)[:,np.newaxis]).sum(0)
        L = (L-np.mean(L))/np.mean(L)
        ax.plot(self.t,L,lw = 0.3)
        

    

#%%
data = main()
data.load('maxwell_stress_rp',500,510,reduced='no')
#856
#%%
'''PLOT BB'''
#nu = np.linspace(10**(-0.45),10**(0.04),10) #above max
fig,ax = plt.subplots(1, 1, figsize=(6,4))
N = 10
nu = np.linspace(10**(-6),10**(0),N)
y = spectrum(data,nu,0,328).sum((1,2)).transpose(1,0)



peak = 3

for i in range(N):
    y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.mean(y[:,i])

title   = r'Spectral radiance above $\nu_{max}$'
x_label = r'$Time$ $(GM/c^3)\times 10^4$'
y_label = r'$B_{\nu}-\bar{B}_{\nu}$'

plot_cartesian(fig,ax,data.t,y[:,:peak], range(peak), 0, 0, 0.3, 'lin',title,x_label,y_label,labels = nu,fit = None,color = 'b')
plot_cartesian(fig,ax,data.t,y[:,peak]  , (0,)      , 0, 0, 0.5, 'lin',title,x_label,y_label,labels = nu,fit = None,color = 'k')
plot_cartesian(fig,ax,data.t,y[:,peak:], range(10-peak), 0, 0, 0.3, 'lin',title,x_label,y_label,labels = nu,fit = None,color = 'r')
nu = np.linspace(10**(-0.45),10**(0),N)


#%%
'''same as plotBB'''
data.cartesian_t(nu)

#%%
'''Temp vs radius r^(-3/4)'''
data.temperature(skip=30)
    
#%%
'''spectrogram'''

data.spectrogram()

#%%
'''luminosity'''
fig,ax = plt.subplots(1, 1, figsize=(6,4))
data.luminosity(fig,ax)

#%%
'''coherence'''

#nu = np.logspace(np.log10(0.1),np.log10(2),5)
#data.BB_spectrum(nu)

def coherence(self,nu):
#    N=11
    
#    nu = np.linspace(10**(-0.45),10**(0.04),N) #above max
#    nu = np.linspace(10**(-0.45),10**(-0.2),N) #above max
    y = self.spectrum
    N = len(nu)
    peak = 2
    for i in range(N):
#        y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.mean(y[:,i])
        y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.std(y[:,i])
        
    fig1,ax1 = plt.subplots(1, 1, figsize=(6,4))
    fig3,ax3 = plt.subplots(1, 1, figsize=(6,4))
    
    
    '''Below'''
    for i in range(peak):
        result = signal.correlate(y[:,peak],y[:,i])
        w,cf = signal.coherence(y[:,peak],y[:,i])
        
        ax3.plot(w,cf,lw = 0.4,color = 'r')
        ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1),result,lw = 0.4,color = 'r')
    '''Above'''
    for i in range(peak+1,N):
        result = signal.correlate(y[:,peak],y[:,i])
        w,cf = signal.coherence(y[:,peak],y[:,i])
        
        ax3.plot(w,cf,lw = 0.4,color = 'b')
        ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1),result,lw = 0.4,color = 'b')

    '''At'''
    result = signal.correlate(y[:,peak],y[:,peak])
    ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1),result,lw = 0.4,color = 'k')
    w,cf = signal.coherence(y[:,peak],y[:,peak])
    ax3.plot(w,cf,lw = 0.4,color = 'k')
        
    ax1.set_xlabel(r'Time lag $(GM/c^3)\times 10^4$'); ax1.set_ylabel('Cross correlation')
    ax3.set_xlabel('Temporal frequency'); ax3.set_ylabel('Coherence')
    

    fig2,ax2 = plt.subplots(1, 1, figsize=(6,4))
    title   = r'Spectral radiance above $\nu_{max}$'
    x_label = r'$Time$ $(GM/c^3)\times 10^4$'
    y_label = r'$B_{\nu}-\bar{B}_{\nu}$'



    plot_cartesian(fig2,ax2,self.t,y[:,:peak], range(peak), 0, 0, 0.3, 'lin',title,x_label,y_label,labels = nu[:peak],fit = None,color = 'b')
    plot_cartesian(fig2,ax2,self.t,y[:,peak]  , (0,)      , 0, 0, 0.5, 'lin',title,x_label,y_label,labels = [nu[peak]],fit = None,color = 'k')
    plot_cartesian(fig2,ax2,self.t,y[:,(peak+1):], range(N-peak-1), 0, 0, 0.3, 'lin',title,x_label,y_label,labels = nu[peak:],fit = None,color = 'r')

    
coherence(data,nu)

#%%
'''half light radius'''

def half_radius(self,nu,R_max):
    
    a1 = 0.004335413952759297 #dimensionless
    a2 = 0.009126342080710765 #dimensionless
    
    dB = (nu**3)[:,np.newaxis,np.newaxis,np.newaxis] * (np.exp( np.divide.outer(nu,self.dT[:R_max]))-1) ** -1
    
    if R_max <= 328:
        
        B = a1 * (self.x1[np.newaxis,:R_max,np.newaxis,np.newaxis]**2 * dB).sum(2)
    
    elif R_max > 328:
        
        B_inner = a1 * (self.x1[np.newaxis,:328     ,np.newaxis,np.newaxis]**2 * dB[:,:328,:,:]).sum(2)
        B_outer = a2 * (self.x1[np.newaxis,328:R_max,np.newaxis,np.newaxis]**2 * dB[:,328:R_max,:,:]).sum(2)
        B = np.concatenate((B_inner,B_outer),axis=1)
    
    B_cumsum = np.cumsum(B,axis = 1)
    
    L = (abs(self.q_rp[:R_max].sum(1)) * (a1 * self.x1 ** 1.5 * self.C)[:R_max,np.newaxis]).sum(0)
    
    Norm_disc = B_cumsum / L[np.newaxis,np.newaxis,:]
    Norm_freq = B_cumsum / B_cumsum[:,-1,:][:,np.newaxis,:]
    
    return np.mean(Norm_disc,axis=2), np.mean(Norm_freq,axis=2),L

#nu = np.logspace(np.log10(0.03),np.log10(0.2),30)
nu = np.linspace(0.03,0.3,30)

N = len(nu)
r_max = 328


def plot_half_radius(B_d,B_f,lum):
    hlr = np.zeros(N)
    for j in range(N):
        for i in range(r_max):
            if B_freq[j,i] > 0.5:
                hlr[j] = data.x1[i]
                break
    
    fig3,ax3 = plt.subplots(1, 1, figsize=(9,6))
    ax3.plot(nu,hlr,lw=0.4,marker='o',ms=2,color='b')
    
    ax3.set_title(r'Half light radius and spectral intensity$')
    ax3.set_xlabel(r'Spectral frequency $\nu$')
    ax3.set_ylabel(r'Radius $r / r_g$')
    
    ax4 = ax3.twinx()
    ax4.plot(nu,B_disc.sum(1),lw=0.4,marker='o',ms=2,color='r')
    ax4.set_ylabel(r'Spectral intensity $L(\nu)$')
    
    fig1,ax1 = plt.subplots(1, 1, figsize=(9,6))
    ax1.set_title(r'Half light radius normalised by $L_{\nu}$')
    ax1.set_xlabel(r'Radius $r/r_g$')
    ax1.set_ylabel(r'$B_{\nu}(<r)$')
    
    ax1.axhline(y=0.5,lw=0.4,ls='--')
    
    plot_cartesian(fig1,ax1,data.x1[:r_max],B_f.transpose(1,0),range(len(nu)),0.3,'lin',labels = nu)
#    plt.savefig("HLR.pdf",dpi = 300,bbox_inches="tight")
    
    
    fig2,ax2 = plt.subplots(1, 1, figsize=(9,6))
    ax2.set_title(r'Half light radius, normalised by $L$')
    ax2.set_xlabel(r'Radius $r/r_g$')
    ax2.set_ylabel(r'$B_{\nu}(<r)$')
    
    plot_cartesian(fig2,ax2,data.x1[:r_max],B_d.transpose(1,0),range(len(nu)),0.3,'lin',labels = nu)
#    plt.savefig("HLR2.pdf",dpi = 300,bbox_inches="tight")

B_disc,B_freq,L = half_radius(data,nu,r_max)
plot_half_radius(B_disc,B_freq,L)



