import numpy as np
import pickle
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from funcs.load_coords import load_coords
from funcs.load_quantity import load_quantity
from funcs.load_quantity_reduced import load_quantity_reduced
from funcs.plot_cartesian import plot_cartesian
from funcs.animation_generic import animation_generic
from spectrum import spectrum, single_spectrum
from funcs.animation_polar import animation_polar

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
        
        self.dF = abs((self.q_rp/170.0) * (1.5 * self.C * self.x1 ** (-0.5))[:,np.newaxis,np.newaxis]) #Local flux
        self.dT = (60.0/(np.pi**2) * abs(self.dF)**0.25)                                               #Local temperature           
    
    def doppler(self,i):
        
        r = self.x1
        theta = (self.x3+np.pi) # *6 propagates the disc around 2pi, makes the orbital modulation frequency much cleaner! However, velocities now *6
        
        a = ( (r / (r-3)) ** 0.5 )[:,np.newaxis]
        b = np.outer( (r-2)**(-0.5) , np.cos(theta)*np.sin(i) )
        
        self.z = a*(1+b)
               
        
    def BB_spectrum(self,nu):
        self.spectrum = spectrum(self,nu,0,328).sum((1,2)).transpose(1,0)

    def multi_animation(self):
        animation_generic(self.x1, self.x3, abs(self.q_rp), self.x1, abs(self.q_rp).sum(1), n_frames = 98, n_fps = 7, save_as = 'generic_test_100_450')
        
    def ani_polar(self):
        animation_polar(self.x1,self.x3,abs(self.q_rp),0,328,0,128,'TEST3',10,'log',v_min=1e-5,v_max=8e-3)
        
    def spectrogram_(self):
        x = abs(self.q_rp).sum((1,2))
        fig,ax = plt.subplots(1, 1, figsize=(6,4))
        freqs, times, signal = signal.spectrogram(x)
        ax.pcolormesh(freqs,times,signal)
    
    def cartesian_nu(self,nu):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        y = spectrum(self,nu,0,328).sum((1,2))
        plot_cartesian(fig,ax,nu,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'log',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Freq, $c^3/GM$',r'$L$',self.t)
        
#        plot_cartesian(fig,ax,x,y,plot_list,x_min,x_max,pt,log_or_lin,title='',x_label='',y_label='',labels=np.empty(0),fit=None):

    def cartesian_t(self,nu,norm='norm'):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        y = spectrum(self,nu,0,328).sum((1,2)).transpose(1,0)
        
        ax.set_title(r'Luminosity variation for a given frequency $6r_g-25r_g$')
        ax.set_xlabel(r'Time, $GM/c^3$')
        ax.set_ylabel(r'$L$')
        
        plot_cartesian(fig,ax,self.t,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'lin',nu)

    def temperature(self,skip):
        fig,ax = plt.subplots(1, 1, figsize=(6,4))  
        fig2,ax2 = plt.subplots(1, 1, figsize=(6,4))  
        
        ax.set_title(r'Temperature variation')
        ax.set_xlabel(r'log$(r_g)$')
        ax.set_ylabel(r'log(T)')
        
        ax2.set_title(r'Peak frequency of emission')
        ax2.set_xlabel(r'$(r_g)$')
        ax2.set_ylabel(r'Freq, $c^3/GM \times 10^{12}$')

        
#        plot_cartesian(fig,ax,self.t,y, range(self.q_rp.shape[-1]), 0, 592, 0.3, 'lin',r'Luminosity variation for a given frequency, $6r_g-25r_g$',r'Time, $GM/c^3$',r'$L$',nu)

        plot_cartesian(fig,ax,self.x1,self.dT[:,:,::skip].sum(1),range(self.dT.shape[-1]/skip),0.3,'log',self.t[::skip],'fit')
        
        
        
        plot_cartesian(fig2,ax2,self.x1,self.dT[:,:,::skip].sum(1)*(3.0/2.9),range(self.dT.shape[-1]/skip),0.3,'lin',self.t[::skip],'fit')
#        Labels are wrong, should be t not nu
        
    def luminosity(self,R_min=0,R_max=592):
    #        load_quantity_reduced
        a1 = 0.004335413952759297 #dimensionless
    #        a2 = 0.009126342080710765 #dimensionless
        
        L_shift_ = (abs( (self.q_rp * (self.z[:,:,np.newaxis]**-4)) .sum(1))[R_min:R_max] * (a1 * self.x1 ** 1.5 * self.C)[R_min:R_max,np.newaxis]).sum(0)
        L_       = (abs(  self.q_rp.sum(1))                            [R_min:R_max] * (a1 * self.x1 ** 1.5 * self.C)[R_min:R_max,np.newaxis]).sum(0)
      
#        self.L = L_
#        self.L_shift = L_shift_
       
        self.L = (L_-np.mean(L_))/np.mean(L_)
        self.L_shift = (L_shift_-np.mean(L_shift_))/np.mean(L_shift_)
        

    

#%%
data = main()
data.load('maxwell_stress_rp',239,1262,reduced='no')
#665,685
#1262
#nu = np.logspace(10**(-0.45),10**(0.04),5) #above max
nu = np.linspace(0.01,1,5)
data.BB_spectrum(nu)
#%%
'''PLOT BB'''
#nu = np.linspace(10**(-0.45),10**(0.04),10) #above max

N = 9
#nu = np.logspace(-2,0,N)
nu = np.linspace(0.05,1.1,N)
y = abs(spectrum(data,nu,0,326).sum((1,2)).transpose(1,0))

#for i in range(N):
#    y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.mean(y[:,i])
fig,ax = plt.subplots(1, 1, figsize=(6,4))

'''subtract the downward trend, assuming it is linear'''
for i in range(N):
    a,b = np.polyfit(data.t,y[:,i],1)
#    ax.plot(data.t,a*data.t+b,lw=0.3)
#    print(b)
    y[:,i] = abs(y[:,i] - a*data.t + a*data.t[len(data.t)/2])

ax.set_title(r'Spectral radiance above $\nu_{max}$')
ax.set_ylabel(r'$Time$ $(GM/c^3)\times 10^4$')
ax.set_ylabel(r'$B_{\nu}-\bar{B}_{\nu}$')
plot_cartesian(fig,ax,data.t,y[:,:], range(N), 0.3, 'lin',labels = nu,fit = None)
#plot_cartesian(fig,ax,data.t,y[:, peak],        (0,), 0.5, 'lin',labels = nu,fit = None)
#plot_cartesian(fig,ax,data.t,y[:,peak:], range(N-peak),0.3, 'lin',labels = nu,fit = None)

def log_normal(x,A,mu,sigma):
    return A/(x*sigma)*( np.exp( -( np.log(x)-mu )**2/( 2*sigma**2 ) ) )

def normal(x,A,mu,sigma):
    return A*( np.exp( -( x-mu )**2/( 2*sigma**2 ) ) )

fig2,ax2 = plt.subplots(3,3)

for i in range(N):
    j = (i / 3,i % 3)
    ydata,bins,irrev = ax2[j].hist(y[:,i],30,label = r'$\nu = $%.2f'%nu[i],color = "skyblue", ec="k")

    ax2[j].set_xlabel(r'$L_{\nu}$')
    ax2[j].set_ylabel(r'Occurance')
    
    n = len(bins)
    
    x = np.linspace(bins[0],bins[-1],n-1)
    smoothedx = np.logspace(np.log10(bins[0]*0.95),np.log10(bins[-1]),100)
    mean = sum(x*ydata)/n
    sigma = np.sqrt(sum(ydata * (x - mean)**2) / sum(ydata))
    #    p_0 = [max(ydata)/(sigma)*np.exp(0.5*sigma**2-mean) ,np.exp(mean+sigma**2), np.exp(2*mean+sigma**2)*(np.exp(sigma**2)-1)]
    if i < 4:
        popt2,pcov2 = scipy.optimize.curve_fit(    normal,x,ydata,p0=[max(ydata),mean,sigma])
        ax2[j].plot(smoothedx,    normal(smoothedx,popt2[0],popt2[1],popt2[2]),label='normal',lw = 1.2,color = 'b')
#    if i >= 4:
    popt1,pcov1 = scipy.optimize.curve_fit(log_normal,x,ydata,p0=[max(ydata),np.log(mean),sigma])
    ax2[j].plot(smoothedx,log_normal(smoothedx,popt1[0],popt1[1],popt1[2]),label='log normal',lw = 1.2,color = 'r')
    ax2[j].legend(loc=1, prop={'size': 9})
    
plt.subplots_adjust(hspace=0.3)
#    ax2[i / 3,i % 3].set_title(r'$\nu = $%.2f'%nu[i])
    



#%%
'''same as plotBB'''
data.cartesian_t(nu)

#%%
'''Temp vs radius r^(-3/4)'''
data.temperature(skip=10)

#%%
'''spectrogram'''

#data.spectrogram()

x = spectrum(data,np.array([0.35]),0,328)
x = x.sum(2)[0,:,:]


#x = np.mean(abs(data.q_rp),axis=(1,2))
fig,ax = plt.subplots(1, 1, figsize=(6,4))
freqs, times, signals = signal.spectrogram(x,nperseg=32)
F,T = np.meshgrid(times,freqs)
ax.pcolormesh(T,F,signals)

#%%
'''luminosity'''
fig,ax = plt.subplots(1, 1, figsize=(6,4))
data.doppler(0)
data.luminosity(0,17)

unshifted = data.L
shifted = data.L_shift



#%%
'''coherence'''
#

#nu = np.logspace(np.log10(0.1),np.log10(2),5)


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
    
#    mid = range(20,30)
    '''Below'''
    for i in range(peak):
        result = signal.correlate(y[:,peak],y[:,i],mode='valid')
        w,cf = signal.coherence(y[:,peak],y[:,i])
        
        ax3.plot(w,cf,lw = 0.4,color = 'r')
#        ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1)[4:11],result,lw = 0.4,color = 'r')
        ax1.plot(result,lw = 0.4,color = 'r')
    '''Above'''
    for i in range(peak+1,N):
        result = signal.correlate(y[:,peak],y[:,i],mode='valid')
        w,cf = signal.coherence(y[:,peak],y[:,i])
        
        ax3.plot(w,cf,lw = 0.4,color = 'b')
#        ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1)[4:11],result,lw = 0.4,color = 'b')
        ax1.plot(result,lw = 0.4,color = 'b')

    '''At'''
    result = signal.correlate(y[:,peak],y[:,peak],mode='valid')
#    ax1.plot(np.linspace(-self.t[-1],self.t[-1],2*len(self.t)-1),result,lw = 0.4,color = 'k')
    ax1.plot(result,lw = 0.4,color = 'k')
    w,cf = signal.coherence(y[:,peak],y[:,peak])
    ax3.plot(w,cf,lw = 0.4,color = 'k')
        
    ax1.set_xlabel(r'Time lag $(GM/c^3)\times 10^4$'); ax1.set_ylabel('Cross correlation')
    ax3.set_xlabel('Temporal frequency'); ax3.set_ylabel('Coherence')
    
    
    fig2,ax2 = plt.subplots(1, 1, figsize=(6,4))
    ax2.set_title(r'Spectral radiance above $\nu_{max}$')
    ax2.set_xlabel(r'$Time$ $(GM/c^3)\times 10^4$')
    ax2.set_ylabel(r'$B_{\nu}-\bar{B}_{\nu}$')



    plot_cartesian(fig2,ax2,self.t,y[:,:peak], range(peak), 0.3, 'lin',labels = nu[:peak],fit = None)
    plot_cartesian(fig2,ax2,self.t,y[:,peak]  , (0,)      , 0.5, 'lin',labels = [nu[peak]],fit = None)
    plot_cartesian(fig2,ax2,self.t,y[:,(peak+1):], range(N-peak-1), 0.3, 'lin',labels = nu[peak:],fit = None)

    
coherence(data,nu)

#%% 
''' PSD '''
def psd(ax1,ax2,y):
    N = len(nu)

    y = (y-np.mean(y,axis=0))/np.std(y,axis=0)

    for i in range(N):
        a1,a2 = signal.welch(y[:,i],nperseg=256)
#        a2 = (a2-np.mean(a2))/np.std(a2)
        ax1.plot(a1,a2,label=r'$\nu = $%.2f'%nu[i],lw=0.4,color=plt.cm.jet(30*(N-i)))
        ax2.plot(y[:,i],label=r'$\nu = $%.2f'%nu[i],lw=0.4,color=plt.cm.jet(30*(N-i)))
    ax1.legend()
        
fig1,ax1 = plt.subplots(1, 1, figsize=(6,4))
fig2,ax2 = plt.subplots(1, 1, figsize=(6,4))

ax1.set_xlabel(r'Temporal frequency')
ax1.set_ylabel(r'Power')
ax1.set_title(r'PSDs for different spectral frequencies')

psd(ax1,ax2,data.spectrum)


#plt.savefig("HLR.pdf",dpi = 300,bbox_inches="tight")


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
    
    return np.mean(Norm_disc,axis=2), np.mean(Norm_freq,axis=2),L,B

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
    
    ax3.set_title(r'Half light radius and spectral intensity')
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
    plt.savefig("splitting_long.pdf",dpi = 300,bbox_inches="tight")

B_disc,B_freq,L,B = half_radius(data,nu,r_max)
plot_half_radius(B_disc,B_freq,L)
plt.plot(B)

#%%
'''slice animation'''

data.ani_polar()

#%%


animation_polar(x1,x2,q,0,352,0,170,'TEST3',10,'lin',v_min=-5e-4,v_max=2e-4)



#%%
''' IMPORTING DBL '''

import pyPLUTO as pp
D = pp.pload(300,w_dir=wdir)
q = (D.bx1*D.bx3)[:352,39:209,:]
x1 = D.x1
x2 = D.x2[39:209]
x3 = D.x3

#%%
''' TIME LAGS '''
y = data.spectrum
#for i in range(11):
#    y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.std(y[:,i])

sig1 = np.fft.fft(y[:,0])
sig2 = np.fft.fft(y[:,1])
sig3 = np.fft.fft(y[:,2])

plt.plot(y[:,0])
plt.plot(y[:,1])
plt.plot(y[:,2])
plt.figure()
#freqs = np.fft.rfftfreq(len(sig1))  
freqs = range(1,len(sig1)+1)

time_lag = np.angle(sig1*np.conj(sig2))/freqs
time_lag2 = np.angle(sig1*np.conj(sig3))/freqs

plt.plot(freqs,time_lag,marker='o',lw=0.3)
plt.plot(freqs,time_lag2,marker='+',lw=0.3)

#%%
''' TESTING MODELS '''

fig,ax = plt.subplots(1,1,figsize=(6,4))

def model1(r,A,B,C):
    return A * r**(-B) * (1- (C / r) ** 0.5)**0.25

def model2(r,A,B):
    return A * r**(-B)

xdata = data.x1
ydata = data.dT.mean((1,2))

l,m=np.polyfit(np.log(xdata),np.mean(np.log(ydata),axis=tuple(range(1, ydata.ndim))),1)
#ax.plot(x,l*x+m)
#print 'power: ',l

popt1,pcov1 = scipy.optimize.curve_fit(model1,xdata,ydata,p0=[1,0.75,6])
popt2,pcov2 = scipy.optimize.curve_fit(model2,xdata,ydata,p0=[1,0.75])

ax.plot(xdata,model1(xdata,popt1[0],popt1[1],popt1[2]),label='model_full',lw = 0.5)
ax.plot(xdata,model2(xdata,popt2[0],popt2[1]),label='model_power', lw = 0.5)
ax.plot(xdata,ydata,label='data',lw = 0.5)
ax.legend()

ax.set_xlabel(r'Radius $r / r_g$')
ax.set_ylabel(r'Temperature')
ax.set_title(r'Testing models')


#%%
''' EFFECT OF REDSHIFT TO T(R)'''
data.doppler(np.pi/2)
shifted = np.mean(data.dT / data.z[:,:,np.newaxis],axis=1)[:,0]
unshifted = np.mean(data.dT,axis=1)[:,0]

fig,ax = plt.subplots(1,1,figsize=(6,4))

ax.plot(shifted,label='shifted',lw=0.7)
ax.plot(unshifted,label='unshifted',lw=0.7)

ax.legend()

#%%
''' EFFECT OF REDSHIFT TO L(t) '''

data.doppler(np.pi/2)
data.luminosity(300,301)

unshifted = data.L
shifted = data.L_shift
N = len(shifted)


fig,ax = plt.subplots(1,1,figsize=(6,4))

ax.plot(shifted,label='shifted',lw=0.7)
ax.plot(unshifted,label='unshifted',lw=0.7)

#ax3 = ax.twinx()
fig3,ax3 = plt.subplots(1,1)

freqs3, psd3 = signal.welch((shifted-unshifted)* np.blackman(N))

ax3.plot( freqs3,psd3 , lw=0.4)

ax.legend()

fig2,ax2 = plt.subplots(1,1,figsize=(6,4))

freq1,psd1 = signal.welch(shifted * np.blackman(N))
freq2,psd2 = signal.welch(unshifted * np.blackman(N))

ax2.plot(freq1,psd1,label='shifted',lw=0.6)
ax2.plot(freq2,psd2,label='unshifted',lw=0.6)
#ax4 = ax.twinx()
#ax4.plot( np.fft.fft(shifted) - np.fft.fft(unshifted) ,lw=0.4)
ax2.legend()







