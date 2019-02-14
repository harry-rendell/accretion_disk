import numpy as np
import pandas as pd
import cPickle as pickle 
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from funcs.load_coords import load_coords
from funcs.load_quantity import load_quantity
from funcs.load_quantity_reduced import load_quantity_reduced
from funcs.plot_cartesian import plot_cartesian
from funcs.animation_generic import animation_generic
from spectrum import spectrum
from funcs.animation_polar import animation_polar


wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'


''' use *args to pass self.x1 etc'''
class main():
    
    def __init__(self,start,end):
        fileObject = open(wdir + 'coords_24_486')
        coords = pickle.load(fileObject)
        fileObject.close()
        self.x1, self.x2, self.x3, self.C = coords
        
        self.start = start
        self.end = end
        self.t = np.linspace(0,end-start,end-start+1)*30.7812    
        
    def csv_load(self,name):
        self.q_rp = np.zeros((462,128,self.end-self.start+1))
        for i in range(self.end-self.start+1):
            self.q_rp[:,:,i] = pd.read_csv(wdir + name + '/' + name +'_0%i' %(i+start),header=None).values

    def doppler(self,i):
        r = self.x1
        theta = (self.x3+np.pi) # *6 propagates the disc around 2pi, makes the orbital modulation frequency much cleaner! However, velocities now *6
        a = ( (r / (r-3)) ** 0.5 )[:,np.newaxis]
        b = np.outer( (r-2)**(-0.5),np.cos(theta)*np.sin(i) )
        self.z = a*(1+b)
               
    def BB_spectrum(self,nu,R_min=0,R_max=462):
        self.spec = spectrum(self,nu,R_min,R_max).sum((1,2)).transpose(1,0)

    def spectrogram_(self):
        x = abs(self.q_rp).sum((1,2))
        fig,ax = plt.subplots(1, 1, figsize=(6,4))
        freqs, times, sig = signal.spectrogram(x)
        ax.pcolormesh(freqs,times,sig)
    
    def luminosity(self,R_min=0,R_max=592):
        a1 = 0.004335413952759297 #dimensionless
        
        L_shift_ = (abs( (self.q_rp * (self.z[:,:,np.newaxis]**-4)) .sum(1))[R_min:R_max] * (a1 * self.x1 ** 1.5 * self.C)[R_min:R_max,np.newaxis]).sum(0)
        L_       = (abs(  self.q_rp.sum(1))                            [R_min:R_max] * (a1 * self.x1 ** 1.5 * self.C)[R_min:R_max,np.newaxis]).sum(0)
      
        self.L = (L_-np.mean(L_))/np.mean(L_)
        self.L_shift = (L_shift_-np.mean(L_shift_))/np.mean(L_shift_)
        
#        ax1.plot(data.t,y0[:,0],lw=0.5,label = r'$\nu = $%.2f'%0)

#%%
data = main(239,1262)
data.csv_load('csv_ms')

#%%
''' spectra from different annuli '''
nu = np.array([0.5])

spacing = 30
start = 150
radii = range(start,start+9*spacing,spacing)
data.BB_spectrum(nu,start,start+spacing)
y0 = data.spec
for i in range(1,9):
    
    data.BB_spectrum(nu,start+i*spacing,start+(i+1)*spacing)
    y0 = np.concatenate((y0,data.spec),axis=1)
#data.BB_spectrum(nu,100+i*40,140*i*40)
#data.spec

#665,685
#1262
#nu = np.logspace(10**(-0.45),10**(0.04),5) #above max

#%%
'''PLOT BB'''
N = 9
nu = np.linspace(0.05,1.1,N)
y = abs(spectrum(data,nu,0,326).sum((1,2)).transpose(1,0))

#for i in range(N):
#    y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.mean(y[:,i])

fig,ax = plt.subplots(1, 1, figsize=(6,4))

'''subtract the downward trend, assuming it is linear'''
for i in range(N):
    a,b = np.polyfit(data.t,y[:,i],1)
#    ax.plot(data.t,a*data.t+b,lw=0.3,label = nu[i])
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
x = spectrum(data,np.array([0.35]),0,462)
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
#nu = np.logspace(np.log10(0.1),np.log10(2),5)
#nu = np.array([0.12,0.34])
def coherence(self,nu):
#    N=11
#    nu = np.linspace(10**(-0.45),10**(0.04),N) #above max
#    nu = np.linspace(10**(-0.45),10**(-0.2),N) #above max
#    y = self.spec
    y=y0
    N = len(nu)
    peak = 1
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

    
coherence(data,range(9))

#%%
''' correlation across radii '''

fig,ax = plt.subplots(3,3)

fig.suptitle(r'Cross correlation between non overlapping annuli, $\nu = 0.5$')
fig.tight_layout()
fig.subplots_adjust(top=0.88)

for i in range(9):

    j = (i / 3,i % 3)
    
    times = data.t
    full_times = np.linspace(-times[-1],times[-1],2*len(times)-1)
    result = signal.correlate(y0[:,4],y0[:,i])
    ax[j].plot(full_times,result,lw = 0.6,color = 'b',label = r'%i$r_g$'%data.x1[radii[i]])
    ax[j].legend(loc=1)
    ax[j].plot(full_times[np.argmax(result)],np.max(result),'.',color='r')
    ax[j].set_xlabel(r'Time lag, $(GM/c^3 \times 10^4)$'); ax[j].set_ylabel(r'Cross Correlation');


    print full_times[np.argmax(result)],np.max(result)
    
#%% 
''' PSDs '''
#nu = np.logspace(np.log10(0.01),np.log10(2),10)
nu = np.concatenate((np.logspace(np.log10(0.005),np.log10(0.5),7),np.linspace(0.6,2,8)))
data.BB_spectrum(nu)
y = data.spec
#for i in range(5):
#    np.savetxt('%.3f.txt'%nu[i],np.concatenate((t[:,np.newaxis],y[:,i][:,np.newaxis]),axis=1),delimiter = ',')
def psd(ax1,ax2,y):
    N = len(nu)

#    y = (y-np.mean(y,axis=0))/np.std(y,axis=0)

    for i in range(N):
        a1,a2 = signal.welch(y[:,i],nperseg=256)
#        a2 = (a2-np.mean(a2))/np.std(a2)
        ax1.loglog(a1,a2,label=r'$\nu = $%.2f'%nu[i],lw=0.4,color=plt.cm.jet(30*(N-i)))
        ax2.plot(y[:,i],label=r'$\nu = $%.2f'%nu[i],lw=0.4,color=plt.cm.jet(30*(N-i)))
        ax1.legend()
        
fig1,ax1 = plt.subplots(1, 1, figsize=(12,8))
fig2,ax2 = plt.subplots(1, 1, figsize=(12,8))

ax1.set_xlabel(r'Temporal frequency ($c^3/GM$)')
ax1.set_ylabel(r'Power')
ax1.set_title(r'PSDs of $L_{\nu}$ for various $\nu$')

psd(ax1,ax2,data.spec)


#plt.savefig("HLR.pdf",dpi = 300,bbox_inches="tight")

#%%
''' PSDs CSV'''

nu = np.array([1,1.2,1.4,1.6,1.8,2])
#data.BB_spectrum(nu)
#y = data.spec
#for i in range(3):
#    np.savetxt('%.3f.ascii'%nu[i],y[:,i])
''' Bending power law '''
def P(nu,nu_bend,a_low,a_high,norm):
    return norm*nu**(a_low)/(1+(nu/nu_bend)**(a_low-a_high))

def psd(ax1,ax2,n):
    N = len(nu)

    for i in range(N):
        
        y = pd.read_csv(wdir + 'spec/' + '%.3f.txt'%nu[i],header=None).values[:,1]
        y = (y-np.mean(y))/np.std(y)
        a1,a2 = signal.welch(y,nperseg=n)
        
#        popt,pcov = scipy.optimize.curve_fit(P,a1,a2,p0=[0.1,1,1,1])
#        print(popt)
#        ax1.plot(a1,P(a1,popt[0],popt[1],popt[2],popt[3]),label=r'$\nu = $%.3f'%nu[i],lw=0.4,color=plt.cm.jet((N-i)/(1.0*N)))
        ax1.plot(a1,a1*a2,label=r'$\nu = $%.3f'%nu[i],lw=0.4,color=plt.cm.jet((N-i)/(1.0*N)))
        
        
        ax2.plot(data.t*1e-4,y,label=r'$\nu = $%.3f'%nu[i],lw=0.4,color=plt.cm.jet((N-i)/(1.0*N)))
        ax1.legend()
        ax2.legend()
        
fig1,ax1 = plt.subplots(1, 1, figsize=(12,8))
fig2,ax2 = plt.subplots(1, 1, figsize=(12,8))

ax1.set_yscale('log')
ax1.set_xscale('log')

ax2.set_xlabel(r'Time $(GM/c^3) \times 10^4$')
ax2.set_ylabel(r'Normalised spectral intensity')
ax2.set_title(r'Spectral light curves')

ax1.set_xlabel(r'Temporal frequency ($c^3/GM$)')
ax1.set_ylabel(r'Power')
ax1.set_title(r'PSDs of $L_{\nu}$ for various $\nu$')

psd(ax1,ax2,216)


#%%


    




#%%
''' RMS relations - basically just looks like BB spectrum as expected. '''
fig1,ax1 = plt.subplots(1, 1, figsize=(12,8))
#fig2,ax2 = plt.subplots(1, 1, figsize=(12,8))
y_rms = np.zeros(15)
y_std = np.zeros(15)
for i in range(15):
    y = pd.read_csv(wdir + 'spec/' + '%.3f.txt'%nu[i],header=None).values[:,1]
    y_rms[i] = ((sum(y**2))/len(y))**0.5
#    y_std[i] = np.std(y)

ax1.loglog(nu,y_rms,marker = '.',lw = 0.7)
ax1.set_xlabel(r'Spectral frequency ($c^3/GM$)')
ax1.set_ylabel('rms Spectral luminosity')
#ax1.set_title(r'')
#ax2.plot(nu,y_std)

#%%
'''half light radius'''
def calculate_half_radius(self,nu,R_max):
    
    a1 = 0.004335413952759297 #dimensionless
    a2 = 0.009126342080710765 #dimensionless
    
    dB = (nu**3)[:,np.newaxis,np.newaxis,np.newaxis] * (np.exp( np.divide.outer(nu,self.dT[:R_max]))-1) ** -1
    
    if R_max <= 462:
        
        B = a1 * (self.x1[np.newaxis,:R_max,np.newaxis,np.newaxis]**2 * dB).sum(2)
    
    elif R_max > 462:
        
        B_inner = a1 * (self.x1[np.newaxis,:462     ,np.newaxis,np.newaxis]**2 * dB[:,:462,:,:]).sum(2)
        B_outer = a2 * (self.x1[np.newaxis,462:R_max,np.newaxis,np.newaxis]**2 * dB[:,462:R_max,:,:]).sum(2)
        B = np.concatenate((B_inner,B_outer),axis=1)
    
    B_cumsum = np.cumsum(B,axis = 1)
    
    L = (abs(self.q_rp[:R_max].sum(1)) * (a1 * self.x1 ** 1.5 * self.C)[:R_max,np.newaxis]).sum(0)
    
    Norm_disc = B_cumsum / L[np.newaxis,np.newaxis,:]
    Norm_freq = B_cumsum / B_cumsum[:,-1,:][:,np.newaxis,:]
    
    return np.mean(Norm_disc,axis=2), np.mean(Norm_freq,axis=2),L,B

nu = np.logspace(np.log10(0.03),np.log10(3),20)
#nu = np.linspace(0.03,2,5)

N = len(nu)
r_max = 462

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
    
    ax1.legend()
    ax2.legend()
    
#    plt.savefig("splitting_long.pdf",dpi = 300,bbox_inches="tight")

B_disc,B_freq,L,B = half_radius(data,nu,r_max)
plot_half_radius(B_disc,B_freq,L)
plt.plot(B)

#%%
''' TIME LAGS '''
#for i in range(N):
#    y[:,i] = (y[:,i]-np.mean(y[:,i]))/np.std(y[:,i])
#fig1,ax1 = plt.subplots(1,1)

nu = np.concatenate((np.logspace(np.log10(0.005),np.log10(0.5),7),np.linspace(0.6,2,8)))[6:]
#data.BB_spectrum(nu)
#y = data.spec

y = np.zeros((1024,9))
for i in range(9):
    
    y[:,i] = pd.read_csv(wdir + 'spec/' + '%.3f.txt'%nu[i],header=None).values[:,1]
#    y[:,i] = (y[:,i]-np.mean(y[:,i]))


def time_lags(fig2,ax2,y,ref,nperseg):

    for i in range(9):
        
        j = (i / 3,i % 3)
        
        sig1 = np.fft.fft(y[:,ref]*np.blackman(1024),n=nperseg)
        sig2 = np.fft.fft(y[:,i]*np.blackman(1024),n=nperseg)
        
#            ax1.plot(data.t,y[:,i],lw=0.5,label = r'$\nu = $%.2f'%i)
#        ax1.legend(loc=1)
#        freqs = np.fft.rfftfreq(len(sig1)*2-1)
        freqs = np.linspace(1,len(sig1)+1,len(sig1))
        time_lag = np.angle(sig1*np.conj(sig2))/freqs

        ax2[j].plot(freqs,time_lag,lw=0,marker='.',ms=1,label = '%.3f'%nu[i])
#        l,m=np.polyfit(freqs,time_lag,1)
#        ax2[j].plot(freqs,l*freqs+m,lw=0.5,label = r'slope = %.5f'%l)
        ax2[j].legend(loc=1)
        ax2[j].set_xlabel('Temporal frequency'); ax2[j].set_ylabel(r'Time lag, $GM/c^3$');
        ax2[j].axhline(y=0,ls='--',lw=0.5)
#            ax2[j].set_ylabel(r'Phase shift, $\Delta \phi$');


    fig2.suptitle(r'Time lags, $\nu = 0.5$')
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.88)

fig1,ax1 = plt.subplots(3,3)
time_lags(fig1,ax1,y,5,32)
#%%
''' TESTING MODELS '''

fig,ax = plt.subplots(1,1,figsize=(9,6))

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

ax.plot(xdata,model1(xdata,popt1[0],popt1[1],popt1[2]),label=r'$T(r) = 1.9 \times r^{-0.64}(1-\sqrt{4.42/r})^{0.75}$',lw = 0.7)
ax.plot(xdata,model2(xdata,popt2[0],popt2[1]),label=r'$T(r) = 0.96 \times r^{-0.48}$', lw = 0.7)
ax.plot(xdata,ydata,label='data',lw = 0.7)
ax.legend()

print popt1
print popt2

ax.set_xlabel(r'Radius $r / r_g$')
ax.set_ylabel(r'Temperature')
ax.set_title(r'Testing models')


#%%
''' EFFECT OF REDSHIFT TO T(R)'''
fig,ax = plt.subplots(2,2,figsize=(9,6))
incs = np.array([0,np.pi/6,np.pi/3,np.pi/2])
degs = np.array([0,30,60,90])
for i in range(4):
    
    j = (i / 2,i % 2)
    
    data.doppler(incs[i])
    shifted = np.mean(data.dT / data.z[:,:,np.newaxis],axis=1)[:,0]
    unshifted = np.mean(data.dT,axis=1)[:,0]
    ax[j].set_title(r'$i = $%i$^\circ$'%degs[i])
    ax[j].plot(data.x1,shifted,label='shifted',lw=0.7)
    ax[j].plot(data.x1,unshifted,label='unshifted',lw=0.7)
    ax[j].set_xlabel(r'Radius, $r_g$'); ax[j].set_ylabel(r'Temperature');
    ax[j].legend()

fig.suptitle(r'Effect of redshift on temperature')
fig.tight_layout()
fig.subplots_adjust(top=0.88)

#%%
''' EFFECT OF REDSHIFT TO L(t) '''

data.doppler(0)
data.luminosity(200,400)

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