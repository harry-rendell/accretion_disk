#import scipy
from scipy import signal
import numpy as np
import pandas as pd
import cPickle as pickle 
import matplotlib.pyplot as plt

wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/input/'

''' loading data from original simulation '''
class simulation(object):
    
    def __init__(self,start,end):
        fileObject = open(wdir + 'coords_24_486')
        coords = pickle.load(fileObject)
        fileObject.close()
        self.x1, self.x2, self.x3, self.C = coords
        
        self.start = start
        self.end = end
        self.t = np.linspace(0,end-start,end-start+1)*30.7812    
    
    def csv_load_maxwell_stress(self,name):
    
        A = np.zeros((462,128,self.end-self.start+1))
        for i in range(self.end-self.start+1):
            A[:,:,i] = pd.read_csv(wdir + name + '/' + name +'_0%i' %(i+self.start),header=None).values
        
        return A
    
''' Lightcurve class '''
class light_curve(object):
    
    def __init__(self,curve_type,nu_or_r,norm='none'):
        
        fileObject = open(wdir + 'coords_24_486')
        coords = pickle.load(fileObject)
        fileObject.close()
        self.x1 = coords[0]
        
        if curve_type == 'ms':
            self.times  = np.linspace(0,1023,1024)*30.7812
            self.values = pd.read_csv(wdir + 'spec/' + '%.2e.txt'%nu_or_r,header=None).values[:,1]
            self.label  = r'$\nu = $%.2f'%nu_or_r
            self.nu     = nu_or_r
            
        if curve_type == 'mdot':
            name = 'csv_M_dot'
            self.times  = np.linspace(0,1023,1024)*30.7812
            self.values = -pd.read_csv(wdir + name + '/' + name +'_0%i' %(1262),header=None,delim_whitespace=True).values[:,nu_or_r]
            self.label  = r'$r = $%.2f$r_g$'%self.x1[nu_or_r]
            self.radius = nu_or_r
            
        if norm == 'standard_score':
            self.values = (self.values-np.mean(self.values))/np.std(self.values)
        if norm == 'subtract_mean':
            self.values = (self.values-np.mean(self.values))
        if norm == 'fractional':
            self.values = (self.values-np.mean(self.values))/np.mean(self.values)
        if norm == 'subtract_lintrend':
            a,b = np.polyfit(self.times,self.values,1)
            self.values = self.values - a*self.times + a*self.times[len(self.times)/2]
#           ax.plot(data.t,a*data.t+b,lw=0.3,label = nu[i])

    def plot_series(self,color):
        plt.plot(self.times*1e-4,self.values,lw=0.5,label=self.label,color = color)
        plt.xlabel(r'Time, $(GM/c^3) \times 10^4$')
        plt.ylabel(r'Spectral luminosity')
        plt.legend()
    
    ''' FREQUENCY DEPENDENT COHERENCE '''
    def coherence(self,lightcurves,n1,n2,save='no'):
        N = len(lightcurves)
        freqs = []
        coheres = []
        fig,ax = plt.subplots(n1,n2,figsize=(12,8))
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):            
            freq,cohere = signal.coherence(self.values,curves.values)            
            freqs.append(freq)
            coheres.append(cohere)            

            axes.plot(freq,cohere,lw=0.7,label=curves.label)            
            axes.set_xscale('log')
            axes.set_xlabel( r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Coherence')
            axes.legend(loc=3)
            axes.set_ylim([0,1])
        
        fig.suptitle(r'Coherence with ' + self.label)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        if save == 'yes':
            fig.savefig(r'Coherence with ' + self.label + '.pdf',dpi = 300,bbox_inches="tight")

        return freqs,coheres            
    
    ''' FREQUENCY DEPENDENT TIMELAGS '''
    def timelags(self,lightcurves,nperseg,n1,n2,save='no'):
        N = len(lightcurves)
        freqs = []
        time_lags = []
        m = 1024/nperseg
        fig,ax = plt.subplots(n1,n2,figsize=(12,8))
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            sig1 = np.fft.fft(self.values,n=nperseg)
            sig2 = np.fft.fft(curves.values,n=nperseg)
            
            freq = np.linspace(1.0/self.times[1],m/(self.times[-1]),len(sig1)) #this isnt in right units?
            time_lag = (np.angle(sig1*np.conj(sig2))/(2*np.pi*freq))
        
            freqs.append(freq)
            time_lags.append(time_lag)
            
            axes.plot(freq,time_lag,lw=0,marker='.',label=curves.label)
            axes.set_xlabel(r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Time lag, $(GM/c^3)$')
            axes.legend(loc=1)
            axes.axhline(y=0,color = 'k',lw=0.4,ls='--')
            axes.set_xscale('log')
        
        fig.suptitle(r'Lags with ' + self.label)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        if save == 'yes':
            fig.savefig(r'Lags with ' + self.label + '.pdf',dpi = 300,bbox_inches="tight")
            
        return freqs,time_lags
    
    ''' CROSS CORRELATION FUNCTION '''
    def ccf(self,lightcurves,mode='full',window=0,n1=1,n2=1,save='no'):
        N = len(lightcurves)
        ts = []
        ccfs = []
        fig,ax = plt.subplots(n1,n2,figsize=(12,8))
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            w = window
            
            if mode == 'valid':
                ccf = signal.correlate(self.values,curves.values[w:-w],mode='valid')
                t = (np.linspace(-self.times[-1],self.times[-1],2*len(self.times)-1)*1e-4)[(1024-w):(1024+w+1)]
                
            if mode == 'full':
                ccf = signal.correlate(self.values,curves.values,mode='full')
                t = (np.linspace(-self.times[-1],self.times[-1],2*len(self.times)-1)*1e-4)
            
            ts.append(t)
            ccfs.append(ccf)
            print 'ccf timelag: ', t[np.argmax(ccf)] #put this in if statement and restrict np.max to window?
            
            axes.plot(t,ccf,lw=0.7,label=curves.label)
            axes.plot(t[np.argmax(ccf)],np.max(ccf),'.')
            axes.set_xlabel(r'Time difference $(GM/c^3)$')
            axes.set_ylabel(r'Cross correlation')
            axes.legend(loc=1)
        
        fig.suptitle(r'Correlation with ' + self.label)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        if save == 'yes':
            fig.savefig(r'Correlation with ' + self.label + '.pdf',dpi = 300,bbox_inches="tight")
            
        return ts,ccfs
            
    ''' POWER SPECTRUM DENSITY '''
    def psd(self,lightcurves,n,n1,n2,save='no'):
        
        N = len(lightcurves)
        freqs = []
        psds = []        
        ref_freq,ref_psd = signal.welch(self.values,nperseg=n)
        fig,ax = plt.subplots(n1,n2,figsize=(12,8))
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            freq,psd = signal.welch(curves.values,nperseg=n)
            freqs.append(freq)
            psds.append(psd)
    
            axes.loglog(ref_freq,ref_freq*ref_psd,lw=0.7,label= self.label)
            axes.loglog(freq,freq*psd,lw=0.7,label=curves.label)
            axes.set_xlabel(r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Power')
            axes.legend(loc=3)
        
        fig.suptitle(r'Power spectrum density')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
            
        freqs.insert(0,ref_freq)
        psds.insert(0,ref_psd)
        
        if save == 'yes':
            fig.savefig(r'PSD, ref_nu = ' + self.label + '.pdf',dpi = 300,bbox_inches="tight")
        
        return freqs,psds

'''lc0005.txt  lc0050.txt  lc0500.txt  lc1000.txt  lc1600.txt
   lc0011.txt  lc0108.txt  lc0600.txt  lc1200.txt  lc1800.txt
   lc0023.txt  lc0232.txt  lc0800.txt  lc1400.txt  lc2000.txt '''

#%%
''' n1 x n2 subplots '''
n1=2
n2=2

''' normalisation method '''
#normalise_method = 'none'
normalise_method = 'standard_score'
#normalise_method = 'fractional'
#normalise_method = 'subtract_mean'
#normalise_method = 'subtract_lintrend'

''' save? '''
save = 'no'
#%%
''' X-ray curves '''
radii = [0,300,200,100,0]
curve1 = light_curve('mdot',radii[0],norm=normalise_method)
#curve2 = light_curve('mdot',radii[1],norm=normalise_method)
#curve3 = light_curve('mdot',radii[2],norm=normalise_method)
#curve4 = light_curve('mdot',radii[3],norm=normalise_method)
#curve5 = light_curve('mdot',radii[4],norm=normalise_method)

#%%
''' UV/optical curves '''
nu = [0.232,0.5,1,2,3.3430]
#nu = [0.00147,0.00400,0.00500,0.0110,0.232]
#nu = [0.00147,0.232,2,3.3430,4.5860]
#curve1 = light_curve('ms',nu[0],norm=normalise_method)
curve2 = light_curve('ms',nu[1],norm=normalise_method)
curve3 = light_curve('ms',nu[2],norm=normalise_method)
curve4 = light_curve('ms',nu[3],norm=normalise_method)
curve5 = light_curve('ms',nu[4],norm=normalise_method)

#%%
''' series plots '''
plt.figure()
colors = plt.cm.rainbow_r(np.linspace(0,1,5))
curve1.plot_series(colors[0])
curve2.plot_series(colors[1])
curve3.plot_series(colors[2])
curve4.plot_series(colors[3])
curve5.plot_series(colors[4])

#[curve2,curve3,curve4,curve5]
#%%
''' Time lags between curve1 and others'''
freqs1,time_lags = curve1.timelags([curve2,curve3,curve4,curve5],64,n1,n2,save=save) #only seem to recreate significant features with nperseg= 64
#%%
''' Coherence between curve1 and others '''
freqs2,coheres = curve1.coherence([curve2,curve3,curve4,curve5],n1,n2,save=save)
#%%
''' CCFs between curve 1 and others '''
freqs3,ccfs = curve1.ccf([curve2,curve3,curve4,curve5],'full',200,n1,n2,save=save) #window = 512 MAX
#%%
''' PSDS '''
freqs4,psds = curve1.psd([curve2,curve3,curve4,curve5],256,n1,n2,save=save)
#%%







