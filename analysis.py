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
    
''' loading lightcurve data '''

def csv_load_lc(name,nu):
    N = len(nu)
    A = np.zeros((1024,N))
    times = pd.read_csv(wdir + 'spec/' + '%.2e.txt'%nu[0],header=None).values[:,0]
    for i in range(N):
        A[:,i] = pd.read_csv(wdir + 'spec/' + '%.2e.txt'%nu[i],header=None).values[:,1]
    
    return times,A

    
def csv_load_mdot(r_index):
    name = 'csv_M_dot'
    return pd.read_csv(wdir + name + '/' + name +'_0%i' %(1262),header=None,delim_whitespace=True).values[:,r_index]
    
    
class light_curve(object):
    
    def __init__(self,times,values):
        self.times = times
        self.values = values

    def plot_series(self,norm='yes'):
        
        y = self.values
        
        if norm == 'yes':
            y = (y-np.mean(y))/np.std(y)
        
        plt.plot(self.times*1e-4,y,lw=0.5)
        plt.xlabel(r'Time, $(GM/c^3) \times 10^4$')
        plt.ylabel(r'Spectral luminosity')

    def coherence(self,lightcurves,n1,n2):
        
        N = len(lightcurves)
        freqs = []
        coheres = []

        fig,ax = plt.subplots(n1,n2)
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            freq,cohere = signal.coherence(self.values,curves.values)
            
            freqs.append(freq)
            coheres.append(cohere)
            
            axes.plot(freq,cohere,lw=0.7,label=r'$\nu = $%.3f'%nu[i+1])
            
            axes.set_xlabel( r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Coherence')
            axes.legend(loc=1)
        
        fig.suptitle(r'Coherence with $\nu = $%.3f'%nu[0])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        return freqs,coheres            
    
    def timelags(self,lightcurves,nperseg,n1,n2):
        
        N = len(lightcurves)
        freqs = []
        time_lags = []
        m = 1024/nperseg

        fig,ax = plt.subplots(n1,n2)
#        if n1*n2 == 1:
#            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            sig1 = np.fft.fft(self.values,n=nperseg)
            sig2 = np.fft.fft(curves.values,n=nperseg)
            
            freq = np.linspace(1.0/times[1],m/(times[-1]),len(sig1)) #this isnt in right units?
            time_lag = (np.angle(sig1*np.conj(sig2))/(2*np.pi*freq))
        
            freqs.append(freq)
            time_lags.append(time_lag)
            
            axes.plot(freq,time_lag,lw=0,marker='.',label=r'$\nu = $%.3f'%nu[i+1])
            axes.set_xlabel(r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Time lag, $(GM/c^3)$')
            axes.legend(loc=1)
            axes.axhline(y=0,color = 'k',lw=0.4,ls='--')
            axes.set_xscale('log')
        
        fig.suptitle(r'Lags with $\nu = $%.3f'%nu[0])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
            
        return freqs,time_lags
    
    def ccf(self,lightcurves,n1,n2):
        
        N = len(lightcurves)
        freqs = []
        ccfs = []

        fig,ax = plt.subplots(n1,n2)
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            ccf = signal.correlate(self.values,curves.values)
            freq = np.linspace(1,len(ccf)+1,len(ccf))
            
            freqs.append(freq)
            ccfs.append(ccf)
            
            axes.plot(freq,ccf,lw=0.7,label=r'$\nu = $%.3f'%nu[i+1])
            
            axes.set_xlabel(r'Time difference $(GM/c^3)$')
            axes.set_ylabel(r'Cross correlation')
            axes.legend(loc=1)
        
        fig.suptitle(r'Correlation with $\nu = $%.3f'%nu[0])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        return freqs,ccfs
            
    ''' PSD '''
    def psd(self,lightcurves,n,n1,n2):
        
        N = len(lightcurves)
        freqs = []
        psds = []
        
        ref_freq,ref_psd = signal.welch(self.values,nperseg=n)
        
        fig,ax = plt.subplots(n1,n2)
        if n1*n2 == 1:
            ax = [ax]
        for axes,i,curves in zip(ax.reshape(-1),range(N),lightcurves):
            
            freq,psd = signal.welch(curves.values,nperseg=n)
            freqs.append(freq)
            psds.append(psd)
            
            axes.loglog(ref_freq,ref_freq*ref_psd,lw=0.7,label=r'$\nu = $%.3f'%nu[0])
            axes.loglog(freq,freq*psd,lw=0.7,label=r'$\nu = $%.3f'%nu[i+1])
            axes.set_xlabel(r'Temporal frequency $(c^3/GM)$')
            axes.set_ylabel(r'Power')
            axes.legend(loc=1)
        
        fig.suptitle(r'Power spectrum density')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
            
        freqs.insert(0,ref_freq)
        psds.insert(0,ref_psd)
        
        return freqs,psds


'''lc0005.txt  lc0050.txt  lc0500.txt  lc1000.txt  lc1600.txt
   lc0011.txt  lc0108.txt  lc0600.txt  lc1200.txt  lc1800.txt
   lc0023.txt  lc0232.txt  lc0800.txt  lc1400.txt  lc2000.txt '''

''' initiate lightcurve classes '''
#nu = [1.4,1.8,2,3.3430,4.5860]
#nu = [0.00147,0.00400,0.00500,0.0110,0.232]

nu = [0.00147,0.232,2,3.3430,4.5860]
times,values = csv_load_lc('csv_ms',nu)

#curve1 = light_curve(times,-csv_load_mdot(0))

def norm(y):
    return y
#    return (y-np.mean(y))/np.std(y)

curve1 = light_curve(times,norm(values[:,0]))
curve2 = light_curve(times,norm(values[:,1]))
curve3 = light_curve(times,norm(values[:,2]))
curve3 = light_curve(times,norm(values[:,2]))
curve4 = light_curve(times,norm(values[:,3]))
curve5 = light_curve(times,norm(values[:,4]))

#curve1.plot_series()
#curve2.plot_series()

#[curve2,curve3,curve4,curve5]

n1=2
n2=2

#%%
''' Time lags between curve1 and others'''
freqs1,time_lags = curve1.timelags([curve2,curve3,curve4,curve5],64,n1,n2) #only seem to recreate significant features with nperseg= 64
#%%
''' Coherence between curve1 and others '''
freqs2,coheres = curve1.coherence([curve2,curve3,curve4,curve5],n1,n2)
#%%
''' CCFs between curve 1 and others '''
freqs3,ccfs = curve1.ccf([curve2,curve3,curve4,curve5],n1,n2)
#%%
''' PSDS '''
freqs4,psds = curve1.psd([curve2,curve3,curve4,curve5],1024,n1,n2)
#%%








