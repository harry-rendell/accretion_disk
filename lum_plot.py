import numpy as np
import matplotlib.pyplot as plt
import pickle



# =============================================================================
# READING DATA
# =============================================================================

#file = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data_small/lumin', 'rb')
##file = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data/lumin', 'rb')
#L = pickle.load(file)

def plot(start,n,m,label):
    time = np.linspace(0,30.7812e-4*n,n)
    plt.plot(time,L[m,:]/np.average(L[m,:]),linewidth=0.7,label=label)
    plt.xlabel(r'Time $(GM/c^3) \times 10^4$')
    plt.ylabel(r'L')
    plt.legend()
    
    

    
plot(239,1024,0,'lower')
plot(239,1024,1,'upper')
plot(239,1024,2,'full')
plt.clf()
plt.savefig("lum_lo.png",dpi = 900,bbox_inches="tight")