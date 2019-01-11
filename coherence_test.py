import numpy as np
from scipy import signal, fftpack



    
    
    
#    conv = signal.correlate(y[:,0],y[:,i])
#    plt.plot(conv)
#    print(np.argmax(conv))

conv = signal.correlate(y1,y2)
plt.plot(conv)
np.argmax(conv)
