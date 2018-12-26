#loads coordinates from local .dbl file
import pyPLUTO as pp

def load_coords(wdir,r_min=24,r_max=616,t_min=39,t_max=209):
    D = pp.pload(300,w_dir=wdir)
    x1 = D.x1[r_min:r_max]
    x2 = D.x2[t_min:t_max]
    x3 = D.x3
    A = (1.0-2.0/x1)
    B = (1.0-3.0/x1)
    return x1,x2,x3,A,B