import matplotlib.pyplot as plt
import matplotlib.animation as animation
from funcs.animation_cartesian import animation_cartesian
from funcs.rotating_frame_animation_polar import rotating_frame_animation_polar
   
def animation_generic(ax1_x,ax1_y,ax1_z,ax2_x,ax2_y,n_frames,n_fps,save_as):
    
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(211,projection = 'polar')
    ax2 = plt.subplot(212)

    def animate(j):

#                rotating_frame_animation_polar(j, ax,  x,     y,     z,     x_min, x_max, y_min, y_max, log_or_lin, v_min, v_max):
        plot1 = rotating_frame_animation_polar(j, ax1, ax1_x, ax1_y, ax1_z, 100,     450,   0,     128,   'log',      1e-4,  1e-2)
        plot2 = animation_cartesian(j,ax2,ax2_x,ax2_y,100,450,0.5,y_lims=None)
        return [plot1,plot2]
        
        
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=n_fps, bitrate=5000)

    anim = animation.FuncAnimation(fig,animate,frames=n_frames,interval=200)
    anim.save(save_as +'.mp4', writer=writer,dpi=600)
    
    
