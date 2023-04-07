import numpy as np

def rotor_point_spacing(diameter, grid_resolution, angle):
    
    spacing = np.tan(angle)*grid_resolution*2
    n_radius = np.ceil(diameter/(2*spacing)).astype(int)
    r_list = np.linspace(0, diameter/2, n_radius)
    
    n_list = np.ones(r_list.shape)
    
    for i in np.arange(1, len(n_list)):
        points_per_radius = np.ceil(2*r_list[i]*np.pi/spacing).astype(int)
        n_list[i] = points_per_radius
    

    return r_list, n_list