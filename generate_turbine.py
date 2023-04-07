import numpy as np
import matplotlib.pyplot as plt


def generate_turbine(r_list, n_angle, n_vector, n, turbines):
    
    turbine_cord = [turbines.longitude["utm"][n], turbines.latitude["utm"][n], turbines.height[n]+turbines.elevation[n]]
    
    iteration = 0
    rotor_angle = np.arctan(n_vector[0]/n_vector[1])+np.pi/2
    
    points = np.zeros([sum(n_angle).astype(int),3]) # initiate result point (1 extra for center point)
    
    for i, r in enumerate(r_list):
        angle_list = np.linspace(0, 2*np.pi*(1-1/n_angle[i]), n_angle[i].astype(int))
        for j, angle in enumerate(angle_list):
            x_rel = r * np.cos(angle) * np.cos(rotor_angle)
            y_rel = r * np.cos(angle) * np.sin(rotor_angle)
            z_rel = r * np.sin(angle)
            
            points[iteration,:] = np.array([x_rel, y_rel, z_rel])
            
            iteration += 1
    points += turbine_cord

    return points
