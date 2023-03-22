import numpy as np
import matplotlib.pyplot as plt

n_angle = [1, 10, 20, 30, 40, 50, 60]
r_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2]
n_vector = np.array([0.5, 0.5])



def genturbine(r_list, n_angle, n_vector):
    iteration = 0
    rotor_angle = np.arctan(n_vector[0]/n_vector[1])+np.pi/2
    
    points = np.zeros([sum(n_angle),3]) # initiate result point (1 extra for center point)
    
    for i, r in enumerate(r_list):
        angle_list = np.linspace(0, 2*np.pi*(1-1/n_angle[i]), n_angle[i])
        for j, angle in enumerate(angle_list):
            x_rel = r * np.cos(angle) * np.cos(rotor_angle)
            y_rel = r * np.cos(angle) * np.sin(rotor_angle)
            z_rel = r * np.sin(angle)
            
            points[iteration,:] = np.array([x_rel, y_rel, z_rel])
            iteration += 1
    return points
