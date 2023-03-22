import numpy as np
import matplotlib.pyplot as plt



# n_angle = [1, 10, 20, 30, 40, 50, 60]
# r_list = [0, 10, 20, 30, 40, 50, 60]
# n_vector = np.array([0.5, 0.5])
# turbine_cord = np.array([500, 500, 120])

def generate_turbine(r_list, n_angle, n_vector, turbine_cord):
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
    points += turbine_cord

    return points


# R = 30 # rotor diameter [m]
# h = 100 # hub height
# n_vector = np.array([0.5,0.5, 0]) # normal vector to rotor plane (wind direction)

# turbine_cord = np.array([500, 500, 120])

# n_radial = 3 # points in the radial direction (minus to center)
# n_rot = 10 # points in the rotational direction

def old(R, h, n_vector, turbine_cord, n_radial, n_rot):
    """
    Inputs:
        R : Rotor diameter [m]
        h : Hub height [m]
        n_vector : Normal vector to rotor plane (wind direction)
        turbine_cord : Coordinates for turbine hub (3D)
        n_radial : Generated points in radial direction
        n_rot : Generated points in rotational direction
    
    Output:
        points : Rotor plane points in 3D space
    
    
    """
    rotor_angle = np.arctan(n_vector[0]/n_vector[1])+np.pi/2
    r_range = np.linspace(R/n_radial,R,n_radial)
    angle_range = np.linspace(0, 2*np.pi*(1-1/n_rot), n_rot)
    angle_range += np.pi/2
    
    
    points = np.zeros([n_radial*n_rot+1,3]) # initiate result point (1 extra for center point)
    
    for i, r in enumerate(r_range):
        for j, angle in enumerate(angle_range):
            x_rel = r * np.cos(angle) * np.cos(rotor_angle)
            y_rel = r * np.cos(angle) * np.sin(rotor_angle)
            z_rel = r * np.sin(angle)
            
            points[i*len(angle_range)+j,:] = np.array([x_rel, y_rel, z_rel])
    
    points += turbine_cord
    
    return points

# points = generate_turbine(r_list, n_angle, n_vector, turbine_cord)
# x, y, z = points.T

# fig = plt.figure()

# ax = fig.add_subplot(projection='3d')
# ax.scatter(x,y,z)