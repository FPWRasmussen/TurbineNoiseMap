import numpy as np
from suncalc import solar_position
from generate_turbine import generate_turbine
import datetime
from rotor_point_spacing import rotor_point_spacing

def shadow_map_calculator(times, map_data, turbines, process):
    """
    INPUTS:
        times : array of times to calculate
        contour_data : 2d array of elevation details
        input_data : array containing turbine details

    """
    shadow_map = np.zeros(map_data.shape)
    # n_angle = np.linspace(1, 60, 6, dtype=int)
    # r_list = np.linspace(0, 60, 6, dtype=int)
    n_vector = np.array([0.5, 0.5])
    srtm_res = np.amin([map_data.longitude_step["utm"], map_data.latitude_step["utm"]]) # srtm_res = 1/3600 # SRTM30 resolution (approx 1 arc second)
    progress = 0.0
    
    for i, t in enumerate(times): # For every time step
        
        if process == 0: # print only time for first process

            current_progress = np.round(((times[i]-times[0]).total_seconds())/((times[-1]-times[0]).total_seconds())*100,1)
            if progress != current_progress:
                progress = current_progress
                print("Status:", current_progress, "%")
            
        current_shadow = np.zeros(map_data.shape) # initiate array to hold shadow for current time step
        for n in range(turbines.quantity): # For every turbine
    
            solar=solar_position(t, turbines.latitude["wsg"][n], turbines.longitude["wsg"][n])
            
            # if solar["altitude"] < np.arctan((turbines.height[n]+turbines.elevation[n]-turbines.rotor_diameter[n]/2-map_data.max_elevation)/map_data.radius):
            #     break
            min_solar_angle = np.deg2rad(5)
            if solar["altitude"] < min_solar_angle:
                break
            
            direction_vector = [-np.cos(solar["altitude"])*np.sin(solar["azimuth"]), 
                                np.cos(solar["altitude"])*np.cos(solar["azimuth"]), 
                                np.sin(solar["altitude"])]
            r_list, n_angle = rotor_point_spacing(turbines.rotor_diameter[n], srtm_res, solar["altitude"])
            # if process == 0: # print only time for first process
            #     print(np.sum(n_angle))
            turbine_points = generate_turbine(r_list, n_angle, n_vector, n, turbines)

            for ii in np.arange(0, len(turbine_points), 1): # For every turbine point
                
                old_long = turbine_points[ii,0] # initiate coordinates
                old_lat = turbine_points[ii,1]
                old_elev = turbine_points[ii,2]
    
                maximum_unit_vector = np.amax(np.abs(direction_vector))
    
                while True:
                    
                    new_long = old_long - direction_vector[0]/maximum_unit_vector*srtm_res
                    new_lat = old_lat - direction_vector[1]/maximum_unit_vector*srtm_res
                    
                    if (map_data.map_boundaries["utm"][0] > new_long or 
                        map_data.map_boundaries["utm"][1] < new_long or 
                        map_data.map_boundaries["utm"][2] > new_lat or 
                        map_data.map_boundaries["utm"][3] < new_lat):
                        break
                    
                    dist = np.sqrt((old_lat-new_lat)**2+(old_long-new_long)**2)
                    new_elev = old_elev - np.tan(solar["altitude"])*dist
                    
                    closest_long = min(enumerate(map_data.longitude["utm"]), key=lambda x: abs(x[1]-new_long))
                    closest_lat = min(enumerate(map_data.latitude["utm"]), key=lambda x: abs(x[1]-new_lat))
                    
                    surface_elevation = map_data.contour_map[closest_lat[0], closest_long[0]]
                    
                    if surface_elevation >= new_elev:
                        current_shadow[closest_lat[0], closest_long[0]] = 1
                        break
                
                    old_long = new_long
                    old_lat = new_lat
                    old_elev = new_elev
                    
                    # it += 1
                    # if it > np.amax(map_data.shape)/2:
                    #     # print("damn")
                    #     break
        shadow_map = shadow_map + current_shadow
    np.savetxt(f"temp/shadow_map_temp{process}.txt", shadow_map)