import numpy as np
from suncalc import solar_position
from generate_turbine import generate_turbine

def shadow_map_calculator(times, map_data, turbines, threads):
    """
    INPUTS:
        times : array of times to calculate
        contour_data : 2d array of elevation details
        input_data : array containing turbine details
        

    
    """
    shadow_map = np.zeros(map_data.shape)
    n_angle = np.linspace(1, 60, 6, dtype=int)
    r_list = np.linspace(0, 60, 6, dtype=int)
    n_vector = np.array([0.5, 0.5])
    srtm_res = np.amin([map_data.longitude_step["utm"], map_data.latitude_step["utm"]]) # srtm_res = 1/3600 # SRTM30 resolution (approx 1 arc second)
    
    for i, t in enumerate(times): # For every time step
        print(t)
        current_shadow = np.zeros(map_data.shape) # initiate array to hold shadow for current time step
        for n in np.arange(0, turbines.quantity, 1): # For every turbine
    

            
            solar=solar_position(t, turbines.latitude["wsg"][n], turbines.longitude["wsg"][n])
            
            if solar["altitude"] < 0.075:
                break
            
            direction_vector = [-np.cos(solar["altitude"])*np.sin(solar["azimuth"]), np.cos(solar["altitude"])*np.cos(solar["azimuth"]), np.sin(solar["altitude"])]
            
            turbine_points = generate_turbine(r_list, n_angle, n_vector, n, turbines)
            
            for i in np.arange(0, len(turbine_points), 1): # For every turbine point
                
                old_long = turbine_points[i,0] # initiate coordinates
                old_lat = turbine_points[i,1]
                old_elev = turbine_points[i,2]
    
                maximum_unit_vector = np.amax(np.abs(direction_vector))
    
                
                it = 0 # itarator
                while True:
                    
                    new_long = old_long - direction_vector[0]/maximum_unit_vector*srtm_res
                    new_lat = old_lat - direction_vector[1]/maximum_unit_vector*srtm_res
                    
                    
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
                    
                    it += 1
                    if it > np.amax(map_data.shape)/2:
                        print("damn")
                        break
        shadow_map = shadow_map + current_shadow
    total_hours = (times[-1]-times[0]).total_seconds()/(60*60)
    return shadow_map/len(times)*total_hours