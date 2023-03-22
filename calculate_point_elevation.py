import numpy as np
from scipy.interpolate import interpn

def calculate_point_elevation(longitude, latitude, map_boundaries, contour_data):
    """ 
        Input:
        longitude : longitude of point
        latitude : latitude of point
        map_boundaries : longitude and latitude boundaries for the map
        contour_data : 2D array of contour data
        
        Output:
        point_elevation : elevation of given point
    """
    data_len = np.shape(contour_data)
    
    longitude_limit =  map_boundaries[:,0]
    latitude_limit = map_boundaries[:,1]
    

    long_axis = np.linspace(longitude_limit[0], longitude_limit[1], data_len[1]) # generate longitude and latitude coordinates for the contour map
    lat_axis = np.linspace(latitude_limit[0], latitude_limit[1], data_len[0])
    
    point_elevation = interpn((lat_axis, long_axis), contour_data, [latitude, longitude])
    
    return point_elevation, long_axis, lat_axis
