import numpy as np
from scipy.interpolate import interpn

def calculate_point_elevation(longitude, latitude, map_data):
    """ 
        Input:
        longitude : longitude of point
        latitude : latitude of point
        map_boundaries : longitude and latitude boundaries for the map
        contour_data : 2D array of contour data
        
        Output:
        point_elevation : elevation of given point
    """
    point_elevation = np.zeros(len(longitude))
    for i in np.arange(0,len(longitude),1):
        point_elevation[i] = interpn((map_data.latitude["wsg"], map_data.longitude["wsg"]), map_data.contour_map, [latitude[i], longitude[i]])
    
    return point_elevation
