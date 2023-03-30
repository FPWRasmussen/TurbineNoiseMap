import numpy as np
import cartopy.geodesic as cgeo
import pandas as pd
from convert_wgs_to_utm import convert_wgs_to_utm


def calc_extent(long, lat, dist, geod):
    '''This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    '''
    
    # boundary of wind farm
    bot_left_bound = np.array([np.amin(long),np.amin(lat)]) 
    top_right_bound = np.array([np.amax(long),np.amax(lat)])

    
    dist_cnr = np.sqrt(2*dist**2)
    bot_left = cgeo.Geodesic().direct(points=bot_left_bound,azimuths=225,distances=dist_cnr)[:,0:2][0]
    top_right = cgeo.Geodesic().direct(points=top_right_bound,azimuths=45,distances=dist_cnr)[:,0:2][0]
    
    extent = pd.DataFrame(data = {geod : [bot_left[0], top_right[0], bot_left[1], top_right[1]]})
    
    if geod == "wsg":
        temp = extent[geod].to_numpy().reshape(2,2)
        temp_long, temp_lat, temp_utm = convert_wgs_to_utm(temp[0,:], temp[1,:])
        extent.insert(0, "utm", np.concatenate((temp_long, temp_lat)).flatten())
    
    
    return extent