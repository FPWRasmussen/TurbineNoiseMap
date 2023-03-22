import numpy as np
import cartopy.geodesic as cgeo



def calc_extent(data,dist):
    '''This function calculates extent of map
    Inputs:
        lat,lon: location in degrees
        dist: dist to edge from centre
    '''
    lon_list, lat_list = data[:,0:2].T

    # boundary of wind farm
    bot_left_bound = np.array([np.amin(lon_list),np.amin(lat_list)]) 
    top_right_bound = np.array([np.amax(lon_list),np.amax(lat_list)])

    
    dist_cnr = np.sqrt(2*dist**2)
    bot_left = cgeo.Geodesic().direct(points=bot_left_bound,azimuths=225,distances=dist_cnr)[:,0:2][0]
    top_right = cgeo.Geodesic().direct(points=top_right_bound,azimuths=45,distances=dist_cnr)[:,0:2][0]
    
    extent = [bot_left[0], top_right[0], bot_left[1], top_right[1]]
    
    return extent