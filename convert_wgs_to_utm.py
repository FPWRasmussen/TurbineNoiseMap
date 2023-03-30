import numpy as np
from pyproj import Proj
import pandas as pd

def convert_wgs_to_utm(long, lat):
    """Based on lat and lng, return best utm epsg-code"""

    
    if type(long) == pd.core.series.Series:
       long = long.to_numpy()
    
    if type(lat) == pd.core.series.Series:
       lat = lat.to_numpy()
    
    if long.ndim > 1:
        long = long.flatten()
        
    if lat.ndim > 1:
        lat = lat.flatten()
    
    
    utm_long = str(int(np.floor((long[0] + 180) / 6) % 60 + 1))
    utm_lat = (np.floor((lat[0] + 80) / 8 ) % 19).astype(int)
    letters = ["C","D","E","F","G","H","J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W"]
    utm_lat = letters[utm_lat]
    
    utm_zone = utm_long + utm_lat
    myProj = Proj("+proj=utm +zone="+utm_zone+"+south, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    longitude_utm, latitude_utm = myProj(long, lat)
    

    return longitude_utm, latitude_utm, utm_zone

