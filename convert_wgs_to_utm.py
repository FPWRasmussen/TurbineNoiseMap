import numpy as np
from pyproj import Proj


def convert_wgs_to_utm(long: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    
    if isinstance(long, int) or isinstance(long, float):
        long = np.array([long]); lat = np.array([lat])
    
    utm_long = str(int(np.floor((long[0] + 180) / 6) % 60 + 1))
    utm_lat = (np.floor((lat[0] + 80) / 8 ) % 19).astype(int)
    letters = ["C","D","E","F","G","H","J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W"]
    utm_lat = letters[utm_lat]
    
    utm_zone = utm_long + utm_lat
    myProj = Proj("+proj=utm +zone="+utm_zone+"+south, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    conv_long, conv_lat = myProj(long, lat)
    
    conv_cord = np.array([conv_long, conv_lat]).T
    return conv_cord, utm_zone


conv_cord, utm_zone  = convert_wgs_to_utm(12, 55)

