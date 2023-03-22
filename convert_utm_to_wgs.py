import numpy as np
from pyproj import Proj

utm_zone = "33U"
def convert_utm_to_wgs(long: float, lat: float, utm_zone: str):
    """Based on lat and lng, return best utm epsg-code"""
   
    myProj = Proj("+proj=utm +zone="+utm_zone+"+south, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    conv_long, conv_lat = myProj(long, lat, inverse=True)
    return conv_long, conv_lat

conv_long, conv_lat = convert_utm_to_wgs(308124.3678624593, 6098907.825129169, utm_zone)