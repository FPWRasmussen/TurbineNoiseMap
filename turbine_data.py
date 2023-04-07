import numpy as np
from download_contour_data import download_contour_data
from scale_array_func import scale_array_func
import pandas as pd
from convert_wgs_to_utm import convert_wgs_to_utm
from calculate_point_elevation import calculate_point_elevation


class turbine_data():
    
        """
        name : project name
        map_boundaries: Map boundary coordinates
        """
        def __init__(self, file, geod : str):
            data = np.loadtxt("input/"+file, skiprows=1)
            self.geod = geod
            self.longitude = pd.DataFrame(data = {geod : data[:,0]})
            self.latitude = pd.DataFrame(data = {geod : data[:,1]})
            self.height = data[:,2]
            self.rotor_diameter = data[:,3]
            self.noise_level = data[:,4]
            self.quantity = len(data[:,0]) # amount of turbines
            
            if geod == "wsg":
                temp_long, temp_lat, temp_utm = convert_wgs_to_utm(self.longitude[geod], self.latitude[geod])
                self.longitude.insert(0, "utm", temp_long)
                self.latitude.insert(0, "utm", temp_lat)
                
             # IMPORT THIS FUNCTION INTO CLASS
        def calc_elevation(self, map_data):
            self.elevation = calculate_point_elevation(self.longitude["wsg"], self.latitude["wsg"], map_data)

# test7 = turbine_data("map_cord_ostrup.txt", "wsg")