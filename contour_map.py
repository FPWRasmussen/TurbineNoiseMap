import numpy as np
from download_contour_data import download_contour_data
from scale_array_func import scale_array_func
import pandas as pd
from convert_wgs_to_utm import convert_wgs_to_utm
from calc_extent import calc_extent



class contour_map():
        """
        name : project name
        map_boundaries: Map boundary coordinates
        """
        def __init__(self, name, radius, turbine_data):
            """
            geod : geodetic reference frame
            map_boundaries: Map boundary coordinates
            """
            self.geod = turbine_data.geod
            self.map_boundaries = calc_extent(turbine_data.longitude[self.geod], turbine_data.latitude[self.geod], radius, self.geod)
            self.name = name
            self.contour_map = download_contour_data(self.map_boundaries["wsg"], name)
            self.shape = np.shape(self.contour_map)
            self.radius = radius
            

            self.longitude = pd.DataFrame(data = {self.geod : np.linspace(self.map_boundaries[self.geod][0], self.map_boundaries[self.geod][1], self.shape[1])})
            self.latitude = pd.DataFrame(data = {self.geod : np.linspace(self.map_boundaries[self.geod][2], self.map_boundaries[self.geod][3], self.shape[0])})
            self.longitude.insert(0, "utm", np.linspace(self.map_boundaries["utm"][0], self.map_boundaries["utm"][1], self.shape[1]))
            self.latitude.insert(0, "utm", np.linspace(self.map_boundaries["utm"][2], self.map_boundaries["utm"][3], self.shape[0]))
            self.longitude_step = pd.DataFrame(data = {"utm" : self.longitude["utm"][1]-self.longitude["utm"][0], "wsg" : self.longitude["wsg"][1]-self.longitude["wsg"][0]}, index=[0])
            self.latitude_step = pd.DataFrame(data = {"utm" : self.latitude["utm"][1]-self.latitude["utm"][0], "wsg" : self.latitude["wsg"][1]-self.latitude["wsg"][0]}, index=[0])
            self.min_elevation = np.amin(self.contour_map)
            self.max_elevation = np.amax(self.contour_map)
        
        def scale_array(self, scaling_factor, geod):
            self.contour_map = scale_array_func(self.contour_map, scaling_factor)
            self.shape = np.shape(self.contour_map)
            
            self.longitude = pd.DataFrame(data = {self.geod : np.linspace(self.map_boundaries[self.geod][0], self.map_boundaries[self.geod][1], self.shape[1])})
            self.latitude = pd.DataFrame(data = {self.geod : np.linspace(self.map_boundaries[self.geod][2], self.map_boundaries[self.geod][3], self.shape[0])})
            self.longitude.insert(0, "utm", np.linspace(self.map_boundaries["utm"][0], self.map_boundaries["utm"][1], self.shape[1]))
            self.latitude.insert(0, "utm", np.linspace(self.map_boundaries["utm"][2], self.map_boundaries["utm"][3], self.shape[0]))
            self.longitude_step = pd.DataFrame(data = {"utm" : self.longitude["utm"][1]-self.longitude["utm"][0], "wsg" : self.longitude["wsg"][1]-self.longitude["wsg"][0]}, index=[0])
            self.latitude_step = pd.DataFrame(data = {"utm" : self.latitude["utm"][1]-self.latitude["utm"][0], "wsg" : self.latitude["wsg"][1]-self.latitude["wsg"][0]}, index=[0])
            self.min_elevation = np.amin(self.contour_map)
            self.max_elevation = np.amax(self.contour_map)