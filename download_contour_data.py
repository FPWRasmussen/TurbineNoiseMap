import numpy as np
import elevation
import pathlib
import rasterio
import pandas as pd

def download_contour_data(map_boundaries, name):
    cwd = pathlib.Path().absolute() # current working directory
    map_boundaries = map_boundaries.to_numpy()
    fp = cwd.joinpath("temp/"+name+".tif") # TIF file path
    map_boundaries = np.array([map_boundaries[0], map_boundaries[2], map_boundaries[1], map_boundaries[3]])
    elevation.clip(bounds=map_boundaries, output= fp)
    img = rasterio.open(fp)
    band_of_interest = 1 # Which band are you interested.  1 if there is only one band
    contour_map = np.flip(img.read(band_of_interest), axis=0)
    return contour_map