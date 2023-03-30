import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
# from cartopy.io.shapereader import BasicReader
from calc_extent import calc_extent
# import elevation
# import rasterio
# from rasterio.plot import show
import pathlib
from calculate_point_elevation import calculate_point_elevation
from suncalc import solar_position
import datetime
# from pyproj import Geod
# from scipy.interpolate import interpn
from generate_turbine import generate_turbine
from convert_wgs_to_utm import convert_wgs_to_utm
# from convert_utm_to_wgs import convert_utm_to_wgs
# from matplotlib import colors
# from datetime import datetime
from contour_map import contour_map
from turbine_data import turbine_data
from shadow_map_calculator import shadow_map_calculator

cwd = pathlib.Path().absolute() # current working directory
geod = "wsg"

turbines = turbine_data("map_cord_ostrup.txt", geod)
radius = 2000
map_data = contour_map("test_site", radius, turbines)

map_data.scale_array(0.2, "wsg")

shadow_map = np.zeros(map_data.shape) # initiate shadow map

turbines.calc_elevation(map_data)

times = pd.date_range("2023-01-01 00:00:00", "2023-01-01 10:59:59", freq="1min")
threads = 2

shadow_map = shadow_map_calculator(times, map_data, turbines, threads)


#%%



plt.figure(figsize=[9,7])
imagery = GoogleTiles(style = "satellite")
ax = plt.axes(projection=imagery.crs)
ax.set_extent(map_data.map_boundaries["wsg"])
ax.add_image(imagery, 16) # 16
xs, ys = turbines.longitude["wsg"], turbines.latitude["wsg"]
plt.plot(xs, ys, "x", transform=ccrs.PlateCarree(), color='blue', markersize=12, label = "Wind Turbines")
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='red', markersize=12, label = "Houses") # proxy_artist
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='green', markersize=12, label = "Cities") # proxy_artist


# plt.contourf(long_axis,lat_axis, shadow_map ,5, levels = [0.5,10,30,50, 400],  alpha=.5,  transform=ccrs.PlateCarree())
# plt.colorbar(label=r"Shadow hours [hr]")
 
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(np.logspace(0, np.ceil(np.log10(np.amax(shadow_map))), num = 6), ncolors=cmap.N, clip=False)
X,Y = np.meshgrid(map_data.longitude["wsg"], map_data.latitude["wsg"])
plt.pcolormesh(X, Y, shadow_map, alpha=.5,  transform=ccrs.PlateCarree(), norm = norm, cmap = cmap)
plt.colorbar(label=r"Shadow hours [hr]") 
# levels = [33,38,44,49,54], norm = colors.LogNorm(),
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
plt.legend()
plt.title('Shadow Map')
# plt.tight_layout()
plt.show()

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
plt.savefig("results/shadow_map_"+date+".png")

#%%

np.savetxt("results/shadow_map_"+date+".txt", shadow_map, delimiter=',')
