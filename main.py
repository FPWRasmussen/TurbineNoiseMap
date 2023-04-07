import matplotlib.pyplot as plt
import numpy.ma as ma
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
from shadow_map_multiprocessing import shadow_map_multiprocessing
from pyproj import Proj, transform

cwd = pathlib.Path().absolute() # current working directory
geod = "wsg"

turbines = turbine_data("map_cord_test.txt", geod)
radius = 2000
map_data = contour_map("test_site", radius, turbines)

map_data.scale_array(0.5, "wsg")

shadow_map = np.zeros(map_data.shape) # initiate shadow map

turbines.calc_elevation(map_data)

times = pd.date_range("2023-01-01 00:00:00", "2023-12-31 23:59:59", freq="1min")
processes = 12
#%%
calculate_new_shadow_map = False
if calculate_new_shadow_map:
    shadow_map = shadow_map_multiprocessing(processes, times, map_data, turbines)
else:
    shadow_map = np.loadtxt("temp/shadow_map_temp.txt")

#%%



import scipy.interpolate as si





plt.figure(figsize=[9,7])
imagery = GoogleTiles(style = "satellite")
ax = plt.axes(projection=imagery.crs)
ax.set_extent(map_data.map_boundaries["wsg"])
ax.add_image(imagery, 16) # 16
xs, ys = turbines.longitude["wsg"], turbines.latitude["wsg"]
plt.plot(xs, ys, "x", transform=ccrs.PlateCarree(), color='blue', markersize=12, label = "Wind Turbines")
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='red', markersize=12, label = "Houses") # proxy_artist
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='green', markersize=12, label = "Cities") # proxy_artist

shadow_mask = ma.masked_where(shadow_map <= 0, shadow_map)
cmap = plt.get_cmap('jet')
highest_factor_10 = np.ceil(np.log10(np.amax(shadow_map)))
norm = BoundaryNorm(np.logspace((highest_factor_10-3), highest_factor_10, num = 10), ncolors=cmap.N, clip=False, extend="both")
X,Y = np.meshgrid(map_data.longitude["wsg"], map_data.latitude["wsg"])
plt.pcolormesh(X, Y, shadow_mask, alpha=.5,  transform=ccrs.PlateCarree(), norm = norm, cmap = cmap)
# CS  = plt.contourf(X, Y, shadow_mask, alpha=.5,  transform=ccrs.PlateCarree(), norm = norm, cmap = cmap)
plt.colorbar(label=r"Shadow hours [hr]") 

Xflat, Yflat, Zflat = X.flatten(), Y.flatten(), shadow_map.flatten()

x0, x1, y0, y1 = ax.get_extent()
def fmt(x, y):
    a = np.linspace(x0,x1,map_data.shape[1]) 
    X = min(range(len(a)), key=lambda i: abs(a[i]-x))
    b = np.linspace(y0,y1,map_data.shape[0])
    Y  = min(range(len(b)), key=lambda j: abs(b[j]-y))
    z = shadow_map[Y, X]
    
    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')
    x,y = transform(inProj,outProj,x,y)
    return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
plt.gca().format_coord = fmt



gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
plt.legend()
plt.title(f'Shadow Map ({times[0]} - {times[-1]}')

plt.show()

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# plt.savefig("results/shadow_map_"+date+".png")

#%%

np.savetxt("results/shadow_map_"+date+".txt", shadow_map, delimiter=',')




