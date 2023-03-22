import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
# from cartopy.io.shapereader import BasicReader
from calc_extent import calc_extent
import elevation
import rasterio
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
from scale_array import scale_array

cwd = pathlib.Path().absolute() # current working directory
input_data = np.loadtxt("input/map_cord_ostrup.txt", skiprows=1) # array containing turbine details
turbine_data = pd.read_csv("input/map_cord_ostrup.txt", delimiter="\t")

turbine_data["Longitude_utm"], turbine_data["Latitude_utm"], utm_zone = convert_wgs_to_utm(turbine_data.Longitude, turbine_data.Latitude)
#%%

extent = calc_extent(turbine_data.Longitude, turbine_data.Latitude, 2000)
map_boundaries = np.array([extent[0:2],extent[2:4]]).T


#### ELEVATION MAP
fp = cwd.joinpath("temp/MAP.tif") # TIF file path
elevation.clip(bounds=map_boundaries.flatten(), output= fp)
img = rasterio.open(fp)
# show(img)
band_of_interest = 1 # Which band are you interested.  1 if there is only one band
contour_data = np.flip(img.read(band_of_interest), axis=0)
contour_data = scale_array(contour_data, 0.25)


######
data_len = np.shape(contour_data)
shadow_map = np.zeros(data_len) # initiate shadow map

map_boundaries_utm, utm_zone = (convert_wgs_to_utm(map_boundaries[:,0] , map_boundaries[:,1])) # convert map_boundaries

n_angle = np.linspace(1, 60, 6, dtype=int)
r_list = np.linspace(0, 60, 6, dtype=int)
n_vector = np.array([0.5, 0.5])
####

####
#%%
longitude_limit =  map_boundaries[:,0] 
latitude_limit = map_boundaries[:,1]
long_axis = np.linspace(longitude_limit[0], longitude_limit[1], data_len[1]) # generate longitude and latitude coordinates for the contour map
lat_axis = np.linspace(latitude_limit[0], latitude_limit[1], data_len[0])

longitude_limit_utm =  map_boundaries_utm[:,0] 
latitude_limit_utm = map_boundaries_utm[:,1]
long_axis_utm = np.linspace(longitude_limit_utm[0], longitude_limit_utm[1], data_len[1]) # generate longitude and latitude coordinates for the contour map
lat_axis_utm = np.linspace(latitude_limit_utm[0], latitude_limit_utm[1], data_len[0])

hub_elevation = np.zeros(len(turbine_data.index))
for i, x in enumerate(hub_elevation):
    hub_elevation[i], no, yes = calculate_point_elevation(turbine_data.Longitude_utm[i], turbine_data.Latitude_utm[i], map_boundaries_utm, contour_data)
input_data[:,2] = input_data[:,2] + hub_elevation
turbine_data.Height = turbine_data.Height + hub_elevation


#%%
# srtm_res = 1/3600 # SRTM30 resolution (approx 1 arc second)

srtm_res = np.amin([np.abs((longitude_limit_utm[0]-longitude_limit_utm[1])/data_len[1]), np.abs((latitude_limit_utm[0]-latitude_limit_utm[1])/data_len[0])])


# date_string = "2023-01-01 00:00:00"
# t = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')



times = pd.date_range("2023-01-01 00:00:00", "2023-12-31 23:59:59", freq="1min")

for i, t in enumerate(times):
    print(t)
    current_shadow = np.zeros(data_len)
    for n in np.arange(0, len(input_data),1):

        turbine_position = input_data[n,0:3]
        turbine_position_utm = data_utm[n,0:3]
        
        solar=solar_position(t, turbine_position[1], turbine_position[0])
        if solar["altitude"] < 0.075:
            break
        direction_vector = [-np.cos(solar["altitude"])*np.sin(solar["azimuth"]), -np.cos(solar["altitude"])*np.cos(solar["azimuth"]), np.sin(solar["altitude"])]
        
        turbine_points = generate_turbine(r_list, n_angle, n_vector, turbine_position_utm)
        for i in np.arange(0, len(turbine_points),1):
            turbine_position_utm = turbine_points[i,:]
            
            old_long = turbine_position_utm[0] # initiate coordinates
            old_lat = turbine_position_utm[1]
            old_elev = turbine_position_utm[2]

            maximum_unit_vector = np.amax(np.abs(direction_vector))

            
            it = 0 # itarator
            while True:
                
                new_long = old_long - direction_vector[0]/maximum_unit_vector*srtm_res
                new_lat = old_lat - direction_vector[1]/maximum_unit_vector*srtm_res
                
                
                dist = np.sqrt((old_lat-new_lat)**2+(old_long-new_long)**2)
                new_elev = old_elev - np.tan(solar["altitude"])*dist
                
                closest_long = min(enumerate(long_axis_utm), key=lambda x: abs(x[1]-new_long))
                closest_lat = min(enumerate(lat_axis_utm), key=lambda x: abs(x[1]-new_lat))
                
                surface_elevation = contour_data[closest_lat[0], closest_long[0]]
                
                if surface_elevation >= new_elev:
                    current_shadow[closest_lat[0], closest_long[0]] = 1
                    
                    break
            
                old_long = new_long
                old_lat = new_lat
                old_elev = new_elev
                it += 1
                if it > np.amax(np.shape(contour_data))/2:
                    print("damn")
                    break
    shadow_map = shadow_map + current_shadow

shadow_map = shadow_map/len(times)*8760
#%%



plt.figure(figsize=[9,7])
imagery = GoogleTiles(style = "satellite")
ax = plt.axes(projection=imagery.crs)
ax.set_extent(extent)
ax.add_image(imagery, 16) # 16
xs, ys = turbine_cord.T
plt.plot(xs, ys, "x", transform=ccrs.PlateCarree(), color='blue', markersize=12, label = "Wind Turbines")
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='red', markersize=12, label = "Houses") # proxy_artist
plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='green', markersize=12, label = "Cities") # proxy_artist


# plt.contourf(long_axis,lat_axis, shadow_map ,5, levels = [0.5,10,30,50, 400],  alpha=.5,  transform=ccrs.PlateCarree())
# plt.colorbar(label=r"Shadow hours [hr]") 
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm([0, 0.5, 10, 30, 50, 400, 9999], ncolors=cmap.N, clip=False)
X,Y = np.meshgrid(long_axis, lat_axis)
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
date = datetime.now().strftime("%Y%m%d-%H%M%S")
plt.savefig("results/shadow_map_"+date+".png")

#%%

np.savetxt("results/shadow_map_"+date+".txt", shadow_map, delimiter=',')
