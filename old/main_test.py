import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.shapereader import BasicReader
from calc_extent import calc_extent
import elevation
import rasterio
from rasterio.plot import show
import pathlib
from calculate_point_elevation import calculate_point_elevation
from suncalc import solar_position
import datetime
from datetime import datetime
from pyproj import Geod
from scipy.interpolate import interpn


cwd = pathlib.Path().absolute() # current working directory
data = np.loadtxt("map_cord_ostrup.txt")
turbine_cord = data[:,0:2]
extent = calc_extent(data, 300)
map_boundaries = np.array([extent[0:2],extent[2:4]]).T
#### ELEVATION MAP
fp = cwd.joinpath("MAP.tif") # TIF file path
elevation.clip(bounds=map_boundaries.flatten(), output= fp)
img = rasterio.open(fp)
# show(img)
band_of_interest = 1 # Which band are you interested.  1 if there is only one band
contour_data = np.flip(img.read(band_of_interest), axis=0)



######
data_len = np.shape(contour_data)
shadow_map = np.zeros(data_len) # initiate shadow map

longitude_limit =  map_boundaries[:,0] 
latitude_limit = map_boundaries[:,1]
long_axis = np.linspace(longitude_limit[0], longitude_limit[1], data_len[1]) # generate longitude and latitude coordinates for the contour map
lat_axis = np.linspace(latitude_limit[0], latitude_limit[1], data_len[0])

angle1,angle2,dist_lat = Geod(ellps='WGS84').inv(longitude_limit[0], latitude_limit[0], longitude_limit[0] ,latitude_limit[1])
angle1,angle2,dist_long = Geod(ellps='WGS84').inv(longitude_limit[0], latitude_limit[0], longitude_limit[1] ,latitude_limit[0])

lat_axis_m = np.linspace(0,dist_lat, data_len[0])
long_axis_m = np.linspace(0,dist_lat, data_len[1])

hub_elevation = np.zeros(len(turbine_cord[:,0]))
for i, x in enumerate(hub_elevation):
    hub_elevation[i], long_axis, lat_axis = calculate_point_elevation(turbine_cord[i,0], turbine_cord[i,1], map_boundaries, contour_data)
data[:,2] = data[:,2] + hub_elevation


srtm_res = 1/3600 # SRTM30 resolution (approx 1 arc second)

date_string = "2023-01-01 00:00:00"
t = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')


for MM in np.arange(1, 13, 1):
    for DD in np.arange(1,29,1):
        print(t)
        for hh in np.arange(0,24,1):
            for mm in np.arange(0,60,10):
                current_shadow = np.zeros(data_len)

                for n in np.arange(0, len(data),1):
                    turbine_position = data[n,0:3]
                    turbine_position_m = list(data[n,0:3])
                    
                    date_string = f"2023-{MM}-{DD} {hh}:{mm}"
                    print(date_string)
                    t = datetime.strptime(date_string, '%Y-%m-%d %H:%M')
                    solar=solar_position(t, turbine_position[1], turbine_position[0])
                    if solar["altitude"] < 0.075:
                        break
                    
                    direction_vector = [-np.cos(solar["altitude"])*np.sin(solar["azimuth"]), -np.cos(solar["altitude"])*np.cos(solar["azimuth"]), np.sin(solar["altitude"])]
                    
                    turbine_position_m[0] = np.interp(turbine_position[0], long_axis, long_axis_m)
                    turbine_position_m[1] = np.interp(turbine_position[1], lat_axis, lat_axis_m)
                    
                    
                    a = np.zeros(data_len, dtype=object)
                    D = np.zeros(data_len, dtype=object)
                    # contour_data = np.zeros(data_len)
                    for i,lat in enumerate(lat_axis_m):
                        for j,long in enumerate(long_axis_m):
                            A = np.array([long, lat, contour_data[i,j]])
                            # B = np.cross(A-turbine_position_m, direction_vector)
                            left = A-turbine_position_m
                            
                            x = ((left[1] * direction_vector[2]) - (left[2] * direction_vector[1]))
                            y = ((left[2] * direction_vector[0]) - (left[0] * direction_vector[2]))
                            z = ((left[0] * direction_vector[1]) - (left[1] * direction_vector[0]))
                            

                            a[i,j] = [x,y,z]
                            D[i,j] = np.sqrt(np.einsum('i,i', a[i,j], a[i,j]))/np.sqrt(np.einsum('i,i', direction_vector, direction_vector))
                    D = D.astype("float64")
                    masked_array = np.ma.masked_where(D<100,D).mask
                    current_shadow += masked_array
                    
                shadow_map = shadow_map + np.ma.masked_where(current_shadow>0,current_shadow).mask



#%%
from matplotlib import ticker, colors
plt.figure(figsize=[9,7])
imagery = GoogleTiles(style = "satellite")
ax = plt.axes(projection=imagery.crs)
ax.set_extent(extent)
ax.add_image(imagery, 16) # 16
xs, ys = turbine_cord.T

# index = np.where(D == np.min(D))

plt.plot(xs, ys, "x", transform=ccrs.PlateCarree(), color='blue', markersize=12, label = "Wind Turbines")
# plt.plot(long_axis[index[1]], lat_axis[index[0]], "x", transform=ccrs.PlateCarree(), color='red', markersize=12, label = "Shadow")

# plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='red', markersize=12, label = "Houses") # proxy_artist
# plt.plot(0, 0, "-", transform=ccrs.PlateCarree(), color='green', markersize=12, label = "Cities") # proxy_artist


# plt.contourf(long_axis,lat_axis, shadow_map ,5, alpha=.5, levels = [1, 5, 10, 30, 60, 90],  transform=ccrs.PlateCarree())
# plt.colorbar(label=r"Shadow hours [hr]") 

X,Y = np.meshgrid(long_axis, lat_axis)
plt.pcolormesh(X, Y, shadow_map, alpha=.5,  transform=ccrs.PlateCarree(), norm = colors.LogNorm())
plt.colorbar(label=r"Shadow hours [hr]") 
# levels = [33,38,44,49,54],
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
plt.legend()
plt.title('Noise map (max)')
plt.show()
# plt.tight_layout()

