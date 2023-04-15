import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap


import matplotlib as mpl
import matplotlib.patches as mpatches
import os

data=pd.read_excel("Demo/demo.xlsx")

lat = np.array(data['latitude'])
lon = np.array(data['longitude'])
pred = np.array(data['RISK2'])

plt.style.use('ggplot')
plt.figure(figsize=(5, 3))

# Initialize the map object, normal coordinates
map1 = Basemap(projection='robin', lat_0=90, lon_0=0,
               resolution='l', area_thresh=1000.0)  

map1.drawcoastlines(linewidth=0.2)  # draw coastline
#map1.drawcountries(linewidth=0.2)  # draw country
map1.drawmapboundary(fill_color='lightgrey')  # ocean color
map1.fillcontinents(color='white', alpha=0.8)  # fill color

map1.drawmeridians(np.arange(0, 360, 60))  # draw meridians
map1.drawparallels(np.arange(-90, 90, 30))  # Draw parallels

#GROUP
#cm = mpl.colors.ListedColormap(['#F90305', '#3D3E92', '#797979'])

#map1.scatter(lon, lat, latlon=True,
           # alpha=1, s=3.3, c=pred, cmap=cm, linewidths=0, marker='s')

#VALUE
colors_list = [(0, 'blue'), (0, 'white'), (1, '#00FF00')]  # color list
cmap = colors.LinearSegmentedColormap.from_list('my_cmap', colors_list)

map1.scatter(lon, lat, latlon=True,
             alpha=1, s=5, c=pred, cmap=cmap, linewidths=0, marker='s')
#plt.colorbar()
plt.colorbar(orientation='horizontal')


plt.show()

#plt.savefig(outfigure, dpi=300)
#plt.close()