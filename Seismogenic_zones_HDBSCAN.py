#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application of HDBSCAN algorithm
to define seismogenic zones

Lilibeth Contreras

"""

#Required:
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import rasterio
from rasterio.warp import transform
from rasterio.transform import rowcol
from rasterio.mask import mask
from pathlib import Path


#---------------------------------------------------------------------------
#To close all remaining figures
plt.close('all')
#---------------------------------------------------------------------------

directorio = Path(__file__).parent

#---------------------------------------------------------------------------
# Load Focal Mechanisms data
#directorio = '/home/lili/Documents/Maestria/Tesis/Mecanismos_Focales_Catalogo/'
archivo = "2000-2023.csv" #Catalog
file = directorio / archivo
df = pd.read_csv(file)  #Reag catalog
latitudes = df['lat']
longitudes = df['longs']
df['grupo_hdbscan']=0  #Creates new column to save clusters
df = df.drop_duplicates()  #Eliminates duplicated events

# Load Gravimetry data
#directorio='/home/lili/Documents/Maestria/Tesis/Zonas_sismicas/Datos/'
file='grav_INEGI_2010_Grav.txt'
archivo=directorio/file
grav_data = pd.read_csv(archivo, delim_whitespace=True, names=['longs', 'lat', 'grav'])
grav_data['longs']=-1*grav_data['longs']

# Load terrain data
terrain_file='Terrenos_Tecton.shp'
shapefile = directorio/terrain_file
terrenos = gpd.read_file(shapefile)

# Load no focal mechanisms region
nodata_file='Zonas_sin_datos.shp'
shapefile_nodata = directorio/nodata_file
no_data = gpd.read_file(shapefile_nodata)

# Load b values grid
bfile='b_values.geojson'
bvaluesf=directorio/bfile
gdf_b_values = gpd.read_file(bvaluesf)

#Load vs30 grid mexico
#directorio='/home/lili/Documents/Maestria/Tesis/Zonas_sismicas/Datos/'
file='VS30_MEX.tif'
archivo=directorio/file
with rasterio.open(archivo) as src:
    grid_data = src.read(1)  # reads first band
    transform = src.transform  # coordinates to index
    crs = src.crs
    
    no_data_raster_crs = no_data.to_crs(src.crs)
    # extract raster values
    out_image, out_transform = mask(src, no_data_raster_crs.geometry, crop=True)
    # obtain valid values
    vs30_values = out_image[~np.isnan(out_image)]
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Build KDTree for gravity points
gravedad_coords = grav_data[['longs', 'lat']].values
tree = cKDTree(gravedad_coords)

# Finds nearest point of gravity value to assing to dataframe
puntos_coords = df[['longs', 'lat']].values
distances, indices = tree.query(puntos_coords)

# Assings gravity value to df
df['grav'] = grav_data.iloc[indices]['grav'].values
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#Asign terrain to data
df['geometry'] = df.apply(lambda row: Point(row['longs'], row['lat']), axis=1)
gdf_points = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

terrenos = terrenos.to_crs('EPSG:4326')

gdf_joined = gpd.sjoin(gdf_points, terrenos, how='left', predicate='within')

if gdf_joined.index.duplicated().any():
    print("Warning: Duplicated index in spatial union.")
    print("Keeps first coincidence.")

    # Delete duplicated
    gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]

df['terreno'] = gdf_joined['Terrain'] 
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#Asign vs30 value for each data point
def obtener_valor_grid(x, y, grid_data, transform):
    # convert coordinates to index row and column
    row, col = rowcol(transform, x, y)
    
    # Verify limits inside limits of grid
    if 0 <= row < grid_data.shape[0] and 0 <= col < grid_data.shape[1]:
        return grid_data[row, col]
    else:
        return None  # if point is outside
    
#Asign point
df["vs30"] = df.apply(lambda row: obtener_valor_grid(row["longs"], row["lat"], grid_data, transform), axis=1)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#Normalize data before applying HDBSCAN
def normalize_strike(strike):
    return strike % 360  # range [0, 360)

def normalize_rake(rake):
    return rake % 360  # range [0, 360)

# Normalizing...
df['strike1_new'] = df['strike1'].apply(normalize_strike)
df['rake1_new'] = df['rake1'].apply(normalize_rake)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#                           Seismic zones definition
# Parameters for HDBSCAN
Params = df[['lat','longs','prof','strike1_new', 'dip1', 'rake1_new', 'grav','terreno','vs30']]        # metric='euclidean'

#To complete missing values (in terrain and vs30)
imputer = SimpleImputer(strategy='mean')
Params_imputed = imputer.fit_transform(Params)

#Scaling parameters (rescales data values for easier comparison and to improve performance)
scaler = StandardScaler()
Params_scaled = scaler.fit_transform(Params_imputed)
#Apply hdbscan 
#Prepare algorithm...
hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True, metric='euclidean', min_cluster_size=10)
#Apply to parameters
cluster_labels = hdbscan_cluster.fit_predict(Params_scaled)
#Add clusters to dataframe
df['grupo_hdbscan'] = cluster_labels

#Clusters plot
#Discard noise
df_clusters = df[df['grupo_hdbscan'] != -1]

# Creates 3D figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Data
x = df_clusters['longs']  # Longitude
y = df_clusters['lat']    # Latitude
z = -df_clusters['prof']  # Depth

# Color each cluster
c = df_clusters['grupo_hdbscan']

# Scatter plot 3D
sc = ax.scatter(x, y, z, c=c, cmap='tab20', s=20, edgecolor='k', alpha=0.8)

ax.set_title('Cluster distribution', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Depth (km)')
fig.colorbar(sc, label='Cluster', shrink=0.5, aspect=10)  

plt.show()

# Surface distribution
lon_min, lon_max = gdf_b_values.geometry.bounds.minx.min(), gdf_b_values.geometry.bounds.maxx.max()
lat_min, lat_max = gdf_b_values.geometry.bounds.miny.min(), gdf_b_values.geometry.bounds.maxy.max()

# Cartopy map
fig, ax = plt.subplots(
    subplot_kw={'projection': ccrs.PlateCarree()}, 
    figsize=(12, 10)
)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add map characteristics
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

#b value grid
gdf_b_values.plot(
    column='b_value', 
    cmap='viridis', 
    legend=True, 
    ax=ax, 
    alpha=0.5, 
    edgecolor='none', 
    transform=ccrs.PlateCarree()
)

# clusters
scatter = ax.scatter(
    df_clusters['longs'], df_clusters['lat'], 
    c=df_clusters['grupo_hdbscan'], 
    cmap='tab20', 
    s=20, 
    edgecolor='black', 
    linewidth=0.3, 
    transform=ccrs.PlateCarree()
)


# Agregar título y leyendas
ax.set_title('HDBSCAN clusters with b values grid', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Cluster', shrink=0.8)

plt.show()
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#            Seismic zones definition for no seismic data region
#Defining parameters for no data region
grav_data['geometry'] = grav_data.apply(lambda row: Point(row['longs'], row['lat']), axis=1)
gdf_grav = gpd.GeoDataFrame(grav_data, geometry='geometry', crs='EPSG:4326') 

no_data = no_data.to_crs('EPSG:4326')

# Spatial union
gdf_grav_en_region = gpd.sjoin(gdf_grav, no_data, how='inner', predicate='within')

gdf_grav_en_region["vs30"] = gdf_grav_en_region.apply(lambda row: obtener_valor_grid(row["longs"], row["lat"], grid_data, transform), axis=1)

# Defining parameters
Params_nodata = gdf_grav_en_region[['lat','longs','grav','vs30']]       # metric='euclidean'
#Complete missing values
imputer = SimpleImputer(strategy='mean')
Params_imputed_nodata = imputer.fit_transform(Params_nodata)

#Scaling parameters
scaler = StandardScaler()
Params_scaled_nodata = scaler.fit_transform(Params_imputed_nodata)
#Apply hdbscan 
hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True, metric='euclidean', min_cluster_size=100) #50 sale bien
cluster_labels = hdbscan_cluster.fit_predict(Params_scaled_nodata)
gdf_grav_en_region['grupo_hdbscan'] = cluster_labels


#Clusters plot
df_clusters_nodata = gdf_grav_en_region[gdf_grav_en_region['grupo_hdbscan'] != -1]

# get last number of cluster
ultimo_cluster_df1 = df_clusters['grupo_hdbscan'].max()

#re-index clusters for no data region
df_clusters_nodata['grupo_hdbscan'] = df_clusters_nodata['grupo_hdbscan'] + ultimo_cluster_df1 + 1

#Final dataframe
df_final = pd.concat([df_clusters, df_clusters_nodata], ignore_index=True)

#Plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 10))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

sc = ax.scatter(
    df_final['longs'], df_final['lat'], 
    c=df_final['grupo_hdbscan'], cmap='tab20', s=20, label='HDBSCAN clusters'
)

ax.set_title('HDBSCAN final clusters')
plt.legend()
plt.colorbar()
plt.show()

#Save results
df_final.to_csv('seismogenic_zones.csv', index=False)

#The shape was created by loading this file in QGIS
# and applying a concave hull