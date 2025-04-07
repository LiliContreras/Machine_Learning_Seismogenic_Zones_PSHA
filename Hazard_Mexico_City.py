#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seismic hazard for Mexico City

Lilibeth Contreras
"""

#Requisits
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import Point
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from pathlib import Path

#-----------------------------------------------------------------------------
#                                       Functions

#Estimate b value with Maximum likelihood method
def calculate_b_mle(magnitudes):
    magnitudes = np.array(magnitudes)  
    magnitudes = magnitudes[magnitudes > 0]  
    
    if len(magnitudes) < 10:  # at least 10 events to compute b value
        return None
    
    M_min = np.min(magnitudes) #Minimum magnitude
    #M_min=4
    mean_M = np.mean(magnitudes) #Mean of magnitudes
    
    if mean_M == M_min:  # in case there's a cero magnitude
        return None
    
    b_value = np.log10(np.e) / (mean_M - M_min) #Obtain b value
    
    if not np.isfinite(b_value) or b_value <= 0:  # Check wrong values
        return None
    
    return b_value #return estimated b value

#Estimate b value with least squared method
def calculate_b_lsq(magnitudes, M_min=4):
    magnitudes = np.array(magnitudes)
    magnitudes = magnitudes[magnitudes >= M_min]  # Filter data by Mmin

    if len(magnitudes) < 10:
        return None  # at least 10 events

    unique_magnitudes, counts = np.unique(magnitudes, return_counts=True)  
    cumulative_counts = np.cumsum(counts[::-1])[::-1]  # cumulative count

    log_N = np.log10(cumulative_counts)

    # Adjust line with lineal regression
    slope, intercept = np.polyfit(unique_magnitudes, log_N, 1)

    b_value = -slope

    return b_value if b_value > 0 else None

def staircasex(x):
    dx_min = np.min(np.diff(x))
    epsilon = dx_min * 0.0001
    
    xs = [x[0]]
    for i in range(len(x) - 1):
        xs.append(x[i] + epsilon)  # Ajuste con epsilon
        xs.append(x[i + 1])        # Valor siguiente sin ajuste
    xs.append(x[-1] + epsilon)    # Agregar el último valor ajustado
    
    return np.array(xs)


def Garcia(Rcld,M,Hd):
    #Jaimes and García 2020
    #  Horizontal
    c1=0.1571
    c2=1.3581
    c3=-1
    c4=-0.0084
    c5=0.0268
    delta=0.0075*10**(0.507*M)
    R=np.sqrt(Rcld**2+delta**2)
    if Hd<=75:
        H=Hd-50
    else:
        H=75-50
    logY=c1+c2*M+c3*np.log(R)+c4*R+c5*H
    sigma=0.7
    return logY,sigma


coeficientes_garcia = {
    0.01:{'b1': 0.1824, 'b2': 1.3569, 'b3': -1.00, 'b4':-0.0084, 'b5': 0.0266, 'tau':0.36, 'phi':0.60, 'sigma': 0.70},
    0.02:{'b1': 0.3380, 'b2': 1.3502, 'b3': -1.00, 'b4':-0.0086, 'b5': 0.0262, 'tau':0.37, 'phi':0.61, 'sigma': 0.71},
    0.06:{'b1': 1.2826, 'b2': 1.2732, 'b3': -1.00, 'b4':-0.0088, 'b5': 0.0263, 'tau':0.47, 'phi':0.69, 'sigma': 0.83},
    0.08:{'b1': 1.5138, 'b2': 1.2680, 'b3': -1.00, 'b4':-0.0088, 'b5': 0.0272, 'tau':0.42, 'phi':0.69, 'sigma': 0.81},
    0.1 :{'b1': 1.5392, 'b2': 1.2800, 'b3': -1.00, 'b4':-0.0085, 'b5': 0.0284, 'tau':0.34, 'phi':0.70, 'sigma': 0.78},
    0.2 :{'b1': 0.4035, 'b2': 1.4283, 'b3': -1.00, 'b4':-0.0081, 'b5': 0.0284, 'tau':0.34, 'phi':0.57, 'sigma': 0.67},
    0.3 :{'b1':-0.7722, 'b2': 1.5597, 'b3': -1.00, 'b4':-0.0075, 'b5': 0.0231, 'tau':0.34, 'phi':0.53, 'sigma': 0.63},
    0.4 :{'b1':-1.4572, 'b2': 1.5975, 'b3': -1.00, 'b4':-0.0063, 'b5': 0.0216, 'tau':0.30, 'phi':0.54, 'sigma': 0.62},
    0.5 :{'b1':-2.0213, 'b2': 1.6378, 'b3': -1.00, 'b4':-0.0055, 'b5': 0.0153, 'tau':0.24, 'phi':0.54, 'sigma': 0.59},
    0.6 :{'b1':-2.3061, 'b2': 1.6297, 'b3': -1.00, 'b4':-0.0048, 'b5': 0.0178, 'tau':0.21, 'phi':0.56, 'sigma': 0.60},
    0.7 :{'b1':-2.5725, 'b2': 1.6332, 'b3': -1.00, 'b4':-0.0043, 'b5': 0.0165, 'tau':0.20, 'phi':0.58, 'sigma': 0.61},
    0.8 :{'b1':-3.0802, 'b2': 1.6927, 'b3': -1.00, 'b4':-0.0043, 'b5': 0.0137, 'tau':0.21, 'phi':0.57, 'sigma': 0.61},
    0.9 :{'b1':-3.5864, 'b2': 1.7458, 'b3': -1.00, 'b4':-0.0040, 'b5': 0.0134, 'tau':0.21, 'phi':0.57, 'sigma': 0.61},
    1   :{'b1':-3.9575, 'b2': 1.7752, 'b3': -1.00, 'b4':-0.0036, 'b5': 0.0123, 'tau':0.19, 'phi':0.57, 'sigma': 0.61},
    2   :{'b1':-6.2968, 'b2': 1.9592, 'b3': -1.00, 'b4':-0.0029, 'b5': 0.0072, 'tau':0.18, 'phi':0.54, 'sigma': 0.57},
    3   :{'b1':-7.5722, 'b2': 2.0386, 'b3': -1.00, 'b4':-0.0021, 'b5': 0.0044, 'tau':0.26, 'phi':0.49, 'sigma': 0.55},
    4   :{'b1':-8.7329, 'b2': 2.1320, 'b3': -1.00, 'b4':-0.0017, 'b5': 0.0046, 'tau':0.22, 'phi':0.49, 'sigma': 0.53},
    5   :{'b1':-9.6803, 'b2': 2.2118, 'b3': -1.00, 'b4':-0.0016, 'b5':-0.0041, 'tau':0.19, 'phi':0.48, 'sigma': 0.51}
}

def Garcia_SA(Rcld, M, Hd, T):
    # Coeficients per period
    coef = coeficientes_garcia[T]
    c1, c2, c3, c4, c5 = coef["b1"], coef["b2"], coef["b3"], coef["b4"], coef["b5"]

    delta = 0.0075 * 10**(0.507 * M)
    R = np.sqrt(Rcld**2 + delta**2)

    if Hd<=75:
        H=Hd-50
    else:
        H=75-50
    logY = c1 + c2 * M + c3 * np.log(R) + c4 *R + c5 * H
    sigma = coef['sigma'] 
    return logY, sigma

def Arroyo(Rrup,M,SS,NS,RS):
    #Arroyo 2011
    c1=-0.9748
    c2=0.1859
    c3=-0.01151
    e2=-0.03387
    e3=-0.2159
    e4=0.1074
    e5=-0.2528
    e6=-0.08017
    Mref=4.5
    Rref=1
    Mh=6.75
    h=1.35
    sigma=0.695
    R=np.sqrt(Rrup**2+h**2)
    Fd=c1*np.log(R/Rref)+c2*(M-Mref)*np.log(R/Rref)+c3*(R-Rref)
    if M<Mh:
        Fm=e2*SS+e3*NS+e4*RS+e5*(M-Mh)+e6*(M-Mh)**2
    else:
        Fm=e2*SS+e3*NS+e4*RS+e5*(M-Mh)
    y=Fd+Fm
    return y,sigma

def calcular_distancia(lat1, lon1, lat2, lon2, z1, z2):
    """Calcula la distancia geodésica entre dos puntos considerando la profundidad"""
    d_xy = geodesic((lat1, lon1), (lat2, lon2)).km  # Distancia en la superficie
    d_z = abs(z1 - z2) #/ 1000  # Convertir de metros a km
    return np.sqrt(d_xy**2 + d_z**2)  # Distancia total en 3D

def dist_distribution(poly, lat_cdmx, lon_cdmx, z_cdmx=0):
    # Grid definition
    minx, miny, maxx, maxy = poly.bounds
    xA = np.linspace(minx, maxx, 20 * nR)
    yA = np.linspace(miny, maxy, 15 * nR)
    XA, YA = np.meshgrid(xA, yA)
    
    #Obtain depth for each point on grid
    puntos_malla = []
    for x, y in zip(XA.flatten(), YA.flatten()):
        punto = Point(x, y)
        if poly.contains(punto):
            # get Z
            z = poly.interpolate(punto).z if hasattr(poly, "z") else 0  
            puntos_malla.append((punto, z))
    
    # Compute distance
    R = np.array([calcular_distancia(p.y, p.x, lat_cdmx, lon_cdmx, z, z_cdmx) for p, z in puntos_malla])

    #Distance distribution
    minR, maxR = R.min(), R.max()
    rBin = np.linspace(minR, maxR, nR + 1)
    rBinCen = 0.5 * (rBin[:-1] + rBin[1:])

    N, _ = np.histogram(R, bins=rBin, density=False)
    intBx = np.trapz(N, rBin[:-1])  
    fdpR = N / intBx if intBx != 0 else np.zeros_like(N)

    FPA = np.cumsum(fdpR * np.diff(rBin))
    rBinLims = staircasex(rBin)
    
    return rBinCen, fdpR, FPA, R, rBin, N,rBinLims

# Estimate bounded Gutenberg Richter for each zone 
def pdf_gr_bounded(a, b, mmin, mmax, dm=0.1):
    #Parameters: a and b values, maximum magnitude, magnitude interval
    
    #Magnitude vectors 
    m = np.arange(0, 10 + dm, dm)
    
    # Constants for G-R relation
    alpha = np.log(10) * a
    beta = np.log(10) * b
    nu = np.exp(alpha - beta * mmin)

    # Probability density function
    fdpm = beta * np.exp(-beta * (m - mmin)) / (1 - np.exp(-beta * (mmax - mmin)))
    #Equation obtained from derivative of cumulative distribution function
    # and normalizing for bounded interval of magnitudes
    fdpm[m < mmin] = 0
    fdpm[m > mmax] = 0

    # Recurrence rate function for bounded magnitudes
    lm = nu * np.exp(-beta * (m - mmin) - np.exp(-beta * (mmax - mmin))) / (1 - np.exp(-beta * (mmax - mmin)))
    lm[m < mmin] = 0
    lm[m > mmax] = 0

    # Cumulative distribution function
    fpam = (1 - np.exp(-beta * (m - mmin))) / (1 - np.exp(-beta * (mmax - mmin)))
    fpam[m < mmin] = 0
    fpam[m > mmax] = 1

    return m, lm, fdpm, fpam
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                  Seismic hazard estimation for Mexico City

print('Loading data ...')

directorio = Path(__file__).parent

# Load Focal Mechanisms data
archivo = 'Main_df.csv'
file = directorio / archivo
df_fm = pd.read_csv(file)  #Read catalog
df_fm = df_fm.drop_duplicates()  #Eliminates duplicated events

#Load seismic zones
final_cluster_file='poligonos_3D.shp' 
shapefile_clusters = directorio/final_cluster_file
zonas_sismicas = gpd.read_file(shapefile_clusters)


# Map of seismic zones and events
fig, ax = plt.subplots(
    subplot_kw={'projection': ccrs.PlateCarree()}, 
    figsize=(12, 10)
)

# Boundaries of the map
lon_min, lon_max = df_fm['longs'].min(), df_fm['longs'].max()
lat_min, lat_max = df_fm['lat'].min(), df_fm['lat'].max()
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Details of map
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

#Plot seismic zones
zonas_sismicas.plot(
    column='grupo_hdbs',  # Columna que define los colores
    cmap='tab20',            # Mapa de colores
    legend=True,             # Mostrar leyenda
    ax=ax,                   # Eje donde se graficará
    edgecolor='blue',        # Color del borde
    linewidth=1,             # Grosor del borde
    alpha=0.5,               # Transparencia
    transform=ccrs.PlateCarree()  # Proyección
)


# PLot details
ax.set_title('HDBSCAN final clusters', fontsize=14)
#cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Depth', shrink=0.8)
plt.legend()
plt.show()

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#Add type of event (Boore and Atkinson)
df_fm['SS']=0
df_fm['NS']=0
df_fm['RS']=0

for i in range(989):
    if df_fm['rake1'][i]>30 and df_fm['rake1'][i]<150:
        df_fm['RS'][i]=1
        df_fm['NS'][i]=0
        df_fm['SS'][i]=0
    elif df_fm['rake1'][i]>-150 and df_fm['rake1'][i]<-30:
        df_fm['RS'][i]=0
        df_fm['NS'][i]=1
        df_fm['SS'][i]=0
    else:
        df_fm['RS'][i]=0
        df_fm['NS'][i]=0
        df_fm['SS'][i]=1
#-----------------------------------------------------------------------------        

#-----------------------------------------------------------------------------
#                  Gutenberg-Richter estimation per zone
#Zones with data
zonas_validas=np.linspace(0,16,17)

print('Gutenberg Richter estimation per zone')

# Estimation of G-R relation per each zone
resultados = [] #to save a,b values, and magnitudes
fig, axes = plt.subplots(3, 3, figsize=(9, 9)) 
axes = axes.flatten()
num_plots = 0

for i, zona in enumerate(zonas_validas):
    print('Zone:',zona)
    df_zona = df_fm[df_fm["grupo_hdbscan"] == zona] #takes data for each cluster
    magnitudes = np.sort(df_zona["mw"]) #orders magnitudes
    
    # EStimates b value with maximum likelihood method
    b_value = calculate_b_mle(magnitudes)
    #b_value = calculate_b_lsq(magnitudes,4)
    if b_value is None:
        continue  
    
    # estimates a value
    M_min = min(magnitudes)
    N_M_min = len(magnitudes[magnitudes >= M_min])
    a_value = np.log10(N_M_min) + b_value * M_min
    M_max = max(magnitudes)
    resultados.append({"Zona": zona, "a": a_value, "b": b_value, "Mmin":M_min,"Mmax":M_max})
    
    # PLot of G-R relation for each zone
    ax = axes[num_plots]
    unique_mags = np.unique(magnitudes)
    N_m = [np.sum(magnitudes >= M) for M in unique_mags]
    
    ax.scatter(unique_mags, np.log10(N_m), label="Data")
    ax.plot(unique_mags, a_value - b_value * unique_mags, 'r--', label=f"logN = {a_value:.2f} - {b_value:.2f}M")
    ax.set_ylabel("log N(M)",fontsize=10)
    ax.set_title(f"Zone {zona}",fontsize=11)
    if num_plots // 3 == 2:  
        ax.set_xlabel("Magnitude M", fontsize=7)
    ax.legend()
    ax.grid()
    
    
    num_plots += 1
    
    # Plot details
    if num_plots == 9 or i == len(zonas_validas) - 1:
        plt.tight_layout()
        fig.tight_layout()
        plt.show()
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()
        num_plots = 0
#-----------------------------------------------------------------------------
 
#-----------------------------------------------------------------------------
#                                      Site
# Site coordenates   
print('Seismic hazard for site: CCUT') 
lat_ccut=19.45038889
lon_ccut = -99.13730556 

# GRid subdivisions
nR = 20 
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                           Distance distribution
print('Distance distribution...')

# Dictionary to save distance distributions per zone
distribuciones_distancia = {}

for zona in zonas_validas:
    print('Zone:',zona)
    #Gets zone geometry for zone
    poly = zonas_sismicas[zonas_sismicas["grupo_hdbs"] == zona].geometry.iloc[0]
    rBinCen,fdpR,FPA,R,rBin,N,distancia=dist_distribution(poly, lat_ccut, lon_ccut,0)
    
    # Save results to dictionary
    distribuciones_distancia[zona] = {"zona": zona,"rBinCen": rBinCen, "fdpR": fdpR, "FPA": FPA, "dist":distancia}
  
    #  **Uncomment to obtain distance distribution figures**
        
    # fig = plt.figure(figsize=(10, 8))
    # gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1.5])  # Ajusta la proporción de ancho de las columnas

    # # Subplot 1: Distance histogram
    # ax1 = fig.add_subplot(gs[0, :3])  
    # ax1.hist(R, bins=nR, color='b', alpha=0.6, label="Histogram")
    # ax1.plot(rBin[:-1], N, 'g', label="Distribution")
    # ax1.set_title(f"Zone {zona} - Distance histogram", fontsize=11)
    # ax1.legend()
    # ax1.grid()

    # # Subplot 2: Probability density function (PDF)
    # ax2 = fig.add_subplot(gs[1, :3])  
    # ax2.plot(rBinCen, fdpR, 'r-o', label="PDF")
    # ax2.set_title("Probability density function", fontsize=11)
    # ax2.legend()
    # ax2.grid()

    # # Subplot 3: Cumulative distribution function (CDF)
    # ax3 = fig.add_subplot(gs[2, :3])  
    # ax3.plot(rBinCen, FPA, 'm-o', label="CDF")
    # ax3.set_xlabel("Distance (km)")
    # ax3.set_title("Cumulative distribution function", fontsize=11)
    # ax3.legend()
    # ax3.grid()

    # # Subplot 4: Map of zone and site
    # ax4 = fig.add_subplot(gs[:, 3], projection=ccrs.Mercator())  
    # ax4.set_extent([-120, -90, 15, 35], crs=ccrs.PlateCarree())  

    # ax4.add_feature(cfeature.LAND, edgecolor='black')
    # ax4.add_feature(cfeature.COASTLINE)
    # ax4.add_feature(cfeature.BORDERS, linestyle=':')
    # ax4.add_feature(cfeature.LAKES, edgecolor='blue')
    # ax4.add_feature(cfeature.RIVERS)

    # ax4.plot(lon_ccut, lat_ccut, 'ro', markersize=5, transform=ccrs.PlateCarree(), label="Site")
    # ax4.set_title("Location Map", fontsize=11)
    # ax4.add_geometries(
    #         [poly],
    #         crs=ccrs.PlateCarree(),
    #         edgecolor='red',
    #         facecolor='none',
    #         linewidth=1.5,
    #         label=f"Zona {zona}"
    #     )
    # ax4.legend(loc="upper left")
    # plt.tight_layout()
    # plt.show()
#------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                    Gutenber-Richter distribution
print('Bounded Gutenberg Richter distribution...')
#Distribution for G-R BOUNDED
distribuciones_GR = {}
for zona in zonas_validas:
    df_zona = df_fm[df_fm["grupo_hdbscan"] == zona]
    for i in range(len(resultados)):
        if zona==resultados[i]['Zona']:
            a_value = resultados[i]['a']
            b_value = resultados[i]['b']
            mmin_value = resultados[i]['Mmin']
            mmax_value = resultados[i]['Mmax']
    
    m, lm, fdpm, fpam = pdf_gr_bounded(a_value, b_value, mmin_value, mmax_value)
    distribuciones_GR[zona] = {"zona":zona, "m": m, "lm": lm, "fdpm": fdpm, "fpam": fpam}
    #  **Uncomment to obtain Gutenberg-Richter distribution figures**
    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # axes[0].plot(m, lm)
    # axes[0].set_yscale("log")
    # axes[0].set_title(f"G-R bounded - Zone {zona}")
    # axes[0].grid()

    # axes[1].plot(m, fdpm)
    # axes[1].set_title("Probability density function")
    # axes[1].grid()

    # axes[2].plot(m, fpam)
    # axes[2].set_title("Cumulative density function")
    # axes[2].grid()

    # plt.xlabel("Magnitude M")
    # plt.show()
    
print('Done')    
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                            Excedence rate estimation
print('Excedence rate estimaton...')
#Magnitude-Distance combinations per zone
g = 981  # cm/s^2
y = np.arange(0.001, 5.0, 0.001)
ystar = y * g #acceleration values

# variables to save results
LYT = np.zeros(len(y)) #Total excedance rate
lyT = np.zeros((len(y), len(zonas_validas))) #excedence rate per zone
vu=np.zeros(len(zonas_validas))

# Iteration for each acceleration value
for iY, lnystar in enumerate(np.log(ystar)):
    #to check distribution of magnitudes and distances for each acceleration
    for iZ, zona in enumerate(zonas_validas):
        lyT[iY, iZ] = 0
        # Indexes where cumutalive probability is between (0, 1)
        iIR = np.argmax(distribuciones_distancia[iZ]['FPA'] > 0)
        iFR = np.argmax(distribuciones_distancia[iZ]['FPA'] > 0.999999)
        rI, rF = distribuciones_distancia[iZ]['rBinCen'][iIR], distribuciones_distancia[iZ]['rBinCen'][iFR]
        rM = np.linspace(rI, rF, 11)  # 10 intervals

        #Extract parameters for the zone
        for i in range(len(resultados)):
            if zona==resultados[i]['Zona']:
                a_value = resultados[i]['a']
                b_value = resultados[i]['b']
                mmin_value = resultados[i]['Mmin']
                mmax_value = resultados[i]['Mmax']
        # Define magnitude intervals
        mI, mF = mmin_value, mmax_value
        mM = np.linspace(mI, mF, 11)  # 10 intervals
        
        #Extraer profundidad zonas
        prof_zonas=zonas_sismicas.geometry[zona]
        profundidades = [p[2] for p in prof_zonas.exterior.coords]
        h= np.mean(profundidades)
        
        #Definir SS, NS, y RS por zona
        test_df=df_fm[df_fm['grupo_hdbscan']==zona]
        num_SS=test_df['SS'].eq(1).sum()
        num_NS=test_df['NS'].eq(1).sum()
        num_RS=test_df['RS'].eq(1).sum()
        if num_SS > num_NS and num_SS > num_RS:
            g_SS=1
            g_NS=0
            g_RS=0
        elif num_NS > num_SS and num_NS > num_RS:
            g_SS=0
            g_NS=1
            g_RS=0
        elif num_RS > num_SS and num_RS > num_NS:
            g_SS=0
            g_NS=0
            g_RS=1
        else:
            g_SS=0
            g_NS=0
            g_RS=0

        # Recurrence rate for the zone (G-R)
        vu[iZ] = 10 ** (a_value - b_value * mmin_value)
        for iR in range(len(rM) - 1):
            rC = 0.5 * (rM[iR] + rM[iR + 1])
            for iM in range(len(mM) - 1):
                mC = 0.5 * (mM[iM] + mM[iM + 1])
                #FOr al combinations of distance and magnitude pairs

                if abs(h)>50:
                    lnPHA, stdv = Arroyo(rC, mC,g_SS,g_NS,g_RS)
                else:
                    lnPHA, stdv = Garcia(rC, mC,h) 

                # Estimation of excedence probability
                P = 1 - norm.cdf(lnystar, loc=lnPHA, scale=stdv)

                # Probability density interpolation
                fpar_interp = interp1d(distribuciones_distancia[iZ]['rBinCen'],distribuciones_distancia[iZ]['FPA'], fill_value="extrapolate")
                fpam_interp = interp1d(distribuciones_GR[iZ]['m'], distribuciones_GR[iZ]['fpam'], fill_value="extrapolate")

                frdr = fpar_interp(rM[iR + 1]) - fpar_interp(rM[iR])
                fmdm = fpam_interp(mM[iM + 1]) - fpam_interp(mM[iM])

                # save contribution of each distance and magnitude
                ly = vu[iZ] * P * frdr * fmdm
                lyT[iY, iZ] += ly

    # add all contributions
    LYT[iY] = np.sum(lyT[iY, :])

#PLot
plt.figure()
plt.semilogy(ystar, LYT, label='Total', linewidth=2)
plt.xlim(1,3000)
plt.grid()
plt.xlabel("PGA (cm/s^2)")
plt.ylabel(r"$\lambda$")

# plot with different colors separated contributions
for iZ in range(len(zonas_validas)):
    #print(iZ)
    plt.semilogy(ystar, lyT[:, iZ], label=f"Zone {iZ}".format(i=iZ))
    plt.xscale('log')
    plt.xlim(1,3000)

plt.legend(loc='right')
plt.show()
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#                            UHS Estimation
print('Uniform hazard spectra estimation for T=0.004 ...')

#Periods
periodos=[0.01,0.02,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]
# Save LYT for each period
lyt_por_periodo = {}

for T in periodos:
    print('Period:',T)
    # variables for each period
    LYT = np.zeros(len(y))  # Total excedence rate
    lyT = np.zeros((len(y), len(zonas_validas)))  # Rate per zone

    # Iteratoin per y* value
    for iY, lnystar in enumerate(np.log(ystar)):
        for iZ, zona in enumerate(zonas_validas):
            lyT[iY, iZ] = 0
            # Indexes where cumutalive probability is between (0, 1)
            iIR = np.argmax(distribuciones_distancia[iZ]['FPA'] > 0)
            iFR = np.argmax(distribuciones_distancia[iZ]['FPA'] > 0.999999)
            rI, rF = distribuciones_distancia[iZ]['rBinCen'][iIR], distribuciones_distancia[iZ]['rBinCen'][iFR]
            rM = np.linspace(rI, rF, 11)  # 10 intervals

            #Extract parameters for the zone
            for i in range(len(resultados)):
                if zona==resultados[i]['Zona']:
                    a_value = resultados[i]['a']
                    b_value = resultados[i]['b']
                    mmin_value = resultados[i]['Mmin']
                    mmax_value = resultados[i]['Mmax']
            # Define magnitude intervals
            mI, mF = mmin_value, mmax_value
            mM = np.linspace(mI, mF, 11)  # 10 intervals
            
            #Extract depth
            prof_zonas=zonas_sismicas.geometry[zona]
            profundidades = [p[2] for p in prof_zonas.exterior.coords]
            h= np.mean(profundidades)
            
            # Recurrence rate for the zone (G-R)
            vu = 10 ** (a_value - b_value * mmin_value)

            
            for iR in range(len(rM) - 1):
                rC = 0.5 * (rM[iR] + rM[iR + 1])
                for iM in range(len(mM) - 1):
                    mC = 0.5 * (mM[iM] + mM[iM + 1])

                    #GMPE 
                    lnPHA, stdv = Garcia_SA(rC, mC, h, T)

                    P = 1 - norm.cdf(lnystar, loc=lnPHA, scale=stdv)
                    fpar_interp = interp1d(distribuciones_distancia[iZ]['rBinCen'],distribuciones_distancia[iZ]['FPA'], fill_value="extrapolate")
                    fpam_interp = interp1d(distribuciones_GR[iZ]['m'], distribuciones_GR[iZ]['fpam'], fill_value="extrapolate")

                    frdr = fpar_interp(rM[iR + 1]) - fpar_interp(rM[iR])
                    fmdm = fpam_interp(mM[iM + 1]) - fpam_interp(mM[iM])
                    ly = vu * P * frdr * fmdm
                    lyT[iY, iZ] += ly

        # Total contribution for period T
        LYT[iY] = np.sum(lyT[iY, :])

    # Save result for that period 
    lyt_por_periodo[T] = LYT
    LYT_per=lyt_por_periodo[T]

#Plot UHS    
P_ref=0.004
sa_uhs=[]

for T in periodos:
    interp_func=interp1d(y,lyt_por_periodo[T],kind='linear',fill_value='extrapolate')
    sa=interp_func(P_ref)
    sa_uhs.append(sa)
    
sa_arr=np.array(sa_uhs)

plt.figure()
plt.plot(periodos, sa_arr, marker='o', label='UHS')
plt.xlabel("Period (s)")
#plt.xlim(0,1)
plt.ylabel("SA (cm/s²)")
plt.grid()
plt.title("Uniform Hazard Spectra (UHS)")
plt.legend()
plt.show()
