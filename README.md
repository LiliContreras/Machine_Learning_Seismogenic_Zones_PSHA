# Machine_Learning_Seismogenic_Zones_PSHA
Programs for the project of machine learning-based seismogenic zones for seismic hazard estimations in México

This repository contains a Python-based tool to create seismic zones by applying HDBSCAN algorithm and the estimation of seismic hazard for Mexico City. 
It includes:
- Python scripts
- Input CSV files (Focal mechanism catalog for Mexico, main dataframe with clasified clusters)
- Shapefiles (seismic zones, terrains, etc)

## 📦 Folder Structure
Main: contains both main scripts and data

## 🚀 How to Use

1. **Download or clone the repository**  
   Click on the green `Code` button and choose `Download ZIP`, or use Git
   Decompress repository
   Decompress VS30_MEX.zip
3. **Run the program**
   -Seismogenic_zones_HDBSCAN.py: To build seismic zones
   -Hazard_Mexico_city.py: Estimates seismic hazard for CCUT in Mexico City
