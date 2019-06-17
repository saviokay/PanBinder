#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 08 03:16:55 2019
@author: saviokay
"""

#Importing Required Dependecies & Frameworks
import dask
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import time

#Calculating Start Time For Evaluation Of Execution Time
t0 = time.time()

#Importing Files With Python Glob Library
M03_Folder = "/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD03/"
M06_Folder = "/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYM06_Read_L2/"
M03_Sorted = sorted(glob.glob(M03_Folder + "MYD03.A2008*"))
M06_Sorted = sorted(glob.glob(M06_Folder + "MYM06_Read_L2.A2008*"))

#Creating Container Arrays For Total Pixel and Cloud Pixel
total_pix = np.zeros((180, 360))
cloud_pix = np.zeros((180, 360))

#Iterating Through Each M0D03 and MOD06_L2 Files
for M03, M06 in zip (M03_Sorted, M06_Sorted):

    M06_Read = xr.open_dataset(M06)['Cloud_Mask_1km'][:,:,:].values #Initial Read For CM
    M06_ReadCM = M06_Read[::3,::3,0]
    M06_Final = (np.array(M06_ReadCM, dtype = "byte") & 0b00000110) >> 1
    M03_ReadLat = xr.open_dataset(M03, drop_variables = "Scan Type")['Latitude'][:,:].values #Initial Read For Latitude
    M03_ReadLon = xr.open_dataset(M03, drop_variables = "Scan Type")['Longitude'][:,:].values #Initial Read For Longitude
    lat = M03_ReadLat[::3,::3]
    lon = M03_ReadLon[::3,::3]

    latint = (lat + 89.5).astype(int).reshape(lat.shape[0]*lat.shape[1]) #Creating Integer Values Of Latitude
    lat_final = np.where(latint > -1, latint, 0) #Converting All Values Less Than -1 To O For NonZero Division Error
    lonint = (lon + 179.5).astype(int).reshape(lon.shape[0]*lon.shape[1]) #Creating Integer Values Of Longitude
    lon_final = np.where(lonint > -1, lonint, 0) #Converting All Values Less Than -1 To O For NonZero Division Error
    #Marking Final Latitude And Longitude For Total Pixel Array With Zip Function
    for i, j in zip(lat_final, lon_final):
        total_pix[i,j] += 1

    M06_FinalNZ = np.nonzero(M06_Final <= 0) #Filtering All Non Zero Values Of Cloud Mask
    row_ind = M06_FinalNZ[0] #Marking Row Indice With Indexing
    column_ind = M06_FinalNZ[1] #Marking Column Indice With Indexing
    #Perfomring List Comprehension for Cloud Latitude and Longitude With Zip Function
    cloud_lon = [lon_final.reshape(M06_Final.shape[0],M06_Final.shape[1])[i,j] for i, j in zip(row_ind, column_ind)]
    cloud_lat = [lat_final.reshape(M06_Final.shape[0],M06_Final.shape[1])[i,j] for i, j in zip(row_ind, column_ind)]

    #Marking Final Cloud Latitude And Longitude For Cloud Pixel Array With Zip Function
    for x, y in zip(cloud_lat, cloud_lon):
        cloud_pix[int(x),int(y)] += 1

#Obtaining Cloud Fraction With Dividing Cloud Pixel Over Total Pixel
cf = cloud_pix/total_pix
cf1 = xr.DataArray(cf) #Creating Xarray DataArray For Cloud Fraction
cf1.to_netcdf("/umbc/xfs1/jianwu/common/MODIS_Aggregation/savioexe/test/1Months_CF_sub3.hdf") #Creating HDF File With XArray to_netcdf() Function

#Obtaining Total Time Of Execution For Analysis
t1 = time.time()
total = (t1-t0)/60
print(total,'minutes')

#Plotting With MatplotLib
plt.figure(figsize=(14,7))
plt.contourf(range(-180,180), range(-90,90), cf, 100, cmap = "jet")
plt.xlabel("Longitude", fontsize = 14)
plt.ylabel("Latitude", fontsize = 14)
plt.title("One Month's Level 3 Cloud Fraction Aggregation Data", fontsize = 16)
plt.colorbar()
plt.savefig("/umbc/xfs1/jianwu/common/MODIS_Aggregation/savioexe/test/1Months_CF_sub3.png")
