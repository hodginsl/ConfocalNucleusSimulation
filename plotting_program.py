# -*- coding: utf-8 -*-
"""
Program to plot the image arrays generated in Nucleus_w_condensate_sim.py 

"""
#Import Packages

import numpy as np
import matplotlib.pyplot as plt

#Define the folder the image arrays are saved in
foldername = 'C:/Users/Lydia/OneDrive/OneDrive - McMaster University/Documents/Summer Research Project/updated_simulation/'


filename = 'image_array_no_condensate_2_pop.txt'
data_file= foldername + filename

image_intensities_no_condensate = np.loadtxt(   #import the image array without the condensate
    data_file,
    skiprows = 1)

filename = 'image_array_w_condensate_2_pop.txt'
data_file= foldername + filename

image_intensities_w_condensate = np.loadtxt(    #import the image array with the condensate
    data_file,
    skiprows = 1)

#find minimum and maximum intensity values of the image arrays to set the scale of the color bar

minvalue = np.min([np.min(image_intensities_no_condensate), np.min(image_intensities_w_condensate)]) 
maxvalue = np.max([np.max(image_intensities_no_condensate), np.max(image_intensities_w_condensate)])

#plot the image with no condensate

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.axes()

im = plt.imshow(image_intensities_no_condensate, vmin=minvalue, vmax=maxvalue, cmap ='viridis')

plt.xlabel('x (pixels)', fontsize=14)
plt.ylabel('y (pixels)', fontsize=14)
plt.title(' C_in = 100 nM \n z = 60, no condensate')
cbar = fig.colorbar(im, ax= ax)
cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
imagename = 'D_condensate_0.36_image_no_condensate_2_pop.png'
file = foldername + imagename
#plt.savefig(file, dpi=300, bbox_inches='tight')


#plot the image with the condensate

fig = plt.figure(figsize=(4.27,3.2))
ax = plt.axes()

im = plt.imshow(image_intensities_w_condensate, vmin=minvalue, vmax=maxvalue, cmap ='viridis')

plt.xlabel('x (pixels)', fontsize=14)
plt.ylabel('y (pixels)', fontsize=14)
plt.title('C_in = 100 nM \n z = 60, w/ condensate')
cbar = fig.colorbar(im, ax= ax)
cbar.ax.set_ylabel('Intensity (Photons/pixel)', fontsize=14)
imagename = 'D_codensate_0.36_image_w_condensate_2_pop.png'
file = foldername + imagename
#plt.savefig(file, dpi=300, bbox_inches='tight')
