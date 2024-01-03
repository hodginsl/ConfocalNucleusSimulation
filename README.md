## Nucleus_w_condensates.py

This file contains a Monte Carlo simulation of the 3-dimensional diffusion of fluorescent particles exploring a nucleus. The simulation also computes confocal images of the central cross-section of the nucleus at continuous time points.

Particles slower and brighter than the population representing single fluorescent particles are placed in the flocal plane which represent biomolecular condensates. These particles diffuse through the nucleus but are not allowed to cross the nuclear membrane.

A DxDxT (D=120 pixels, T=10 frames) array is saved. This array contains the pixel values of the confocal image series.

## plotting_program.py

This file contains the code for plotting the arrays saved in Nucleus_w_condensates.py as images. 

Edit the foldername and filename to match the directory the arrays are saved in. 
