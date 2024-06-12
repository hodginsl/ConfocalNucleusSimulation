## Nucleus_w_condensates.py

This file contains a Monte Carlo simulation of the 3-dimensional diffusion of fluorescent particles exploring a nucleus. The simulation also computes confocal images of the central cross-section of the nucleus at continuous time points.

Particles slower and brighter than the population of single fluorescent particles are placed near the flocal plane to represent biomolecular condensates (diffraction limited, phase separated, clusters of the fluorescent particles). These particles diffuse through the nucleus but are not allowed to cross the nuclear membrane.

The simulation contains two different slow populations (one near immobile and one slowly diffusing). To simulate only one population comment out the sections corresponding to the undesired populations.

Parameters regarding image acquisition, particle mobility and particle concentrations are defined at the beginning of the code. These parameters can be altered to model different situations.

A NxNxT (N=120 pixels, T=10 frames) array is saved. This array contains the pixel values of the confocal image series.

## plotting_program.py

This file contains the code for plotting the arrays saved in Nucleus_w_condensates.py as images. 

Edit the foldername and filename to match the directory the arrays are saved in. 
