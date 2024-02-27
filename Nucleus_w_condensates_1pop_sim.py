# -*- coding: utf-8 -*-
"""
Simulation of the generation of a confocal image series of a nucleus containing fluorescent proteins 
freely diffusing as well as fluorescent condenensates with different diffusion coefficients
"""

#%% Import Packages
import numpy as np
import os

#%% Set up parameters for imaging and simulation

#folder to save images to
foldername = 'rerun_sim_1pop_trial1'

image_size = 120    #px; The simulation space will be a cube with dimensions 120x120x120 pixels
pixel_size = 0.1    #um
boundary = image_size * pixel_size      #um; boundary of simulation space
radius = image_size * pixel_size / 4    #um; radius of nucleus
V_n = (4/3) * np.pi *radius**3          #um^3; volume of nucleus
V_c = (image_size*pixel_size)**3 - V_n  #um^3; volume of cytoplasm
center_pos = [radius*2, radius*2, radius*2] #um; coordinates of the center of the nucleus
dwell_time = 4.1E-6 #s; dwell time of laser during image acquisition
psf_width = 0.3     #um; width of the point spread function in focus
psf_height = 1.5    #um; height of the point spread function
D_protein = 30      #um^2/s; diffusion coefficient of free mobile proteins
D_condensate_b = 0.01  #um^2/s; diffusion coefficient of "bound" condensates
step_time = 4.1E-6  #s; step time of proteins and condensates
B_protein = 30000 # Hz; Brightness of protein
B_background = 45000 #Hz; Brightness of background noise
B_condensate_b = 20*30000 #Hz; Brightness of condensate
z_condensate = 6.0  #um; position of condensates along z-axis
N_condensate_b = 5  #number of bound condensates in simulation
z_slice = 59        #px; position of focal plane along the z-axis
t_slices = 20       #number of frames to acquire in the time series
steps = image_size * image_size * t_slices #number of steps made by free proteins to measure
steps_cond = 2*steps #number of steps made by condensates to measure
C_in = 75           #nM; nuclear protein concentration 

#####
#Calculate the number of particles based on the nuclear concentration, and the nuclear to cytoplasmic ratio is 6
#####

#Number of free and bound proteins in the nucleus based on nuclear concentration
N_f_N_b = (C_in / 1e9) * (V_n / 1e15) * 6.02e23 
#Number of free proteins in cytoplasm
N_c = (C_in / (6 * 1e9)) * (V_c / 1e15) * 6.02e23
#Total number of free proteins
Nparticles = N_f_N_b + N_c 

#Membrane crossing and protein binding rates
pout = 0.2  #flow rate of proteins from nucleus to cytoplasm
pin = 0.8   #flow rate of proteins from cytoplasm to nucleus
pon = 0.005 #rate that proteins switch from unbound to bound
poff = 0.01 #rate that proteins switch from bound to unbound
#Number of unbound proteins
N_f = ((poff/(poff + (poff * pout)/pin * V_c/V_n + pon)) * Nparticles)
#Number of bound proteins
N_b = ((pon/poff) * N_f)
#Number of proteins in cytoplasm
N_c = ((pout/pin) * (V_c/V_n) * N_f)
#round protein numbers to whole numbers
N_f = round(N_f)
N_b = round(N_b)
N_c = round(N_c)
#Final value of total number of particles
Nparticles = N_f + N_b + N_c

#nM; concentration of protein inside the nucleus
C_in = (((N_f+N_b)/6.02e23)/(V_n/1e15))*1e9 
#nM; concentration of protein outside the nucleus
C_out = ((N_c/6.02e23)/(V_c/1e15))*1e9 

#%% Generate initial posistions of proteins and condensates

#Array to contain initial posistions of free proteins
start_pos = np.zeros((Nparticles,3))
#Array to contain state (bound/unbound/cytoplasm) of free proteins
state = np.zeros((Nparticles))

#Place N_b number of proteins in the nucleus and set state to 0 (bound)
for n in range(0,N_b):        
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) <= radius**2: 
            start_pos[n,:] = x
            state[n] = 0

#Place N_f number of proteins in the nucleus and set state to 1 (free)
for n in range(N_b, N_b+N_f):       
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) <= radius**2: 
            start_pos[n,:] = x
            state[n] = 1
         
#Place N_c number of proteins in the cytoplase and set state to 2 (cytoplasm)
for n in range(N_b+N_f, Nparticles): 
    temp = start_pos[n, :]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x=np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) > radius**2 and (x[0]**2 + x[1]**2 + x[2]**2) > 0:
            start_pos[n,:] = x
            state[n] = 2

#Array to contain initial positions of bound condensates
start_pos_cond_b = np.zeros((N_condensate_b,3))

#Place N_condensate_b number of condensates in the nucleus within 0.5 um from the z_condensate plane
for n in range(N_condensate_b):
    temp = start_pos_cond_b[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x=np.random.rand(3)*image_size*pixel_size
        if (((x - center_pos)**2).sum()) <= (radius-0.5)**2 and (x[2] > z_condensate - 0.5 and x[2] < z_condensate + 0.5):
            start_pos_cond_b[n,:] = x
  

#%% Define the function of Gaussian Beam Profile

def GaussianBeam( start_pos, beam_pos, psf_width, psf_height, B):
    
    #Gaussian Beam profile if the proteins are diffusing in 2D space
    if start_pos.shape[0] == 2:
        GB = B*dwell_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2)
        
    #Gaussian Beam profile if the proteims are diffusing in 3D space
    else:
        GB = B*dwell_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB

#%% Generate the trajectories and image array for the free proteins

#Array to contain the pixel values of the confocal images excluding condensates
image_array = np.zeros((image_size,image_size,t_slices))

#For Loop to generate the trajectories of the free proteins
for n in range(Nparticles):
    
    #Set up more parameters for the particles to diffuse
    loca = np.zeros((steps,3))  #array of the location of the protein at step i
    loca[0,:] = np.transpose(start_pos[n,:])    #set the initial location to the starting position
    
    #Generate random steps (direction and size) for protein i according to normal distribution
    track = np.random.normal(loc=0,scale=np.sqrt(2*D_protein*step_time),size=(steps,3)) 
    
    #Let the particles diffuse inside the simulation box
    for i in range(steps-1):
        
        #protein's distance from center of nucleus
        depth = np.sqrt(((loca[i,:]-center_pos)**2).sum()) 
        #protein's distance from center of nucleus after taking a step
        forwd = np.sqrt(((loca[i,:] + track[i,:] - center_pos)**2).sum()) 
        
        if depth <= radius: #if the particle is in the nucleus
            
            if state[n] == 1: #if the particle is free
                
                proba = np.random.rand()
                if proba <= pon: #if random number is less than p_on the particle becomes bound
                    state[n] = 0
                    loca[i+1,:] = loca[i, :]
                    
                else: #if random number is greater than p_on the particle remains free
                    
                    if forwd <= radius: #if the particle remains in the nucleus let it take a step
                        state[n] = 1
                        loca[i+1, :] = loca[i, :] + track [i, :]
                    
                    else: #if the particle moves out of the nucleus let it take a step if random number is less than p_out
                        
                        proba = np.random.rand()
                        if proba <= pout:
                            state[n] = 2
                            loca[i+1, :] = loca[i, :] + track[i, :]
                        
                        else:
                            state[n] = 1
                            loca[i+1, :] = loca[i, :]
            
            else: #if the particle is bound
                
                proba = np.random.rand()
                if proba <= poff: #if random number is less than p_off the particle becomes free
                    
                    if forwd <= radius: #if the particle remains in the nucleus let it take a step
                        state[n] = 1
                        loca[i+1, :] = loca[i, :] + track[i, :]
                        
                    else: #if the particle moves into the cytoplasm let it take a step if random number is less than p_out
                        
                        proba = np.random.rand()
                        if proba <= pout:
                            state[n] = 2
                            loca[i+1, :] = loca[i, :] + track[i, :]
                        
                        else: 
                            state[n] = 1
                            loca[i+1, :] = loca[i, :]
                
                else: #if random number is greater than p_off the particle remains bound.
                    state[n] = 0
                    loca[i+1, :] = loca[i, :]
        
        else: #if the particle is in the cytoplasm
            
            if forwd >= radius: #if the particle remains in the cytoplasm let it take a step so long as it doesn't leave the simulation window
                state[n] = 2
                x  = loca[i,0] + track[i,0]
                y  = loca[i,1] + track[i,1]
                z  = loca[i,2] + track[i,2]

                if  x > boundary or x < 0: 
                     loca[i+1,0] = loca[i,0] 
                else:
                    loca[i+1,0] = loca[i,0] + track[i,0]

                if y > boundary or y < 0: 
                    loca[i+1,1] = loca[i,1]    
                else:      
                    loca[i+1,1] = loca[i,1]+ track[i,1]     

                if z > boundary or z < 0:
                    loca[i+1,2] = loca[i,2]    
                else:
                    loca[i+1,2] = loca[i,2] + track[i,2]
            
            else: #if the particle moves into the nucleus let it take a step if random number is less than p_in.
                
                proba = np.random.rand()
                if proba <= pin:
                    state[n] = 1
                    loca[i+1,:] = loca[i, :] + track[i, :]
                    
                else:
                    state[n] = 2
                    loca[i+1, :] = loca[i, :]
                
        
    #generate 3D confocal image data of the simulated system 
    
    for k in range(t_slices): # t
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,z_slice]) * pixel_size #position of the beam
                
                particle_pos = loca[ i + image_size * j + image_size*image_size * k ,:] #position of proteins at this time point
                
                image_array[i,j,k] += GaussianBeam(particle_pos,beam_pos,psf_width,psf_height, B_protein) #measured signal at this beam position

#%% Generate trajectories and image array for condensate

#Array to contain the pixel values of the confocal images including condensates
image_array2 = np.zeros((image_size,image_size, t_slices))

#trajegtories of bound condensates
for n in range (N_condensate_b):
    
    #set up more parameters for the condensate to diffuse
    loca_cond = np.zeros((steps_cond,3)) #array to track the condensates location at step i
    loca_cond[0,:] = np.transpose(start_pos_cond_b[n,:])  #set the initial location to the starting position
    
    #Generate random steps (direction and size) for condensate i according to normal distribution
    track_cond = np.random.normal(loc=0, scale=np.sqrt(2*D_condensate_b*step_time),size=(steps_cond,3))
    
    #Let the condensate diffuse in the nucleus
    for i in range(steps_cond - 1):
        #Distance of condensate from center of nucleus after taking step
        forwd_cond = np.sqrt(((loca_cond[i,:] + track_cond[i,:]-center_pos)**2).sum()) 
        
        if forwd_cond <= radius: #If concensate remains in nucleus -> take step
            loca_cond[i+1,:] = loca_cond[i,:] + track_cond[i,:]
            
        else: #if condensate steps out of nucleus -> do not take step
            loca_cond[i+1,:] = loca_cond[i,:]
            
        
    cond_pos_x=loca_cond[0,0]
    cond_pos_y=loca_cond[0,1]
    cond_pos_z=loca_cond[0,2]
    
    #generate 3D confocal image data of the simulated system   
    for k in range(t_slices): # t
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,z_slice]) * pixel_size #position of the beam

                cond_pos = loca_cond[ (i + image_size * j + image_size*image_size * k)*2, :] #position of bound condensates at this time point
                
                image_array2[i,j,k] += GaussianBeam(cond_pos, beam_pos, psf_width, psf_height, B_condensate_b) #measured signal from bound condensates at this beam position

#%% Form and save images

for f in range(t_slices):
    
    #####
    #no condensate image
    #####
    #Add bacground signal and apply poisson noise
    image_array_no_condensate = np.transpose(image_array[:,:, f])
    image_array_no_condensate += B_background*dwell_time #add background signal to the image array
    noisy_no_condensate = np.random.poisson(image_array_no_condensate) #apply poisson noise to the image array

    #Save image array with no condensate
    folder = foldername
    filename = 'frame'+str(f)+'_image_array_no_condensate_2_pop.txt'
    file = os.path.join(folder, filename)
    #Add header containing values of parameters
    np.savetxt(file, noisy_no_condensate, 
                   header = 'No Condensate: ' + 
                   ', Nparticles = ' + str(Nparticles) + 
                   ', C_in = ' + str(C_in) +
                   ', C_out = ' + str(C_out) +
                   ', D_protein = ' + str(D_protein) +
                   ', D_condensate_b = '+ str(D_condensate_b) +
                   ', B_protein = '+ str(B_protein) +
                   ', B_condensate = '+ str(B_condensate_b) +
                   ', Condensate z position ='+ str(z_condensate)+
                   ', pixel_size = ' + str(pixel_size) +
                   ', dwell_time = ' + str(dwell_time))

    #####
    #with condensates image
    #####
    #Combine noisy no condensates image with image of only condensates
    image_array_w_condensate = np.transpose(image_array2[:,:,f])
    noisy_w_condensate = noisy_no_condensate + np.random.poisson(image_array_w_condensate)

    #Save image array with condensate
    folder = foldername
    filename = 'frame' + str(f)+'_image_array_w_condensate_2_pop.txt'
    file = os.path.join(folder, filename)
    #Add header containing values of parameters
    np.savetxt(file, noisy_w_condensate,
               header = 'With Condensate: ' + 
               ', Nparticles = ' + str(Nparticles) + 
               ', C_in = ' + str(C_in) +
               ', C_out = ' + str(C_out) +
               ', D_protein = ' + str(D_protein) +
               ', D_condensate_b = '+ str(D_condensate_b) +
               ', B_protein = '+ str(B_protein) +
               ', B_condensate = '+ str(B_condensate_b) +
               ', Condensate position (x,y,z) ='+ str(cond_pos_x) + ', ' + str(cond_pos_y) +', ' + str(cond_pos_z) +
               ', pixel_size = ' + str(pixel_size) +
               ', dwell_time = ' + str(dwell_time))
    

    

