# -*- coding: utf-8 -*-
"""
Simulation of the confocal image generation of nucleus with condensate
Parameters set to base value
"""

# Import Packages

import numpy as np
import os

# Set up parameters for imaging and simulation
#foldername = r"C:\Users\Lydia\OneDrive\OneDrive - McMaster University\Documents\Summer Research Project\rerun_simulation" #folder to save data
foldername = "rerun_sim_trial1"

image_size = 120 # px (The image will be square 120 x 120 x 120)
pixel_size = 0.1 # um
boundary = image_size * pixel_size
radius = image_size * pixel_size / 4 
V_n = (4/3) * np.pi *radius**3 #um^3; volume of nucleus
V_c = (image_size*pixel_size)**3 - V_n #um^3; volume of cytoplasm
center_pos = [radius*2, radius*2, radius*2]
dwell_time = 4.1E-6 # s
psf_width = 0.3 # um (Width of the point spread function in focus)
psf_height = 1.5 # um
D_protein = 15 # um^2/s; diffusion coefficient of mobile proteins
D_condensate_m = 0.4 #um^2/s; diffusion coefficient of "mobile" condensates
D_condensate_b = 0.01 #um^2/s; diffusion coefficient of "bound" condensates
step_time = 4.1E-6 # s; step time of proteins and condensates
B_protein = 30000 # Hz; Brightness of protein
B_background = 45000 #Hz; Brightness of background noise
B_condensate_m = 7*30000 #Hz; Brightness of "mobile" condensates
B_condensate_b = 20*30000 #Hz; Brightness of "bound" condensates
z_condensate = 6.0
z_slice = 59 # Position of focal plane along the z-axis
t_slices = 20
steps = image_size * image_size * t_slices #number of steps the proteins take
steps_cond = 2*steps # number of steps the condensate takes

C_in = 75 # nM; set the nuclear concentration 

# Calculate the number of particles based on the nuclear concentration, and the nuclear to cytoplasmic ratio is 6
N_f_N_b = (C_in / 1e9) * (V_n / 1e15) * 6.02e23 
N_c = (C_in / (6 * 1e9)) * (V_c / 1e15) * 6.02e23
Nparticles = N_f_N_b + N_c # total number of particles 
N_condensate_b = 5  # number of bound condensates
N_condensate_m = 10  # number of mobile condensates

pout = 0.2 #  flow rate of proteins from nucleus to cytoplasm
pin = 0.8 # flow rate of proteins from cytoplasm to nucleus
pon = 0.005 # rate that proteins switch from unbound to bound
poff = 0.01 # rate that proteins switch from bound to unbound

N_f = ((poff/(poff + (poff * pout)/pin * V_c/V_n + pon)) * Nparticles) # number of unbound (free) proteins
N_b = ((pon/poff) * N_f) # number of bound proteins
N_c = ((pout/pin) * (V_c/V_n) * N_f) # number of proteins in the cytoplasm

N_f = round(N_f)
N_b = round(N_b)
N_c = round(N_c)

Nparticles = N_f + N_b + N_c

print('Set number of bound proteins = ', N_b)
print('Set number of free proteins = ', N_f)
print('Set number of cytoplasmic proteins = ', N_c, '\n')
    
C_in = (((N_f+N_b)/6.02e23)/(V_n/1e15))*1e9 #nM; concentration of protein inside the nucleus
C_out = ((N_c/6.02e23)/(V_c/1e15))*1e9 #nM; concentration of protein outside the nucleus

print('Concentration of proteins inside the nucleus = ', round(C_in, 1), 'nM')
print('Concentration of proteins outside the nucleus = ', round(C_out, 1), 'nM')

print('The expected nuclear to cytoplasmic ratio = ', round(C_in/C_out, 3), '\n')

# Generate initial posistions of particles

start_pos = np.zeros((Nparticles,3))
state = np.zeros((Nparticles))

for n in range(0,N_b):        #Place N_b number of proteins in the nucleus and set state to 0 (bound)
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) <= radius**2: 
            start_pos[n,:] = x
            state[n] = 0

for n in range(N_b, N_b+N_f):       #Place N_f number of proteins in the nucleus and set state to 1 (free)
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) <= radius**2: 
            start_pos[n,:] = x
            state[n] = 1
            
for n in range(N_b+N_f, Nparticles): #Place N_c number of proteins in the cytoplase and set state to 2 (cytoplasm)
    temp = start_pos[n, :]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x=np.random.rand(3) * image_size * pixel_size
        if (((x - center_pos)**2).sum()) > radius**2 and (x[0]**2 + x[1]**2 + x[2]**2) > 0:
            start_pos[n,:] = x
            state[n] = 2
            
# Generate initial position of "bound" condensates inside the nucleus within 1 um of the imaging plane

start_pos_cond_b = np.zeros((N_condensate_b,3))
for n in range(N_condensate_b):
    temp = start_pos_cond_b[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x=np.random.rand(3)*image_size*pixel_size
        if (((x - center_pos)**2).sum()) <= (radius-0.5)**2 and (x[0]**2 + x[1]**2 + x[2]**2) > 0 and (x[2] > z_condensate - 5 and x[2] < z_condensate + 5):
            start_pos_cond_b[n,:] = x
  
# Generate initial position of "mobile" condensates inside the nucleus within 1 um of the imaging plane
start_pos_cond_m = np.zeros((N_condensate_m,3))
for n in range(N_condensate_m):
    temp = start_pos_cond_m[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
         x=np.random.rand(3)*image_size*pixel_size
         if (((x - center_pos)**2).sum()) <= (radius-0.5)**2 and (x[0]**2 + x[1]**2 + x[2]**2) > 0 and (x[2] > z_condensate - 5 and x[2] < z_condensate + 5):
            start_pos_cond_m[n,:] = x

print('Check 1')
# Define the function of Gaussian Beam Profile

def GaussianBeam( start_pos, beam_pos, psf_width, psf_height, B):
    if start_pos.shape[0] == 2:
        GB = B*dwell_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2)
    else:
        GB = B*dwell_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB

# Generate the trajectories and image array for the proteins

image_array = np.zeros((image_size,image_size,t_slices))
    
for n in range(Nparticles):
    
    # Settting up more parameters for the particles to diffuse
    
    loca = np.zeros((steps,3)) #array of the location of the protein at step i
    loca[0,:] = np.transpose(start_pos[n,:]) #set the initial location to the starting position

    track = np.random.normal(loc=0,scale=np.sqrt(2*D_protein*step_time),size=(steps,3)) 
    
    #Let the particles diffuse inside the simulation box
    
    for i in range(steps-1):
        depth = np.sqrt(((loca[i,:]-center_pos)**2).sum())
        forwd = np.sqrt(((loca[i,:] + track[i,:] - center_pos)**2).sum())
        
        if depth <= radius: #if the particle is in the nucleus
            
            if state[n] == 1: #if the particle is free
                
                proba = np.random.rand()
                if proba <= pon: #if random number is less than pon the particle becomes bound
                    state[n] = 0
                    loca[i+1,:] = loca[i, :]
                    
                else: #if random number is greater than pon the particle remains free
                    
                    if forwd <= radius: #if the particle remains in the nucleus let it take a step
                        state[n] = 1
                        loca[i+1, :] = loca[i, :] + track [i, :]
                    
                    else: #if the particle moves out of the nucleus let it take a step if random number is less than pout
                        
                        proba = np.random.rand()
                        if proba <= pout:
                            state[n] = 2
                            loca[i+1, :] = loca[i, :] + track[i, :]
                        
                        else:
                            state[n] = 1
                            loca[i+1, :] = loca[i, :]
            else: #if the particle is bound
                
                proba = np.random.rand()
                if proba <= poff: #if random number is less than poff the particle becomes free
                    
                    if forwd <= radius: #if the particle remains in the nucleus let it take a step
                        state[n] = 1
                        loca[i+1, :] = loca[i, :] + track[i, :]
                        
                    else: #if the particle moves into the cytoplasm let it take a step if random number is less than pout
                        
                        proba = np.random.rand()
                        if proba <= pout:
                            state[n] = 2
                            loca[i+1, :] = loca[i, :] + track[i, :]
                        
                        else: 
                            state[n] = 1
                            loca[i+1, :] = loca[i, :]
                
                else: #if random number is greater than poff the particle remains bound.
                    state[n] = 0
                    loca[i+1, :] = loca[i, :]
        
        else: #if the particle is in the cytoplasm
            
            if forwd >= radius: #if the particle remains in the cytoplasm let it take a step so long as it doesn't leave the simulation window
                state[n] = 2
                x  = loca[i,0] + track[i,0]
                y  = loca[i,1] + track[i,1]
                z  = loca[i,2] + track[i,2]

                if  x > boundary or x < 0: 
                     loca[i+1,0] = loca[i,0] # - track[i,0,n]
                else:
                    loca[i+1,0] = loca[i,0] + track[i,0]

                if y > boundary or y < 0: 
                    loca[i+1,1] = loca[i,1] # - track[i,1,n]    
                else:      
                    loca[i+1,1] = loca[i,1]+ track[i,1]     

                if z > boundary or z < 0:
                    loca[i+1,2] = loca[i,2] # - track[i,2,n]    
                else:
                    loca[i+1,2] = loca[i,2] + track[i,2]
            
            else: #if the particle moves into the nucleus let it take a step if random number is less than pin.
                
                proba = np.random.rand()
                if proba <= pin:
                    state[n] = 1
                    loca[i+1,:] = loca[i, :] + track[i, :]
                    
                else:
                    state[n] = 2
                    loca[i+1, :] = loca[i, :]
                
        
    #generate 3D confocal image data of the simulated system 
    
    for k in range(t_slices): # z
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,z_slice]) * pixel_size
                
                particle_pos = loca[ i + image_size * j + image_size*image_size * k ,:]
                image_array[i,j,k] += GaussianBeam(particle_pos,beam_pos,psf_width,psf_height, B_protein)

print('Check 2')
# Generate trajectories and image array for condensates

image_array2 = np.zeros((image_size,image_size, t_slices))
# "bound" condensates
for n in range (N_condensate_b):
    
    #set up more parameters for the condensate to diffuse
    
    loca_cond = np.zeros((steps_cond,3)) #array to track the condensates location at step i
    
    track_cond = np.random.normal(loc=0, scale=np.sqrt(2*D_condensate_b*step_time),size=(steps_cond,3))
    
    loca_cond[0,:] = np.transpose(start_pos_cond_b[n,:])
    
    #Let the condensate diffuse in the nucleus
    
    for i in range(steps_cond - 1): #let the condensate diffuse for "steps_cond" # of steps starting from the random start time
            
        forwd_cond = np.sqrt(((loca_cond[i,:] + track_cond[i,:]-center_pos)**2).sum())
        
        if forwd_cond <= radius:
            loca_cond[i+1,:] = loca_cond[i,:] + track_cond[i,:]
            
        else:
            loca_cond[i+1,:] = loca_cond[i,:]
            
        
    cond_pos_x=loca_cond[0,0]
    cond_pos_y=loca_cond[0,1]
    cond_pos_z=loca_cond[0,2]
    
    #generate 3D confocal image data of the simulated system
    
    for k in range(t_slices):
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,z_slice]) * pixel_size

                cond_pos = loca_cond[ (i + image_size * j + image_size*image_size * k)*2, :] 
                image_array2[i,j,k] += GaussianBeam(cond_pos, beam_pos, psf_width, psf_height, B_condensate_b)

# "mobile" condensates
for n in range (N_condensate_m):
    
    #set up more parameters for the condensate to diffuse
    
    loca_cond = np.zeros((steps_cond,3)) #array to track the condensates location at step i
    
    track_cond = np.random.normal(loc=0, scale=np.sqrt(2*D_condensate_m*step_time),size=(steps_cond,3))
    
    loca_cond[0,:] = np.transpose(start_pos_cond_m[n,:])
    
    #Let the condensate diffuse in the nucleus
    
    for i in range(steps_cond - 1): #let the condensate diffuse for "steps_cond" # of steps starting from the random start time
            
        forwd_cond = np.sqrt(((loca_cond[i,:] + track_cond[i,:]-center_pos)**2).sum())
        
        if forwd_cond <= radius:
            loca_cond[i+1,:] = loca_cond[i,:] + track_cond[i,:]
            
        else:
            loca_cond[i+1,:] = loca_cond[i,:]
            
        
    cond_pos_x=loca_cond[0,0]
    cond_pos_y=loca_cond[0,1]
    cond_pos_z=loca_cond[0,2]
    
    #generate 3D confocal image data of the simulated system
    
    for k in range(t_slices):
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,z_slice]) * pixel_size

                cond_pos = loca_cond[ 2*(i + image_size * j + image_size*image_size * k), :] 
                image_array2[i,j,k] += GaussianBeam(cond_pos, beam_pos, psf_width, psf_height, B_condensate_m)

print('Check 3')

# Form and save images

for f in range(t_slices):
    #no condensate image
    
    #Apply poisson noise
    image_array_no_condensate = np.transpose(image_array[:,:, f])
    image_array_no_condensate += B_background*dwell_time #add background signal to the image array
    noisy_no_condensate = np.random.poisson(image_array_no_condensate) #apply poisson noise to the image array

    #Save image array with no condensate
    
    folder = foldername
    filename = 'frame'+str(f)+'_image_array_no_condensate_2_pop.txt'
    file = os.path.join(folder, filename)
    np.savetxt(file, noisy_no_condensate, 
                   header = 'No Condensate: ' + 
                   ', Nparticles = ' + str(Nparticles) + 
                   ', C_in = ' + str(C_in) +
                   ', C_out = ' + str(C_out) +
                   ', D_protein = ' + str(D_protein) +
                   ', D_condensate_b = '+ str(D_condensate_b) +
                   ', D_condensate_m = '+ str(D_condensate_m) +
                   ', B_protein = '+ str(B_protein) +
                   ', B_condensate_b = '+ str(B_condensate_b) +
                   ', B_condensate_m = '+ str(B_condensate_m) +
                   ', Condensate z position ='+ str(z_condensate)+
                   ', pixel_size = ' + str(pixel_size) +
                   ', dwell_time = ' + str(dwell_time))

    #with condensates image
    #Apply poisson noise
    image_array_w_condensate = np.transpose(image_array2[:,:,f])
    noisy_w_condensate = noisy_no_condensate + np.random.poisson(image_array_w_condensate)

    #Save image array with condensate

    folder = foldername
    filename = 'frame' + str(f)+'_image_array_w_condensate_2_pop.txt'
    file = os.path.join(folder, filename)
    np.savetxt(file, noisy_w_condensate,
               header = 'With Condensate: ' + 
               ', Nparticles = ' + str(Nparticles) + 
               ', C_in = ' + str(C_in) +
               ', C_out = ' + str(C_out) +
               ', D_protein = ' + str(D_protein) +
               ', D_condensate_b = '+ str(D_condensate_b) +
               ', D_condensate_m = '+ str(D_condensate_m) +
               ', B_protein = '+ str(B_protein) +
               ', B_condensate_b = '+ str(B_condensate_b) +
                ', B_condensate_m = '+ str(B_condensate_m) +
               ', Condensate position (x,y,z) ='+ str(cond_pos_x) + ', ' + str(cond_pos_y) +', ' + str(cond_pos_z) +
               ', pixel_size = ' + str(pixel_size) +
               ', dwell_time = ' + str(dwell_time))
    
print('Done')
    

