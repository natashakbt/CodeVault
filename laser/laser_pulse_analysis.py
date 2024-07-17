#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:14:10 2022

@author: natasha
"""


import numpy as np
import tables
import glob
import os
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import pingouin as pg


#######################################
# IMPORTING DATA AND GETTING SETUP
#######################################
#day2:
dirname="/media/natasha/drive2/Natasha_Data/NB13/NB13_test2_4tastes_220922_074336"
#day3:
#dirname="/media/natasha/drive2/Natasha_Data/NB13/NB13_test3_4tastes_REAL_220923_090323"
h5_name = glob.glob(os.path.join(dirname, '*.h5'))[0]

h5 = tables.open_file(h5_name, 'r')

#import all dig_in 4 laser times
laser_dig_in = h5.get_node("/digital_in", "dig_in_4")[:]

#finding onset of laser pulse
laser_start_times = np.where((laser_dig_in[1:] - laser_dig_in[0:-1])==1)[0]

#pointing to all the sorted units
all_units = h5.iter_nodes("/sorted_units")

#list where each element is a unit. Each unit is a list of all spike times
all_unit_times = []
for unit in all_units:
    unit_times = unit["times"][:]
    all_unit_times.append(unit_times)
    
    
# graphing all units across entire experiment
# into grid of suplots
grid_len = int(np.ceil(np.sqrt(len(all_unit_times))))
fig,ax = plt.subplots(grid_len, grid_len, 
                      sharex=True, sharey = True, figsize = (7,7))

fig,ax = plt.subplots(4, 7, 
                      sharex=True, sharey = True, figsize = (10,5))
for this_dat, this_ax in zip(all_unit_times, ax.flatten()):
    N, bins, patches = this_ax.hist(this_dat)
    patches[-1].set_facecolor('r')
    #this_ax.hist[-1]('r')
    #this_ax.axis('off')

laser_period_time = (np.where(laser_dig_in==1)[0][0]) - np.where(laser_dig_in==1)[0][-1]
fig, ax = plt.subplots()
#ax.plot(all_unit_t, 'go')
plt.xlim([laser_start_times[0]-laser_period_time,88780020])
plt.show()



#######################################
# START ANALYSIS PER LASER PULSE
#######################################
#pull out number of spikes for each unit
#before and after each laser pulse
#to compare whether number of spikes significantly change
    
bsln_pulse_spikes = [] #Becomes list of list. First element sepearates unit
laser_pulse_spikes = []
for unit in range(len(all_unit_times)):
    unit_bsln = []; unit_laser = []
    for laser_pulse in laser_start_times:
        #sum all the spikes 150ms before laser pulse for baseline
        #4500 for 150ms. 750 for 5ms
        bsln = (((all_unit_times[unit]>laser_pulse-750)) & (all_unit_times[unit]<laser_pulse)).sum()
        #sum all the spikes for set time after laser pulse for laser
        laser = (((all_unit_times[unit]>laser_pulse)) & (all_unit_times[unit]<(laser_pulse+750))).sum()
        
        unit_bsln.append(bsln)
        unit_laser.append(laser)
    #append list of baseline spikes of a unit to larger list bsln_spikes
    bsln_pulse_spikes.append(unit_bsln)
    laser_pulse_spikes.append(unit_laser)

bsln_pulse_spikes_array = np.array(bsln_pulse_spikes)
laser_pulse_spikes_array = np.array(laser_pulse_spikes)
inds = np.array(list(np.ndindex(bsln_pulse_spikes_array.shape)))

bsln_frame = pd.DataFrame(
        dict(
                cond = 'bsln',
                unit = inds[:,0],
                pulse = inds[:,1],
                count = bsln_pulse_spikes_array.flatten()
                )
        )
        
laser_frame = pd.DataFrame(
        dict(
                cond = 'laser',
                unit = inds[:,0],
                pulse = inds[:,1],
                count = laser_pulse_spikes_array.flatten()
                )
        )
        
fin_frame = pd.concat([bsln_frame, laser_frame])
group_frame = [x[1] for x in fin_frame.groupby('unit')]

anova_list = [
            pg.anova(
                data = this_frame,
                dv = 'count',
                between = ['cond'],
                )
                for this_frame in group_frame]


p_val_list = [(int(i), x['p-unc'][0]) for i,x in enumerate(anova_list) \
              if 'p-unc' in x.columns]
p_val_list = np.stack(p_val_list)

    
for i in range(len(p_val_list)):
    if p_val_list[i]<0.05:
        print(i, p_val_list[i])



#######################################
# MAKE "PSTH" GRAPHS FOR LASER PULSE
# "stimulus" is laser pulse
#######################################

#array of zeros. Length of entire experiment, repeated for each unit
all_unit_spike_array = np.zeros((len(all_unit_times), len(laser_dig_in[1:])), dtype=int)

#changing zeros to 1 when unit spikes
for unit in range(len(all_unit_spike_array)):
    all_unit_spike_array[unit][all_unit_times[unit]] = 1


#pulling out 1000ms before laser pulse and 1000ms after. 
#stacking them into 
unit_psth = []
for unit in range(len(all_unit_spike_array)):
    temp_psth = []
    for laser_pulse in laser_start_times:
        psth_array = all_unit_spike_array[unit][(laser_pulse-30000):(laser_pulse+30000)]
        #sum all the spikes 150ms after laser pulse for laser
        temp_psth.append(psth_array)
        #after_stim.append(after)
    #append list of baseline spikes of a unit to larger list bsln_spikes
    unit_psth.append(temp_psth)

# compresses spike array from 30Hz sampling to 1ms
# if spike occured during 1ms time bin, value set to 1
unit_psth = np.stack(unit_psth)
bin_width  = 30
unit_psth = np.reshape(unit_psth, (*unit_psth.shape[:2], -1, bin_width))
unit_psth = unit_psth.sum(axis=-1) > 0


#average firing per ms across all laser pulses, per unit
avg_unit_psth = [] 
for unit in range(len(unit_psth)):
    avg_unit_psth.append(np.average(unit_psth[unit], axis = 0))



x = np.linspace(0, len(all_unit_spike_array[0]), len(all_unit_spike_array[0]))

fig, ax = plt.subplots()
ax.scatter(x, all_unit_spike_array[0])


###
fig, ax = plt.subplots()
ax.scatter(np.where(all_unit_spike_array[13])[0], np.random.random(6294), s = 1)
ax.plot([np.where(laser_dig_in)[0][0],
         np.where(laser_dig_in)[0][-1]],
        [0.95, 0.95], 'r')
plt.show()



#######################################################
# 1000ms before and after laser pulse plotting
######################################################

#define x axis
start =-1000; stop = start*-1
n_steps = len(avg_unit_psth[0])
x = np.linspace(start, stop, n_steps) 


sigma = 10 #adjust this for gaussian smoothing

#### PSTH -1000ms to 1000ms, 0 set to laser pulse, 1ms steps
fig, axs = plt.subplots(10)
i = 0
for unit in range(10):
    y = gaussian_filter1d(avg_unit_psth[unit], sigma)
    axs[i].plot(x, y)
    i = i+1
plt.show()



#######################################################
# 5ms bins, 25ms before and after laser pulse plotting
# BROKEN!!!!!!
######################################################

### making 5ms average bins
bin_avg_unit_psth = []
for unit in range(len(avg_unit_psth)):
    start = 975
    bin_unit =[]
    while start < 1025:
        bin_unit.append(np.average(avg_unit_psth[unit][start:start+5]))
        start = start+5
    bin_avg_unit_psth.append(bin_unit)


grid_len = int(np.ceil(np.sqrt(len(all_unit_times))))
fig,ax = plt.subplots(16, 2, 
                      sharex=True, figsize=(15, 20))
for this_dat, this_ax in zip(bin_avg_unit_psth, ax.flatten()):
    this_ax.plot(x, this_dat)
    #this_ax.axis('off')


start =-20; stop = 25
n_steps = len(bin_avg_unit_psth[0])
x = np.linspace(start, stop, n_steps) 


fig, axs = plt.subplots(16, 2)
for i in range(len(bin_avg_unit_psth)):
    if i < 16:
       axs[i, 0].plot(x, bin_avg_unit_psth[i])
    else:
        axs[i-16, 1].plot(x, bin_avg_unit_psth[i])
plt.show()



#######################################################
# stats: spike sum of 25ms before vs. after laser pulse
######################################################

unit_pre_laser = []; unit_post_laser = []

for unit in range(len(unit_psth)):
    pre_laser =[]; post_laser = []
    for laser in range(len(unit_psth[unit])):
        pre_laser.append(sum(unit_psth[unit][laser][975:1000]))
        post_laser.append(sum(unit_psth[unit][laser][1000:1025]))
    unit_pre_laser.append(pre_laser)
    unit_post_laser.append(post_laser)

stat_results = scipy.stats.f_oneway(np.array(unit_pre_laser).T, np.array(unit_post_laser).T)
np.where(stat_results[1]<0.05)




unit_pre_laser_array = np.array(unit_pre_laser)
unit_post_laser_array = np.array(unit_post_laser)
inds = np.array(list(np.ndindex(unit_pre_laser_array.shape)))

bsln_frame = pd.DataFrame(
        dict(
                cond = 'bsln',
                unit = inds[:,0],
                pulse = inds[:,1],
                count = unit_pre_laser_array.flatten()
                )
        )
        
laser_frame = pd.DataFrame(
        dict(
                cond = 'laser',
                unit = inds[:,0],
                pulse = inds[:,1],
                count = unit_post_laser_array.flatten()
                )
        )
        
fin_frame = pd.concat([bsln_frame, laser_frame])
group_frame = [x[1] for x in fin_frame.groupby('unit')]

anova_list = [
            pg.anova(
                data = this_frame,
                dv = 'count',
                between = ['cond'],
                )
                for this_frame in group_frame]


p_val_list = [(int(i), x['p-unc'][0]) for i,x in enumerate(anova_list) \
              if 'p-unc' in x.columns]
p_val_list = np.stack(p_val_list)

#finding significant p-values
wanted_nrns = p_val_list[:,0][np.where(p_val_list[:,1] < 0.05)[0]]





#######################################
# START ANALYSIS FOR ENTIRE LASER PERIOD
#######################################

# making list for laser period (first laser pulse to last laser pulse)
# split into 5 "chunks"
first_laser = np.where(laser_dig_in!=0)[0][0]
last_laser = np.where(laser_dig_in!=0)[0][-1]
start = first_laser
step = ((last_laser-first_laser)/5)
num = 6
laser_period_chunks = np.arange(0,num)*step+start

# baseline period is equal in length to laser period,
#right before the start of the laser period
# making list with baseline period split into 5 "chunks"
#start = first_laser
end = first_laser - (last_laser - first_laser)
step = ((end-start)/5)
num = 6
bsln_period_chunks = np.flip(np.arange(0,num)*step+start)


spikes_laser_chunks = []; spikes_bsln_chunks = []
for unit in range(len(all_unit_times)):
    #per unit, counting number of spikes that occur in first baseline period chunk
    #then chunk 2... until chunk 5
    temp_bsln=[]
    temp_bsln.append(len((np.where((all_unit_times[unit]>bsln_period_chunks[0]) & (all_unit_times[unit]<bsln_period_chunks[1])))[0]))      
    temp_bsln.append(len((np.where((all_unit_times[unit]>bsln_period_chunks[1]) & (all_unit_times[unit]<bsln_period_chunks[2])))[0]))    
    temp_bsln.append(len((np.where((all_unit_times[unit]>bsln_period_chunks[2]) & (all_unit_times[unit]<bsln_period_chunks[3])))[0]))
    temp_bsln.append(len((np.where((all_unit_times[unit]>bsln_period_chunks[3]) & (all_unit_times[unit]<bsln_period_chunks[4])))[0]))    
    temp_bsln.append(len((np.where((all_unit_times[unit]>bsln_period_chunks[4]) & (all_unit_times[unit]<bsln_period_chunks[5])))[0]))
    spikes_bsln_chunks.append(temp_bsln)
    
    #per unit, counting number of spikes that occur in first laser period chunk
    #then chunk 2... until chunk 5
    temp_laser=[]
    temp_laser.append(len((np.where((all_unit_times[unit]>laser_period_chunks[0]) & (all_unit_times[unit]<laser_period_chunks[1])))[0]))      
    temp_laser.append(len((np.where((all_unit_times[unit]>laser_period_chunks[1]) & (all_unit_times[unit]<laser_period_chunks[2])))[0]))    
    temp_laser.append(len((np.where((all_unit_times[unit]>laser_period_chunks[2]) & (all_unit_times[unit]<laser_period_chunks[3])))[0]))
    temp_laser.append(len((np.where((all_unit_times[unit]>laser_period_chunks[3]) & (all_unit_times[unit]<laser_period_chunks[4])))[0]))    
    temp_laser.append(len((np.where((all_unit_times[unit]>laser_period_chunks[4]) & (all_unit_times[unit]<laser_period_chunks[5])))[0]))
    spikes_laser_chunks.append(temp_laser)

stat_results = scipy.stats.f_oneway(np.array(spikes_bsln_chunks).T, np.array(spikes_laser_chunks).T)
np.where(stat_results[1]<0.05)

for i in range(len(spikes_bsln_chunks)):
    if np.mean(spikes_bsln_chunks[i]) < np.mean(spikes_laser_chunks[i]):
        print(i, np.mean(spikes_bsln_chunks[i]), np.mean(spikes_laser_chunks[i]))



#######################################################
# troubleshooting - plotting day 3 unit 13
######################################################

### spikes across entire experiment
# red line is laser period
fig, ax = plt.subplots()
unit_num = 8
ax.scatter(np.where(all_unit_spike_array[unit_num])[0], 
           np.random.random(len(np.where(all_unit_spike_array[unit_num])[0])), 
           s = 1)
ax.plot([np.where(laser_dig_in)[0][0],
         np.where(laser_dig_in)[0][-1]],
        [0.95, 0.95], 'r')
plt.show()

### spikes across experiment
# red dots specific to laser on
x = np.ones(len(np.where(laser_dig_in)[0])).tolist()

fig, ax = plt.subplots( figsize = (15,7))
ax.scatter(np.where(all_unit_spike_array[13])[0], np.random.random(6294), s = 1)
ax.scatter(np.where(laser_dig_in)[0], x, c='r', s = 1)
plt.xlim(80000000, 90000000)
plt.ylim(-1,2)


### 25ms before and after laser period, for each laser pulse
start =-25; stop = 25
n_steps = 1500
x = np.linspace(start, stop, n_steps) 

fig,ax = plt.subplots(34, 3, 
                      sharex=True, sharey = True, figsize = (20,7))
for this_dat, this_ax in zip(unit_psth[13], ax.flatten()):
    this_ax.plot(this_dat[950:1050])
    #this_ax.axis('off')
    
length = unit_psth.shape[-1]
width = 25
lims = [(length//2) - width, (length//2)+width]
plt.imshow(unit_psth[13][:,lims[0]:lims[1]], aspect = "auto")
plt.axvline(width, color = 'yellow', linestyle = '--')

