#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:26:02 2023

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
# condtion -> 0 = laser OFF / -> 1 = laser ON
laser_cond = 0

#rat_name = "NB1" #ad (channel 0) + sty (channel 1). 2 tastes. WITH LASER
#rat_name = "NB16" #NB16 - ad (channel 0)  + sty (channel1). 4 tastes. NO LASER
#rat_name = "NB18" #NB18 - sty (channel 0). 4 tastes. NO LASER
#rat_name = "TG11" #ad (channel 0). 3 tastes
#rat_name = "TG13"
#rat_name = "NB7_test1"
rat_name = "NB27"

if rat_name == "NB1":
    dirname="/media/natasha/drive2/Natasha_Data/NB1/NB1_EMG_2tastes/NB1_2tastes_test1__220603_122545"
elif rat_name == "NB16":
    dirname="/media/natasha/drive2/Natasha_Data/NB16/NB16_Test1_4tastes_221202_144624"
elif rat_name == "NB18":
    dirname="/media/natasha/drive2/Natasha_Data/NB18/NB18_test1_4tastes_230128_144845"
elif rat_name == "TG11":
    dirname="/media/natasha/drive2/Natasha_Data/TG11/"
elif rat_name == "TG13":
    dirname="/media/natasha/drive2/Natasha_Data/TG13/"
elif rat_name == "NB27":
    dirname = "/media/natasha/drive2/Natasha_Data/NB27/Test1/NB27_test1_4tastes_230520_134124"
#elif rat_name == "NB7_test1":
    
h5_name = glob.glob(os.path.join(dirname, '*.h5'))[0]

h5 = tables.open_file(h5_name, 'r')

# SHAPE : channel x laser_cond x taste x trial x time x freq
emg_BSA_results = h5.get_node("/emg_BSA_results", "emg_BSA_results_final")[:]

# SHAPE : channel x laser_cond x taste x time x freq
meanEMG = np.mean(emg_BSA_results, axis=3)
my_vmin = meanEMG.min(axis=None)
my_vmax = meanEMG.max(axis=None)

   
    
######################################
# PLOTTING
#######################################


#Generating plot titles
#Get the number of tastants
taste_num  = sum(1 for i in h5.get_node("/spike_trains"))
if taste_num == 4:
    taste_names = ["Sucrose", "NaCl", "CA", "QHCl"]
elif taste_num == 2:
    taste_names = ["Sucrose", "QHCl"]
elif taste_num == 3:
    taste_names = ["Water", "Sac", "QhCl"]
else:
    print("taste_num is neither 4 nor 2")  

#channel titles
channel_num  = len(emg_BSA_results)
if channel_num == 1:
    channel_names = ["Styloglossus"]
elif channel_num == 2:
    channel_names = ["Anterior Digastric", "Styloglossus"]
#elif channel_num == 1 && rat_name[0:2]=='TG':
#    channel_names = ["Anterior Digastric"]
else:
    print("channel_num is neither 1 nor 2")
    
    
 
    
'''

### show individual 
for channel in range(channel_num):
    plt.figure(channel)
    for i in range(taste_num):  
        plt.subplot(2,2,i+1)
        plt.imshow(meanEMG[channel,0,i].T, interpolation='nearest',aspect='auto', origin='lower')
        plt.title(taste_names[i])
    plt.suptitle(channel_names[channel])
    plt.subplots_adjust(hspace=0.5)
'''
# condtion -> 0 = laser OFF / -> 1 = laser ON
#set at the top
if laser_cond == 0:
    laser_name = 'laser OFF'
elif laser_cond == 1:
    laser_name = 'laser ON (0.7s to 1.2s)'



 
if rat_name == 'NB1':
    my_figsize = (8,4)
elif rat_name == 'NB16':
    my_figsize = (8,8)
elif rat_name == 'NB18':
    my_figsize = (4,8)
else:
    my_figsize = (4,8)

#setting up x_ticklables
#range(start value, total time plotted, steps)
x_ticks = np.arange(-0.5, 3, 0.5)

#setting height ratio
#where top row is 0, so that I can have column titles
hr = np.zeros(taste_num+1)
hr[1:] = 1

#re-run here to generate plot
fig, axs = plt.subplots(
    taste_num+1, channel_num, 
    squeeze = False, 
    figsize=my_figsize, 
    sharey=True, sharex=True, 
    gridspec_kw={'height_ratios':hr})

for channel in range(channel_num):
    for i in range(taste_num):  
        this_dat = meanEMG[channel,laser_cond,i].T
        im = axs[i+1, channel].imshow(this_dat, 
                            interpolation='nearest',aspect='auto', 
                            origin='lower',
                            vmin = my_vmin, vmax=my_vmax)
        plt.colorbar(im, ax = axs[i+1, channel])
        axs[i+1, channel].set_title(taste_names[i])
        axs[i+1, channel].set_xlim(1500,4500)
        axs[i+1, channel].set_xticklabels(x_ticks)
        axs[i+1, channel].set_yticks((0, 10))
        axs[i+1, channel].set_yticklabels((0, 5))
        axs[i+1, channel].axvline(2000, color = 'w')
        #axs[i+1, channel].set_xlabel('Time (s)')
for i, ax in enumerate(axs.flatten()[:channel_num]):
    ax.axis("off")
    ax.set_title(channel_names[i], fontweight='bold')
    
fig.text(0.5, 0.01, 'Time (s)', ha='center', va='center')
fig.text(0.01, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
fig.suptitle(rat_name + ' - ' + laser_name, fontsize=16, fontweight='bold', y= 1.05)
plt.tight_layout()
plt.show()


