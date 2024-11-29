from IHSetMOOSE import ih_moose
from IHSetCalibration import mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np
import rasterio
from affine import Affine
import json
import time

def tic():
    global start_time
    start_time = time.time()

def toc():
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")

config = {'crossshore' : 'YA',                                # EBSEM for cross-shore (MD, YA, SF, JR, JA, LIM)
         'longshore' : 'JA',                                   # EBSEM for longshore (TU, JA)
         'lim': [[(343241.26261840697, 6269439.614534981)],
         [(342874.0174683322, 6266120.806512082)]],            # Simulation boundary
         'npro': 100,                                          # Number of simulating profiles
         'lpro': 500,                                          # Length of simulating profiles
         'Fmean': 109.2900,                                    # Mean energy flux (N)
         'parabola_num' : 2,                                   # Number of planform (1: One side or 2: Both sides)
         'Cp' : [344915.384, 6266136.216],                     # Diffraction point (if parabola_num = 1)
         'Lr' : 1800,                                          # Beach length from control point (m)
         'Cp1' : [343537.570, 6269248.982],                    # Diffraction point (if parabola_num = 2)
         'Cp2' : [344915.384, 6266136.216],                    # Diffraction point (if parabola_num = 2)
         'Cl': [342565.979, 6267899.479],                      # Control pint
         'T': 10,                                              # Wave period (sec)
         'hd': 20                                              # Water depth at diffraction point (m)
         }

wrkDir = os.getcwd()
dataset = xr.open_dataset(wrkDir+'/data_hybrid/Narrabeen5.nc', engine='netcdf4')

dataset.load()
dataset.close()

config_json = json.dumps(config)
dataset.attrs['IH_MOOSE'] = config_json

dataset.to_netcdf(wrkDir+'/data_hybrid/Narrabeen5.nc', engine='netcdf4')
dataset.close()

tic()
### Step 1: Create planform and profile
planform = ih_moose.equilibrium_planform(wrkDir+'/data_hybrid/Narrabeen5.nc')

### Step 2: Calibrate EBSEM
model = ih_moose.cal_IH_MOOSE(wrkDir+'/data_hybrid/Narrabeen5.nc')

### Step 3: Simulate IH-MOOSE
results = ih_moose.ih_moose(wrkDir+'/data_hybrid/Narrabeen5.nc', planform, model)
toc()

########################################     IH-MOOSE Results (Map)     ########################################
img = rasterio.open(wrkDir+'/data/map.tif')

if isinstance(img.transform, Affine):
     transform = img.transform
else:
     transform = img.affine

N = img.width
M = img.height
dx = transform.a
dy = transform.e
minx = transform.c
maxy = transform.f

red = img.read(1)
green = img.read(2)
blue = img.read(3)
if dy < 0:
    dy = -dy
    red = np.flip(red, 0)
    green = np.flip(green, 0)
    blue = np.flip(blue, 0)
    
xdata = minx + dx/2 + dx*np.arange(N)
ydata = maxy - dy/2 - dy*np.arange(M-1,-1,-1)

extent = [xdata[0], xdata[-1], ydata[0], ydata[-1]]
color_image = np.stack((red, green, blue), axis=-1)

plt.imshow(np.flipud(color_image), extent=extent)

if planform.parabola_num == 1:
    plt.plot(planform.costa_xe, planform.costa_ye,'y')
    plt.plot(planform.alpha_curve[:,0], planform.alpha_curve[:,1],'c')
    plt.plot(planform.Cp[0], planform.Cp[1], 'ro', markersize = 5)
    plt.plot(planform.Cl[0], planform.Cl[1], 'ro', markersize = 5)

if planform.parabola_num == 2:
    plt.plot(planform.costa_xe, planform.costa_ye,'y')
    plt.plot(planform.alpha_curve1[:,0], planform.alpha_curve1[:,1],'c')
    plt.plot(planform.alpha_curve2[:,0], planform.alpha_curve2[:,1],'c')
    plt.plot(planform.Cp1[0], planform.Cp1[1], 'ro', markersize = 5)
    plt.plot(planform.Cp2[0], planform.Cp2[1], 'ro', markersize = 5)
    plt.plot(planform.Cl[0], planform.Cl[1], 'ro', markersize = 5)

for i in range(planform.npro):
    plt.plot([planform.prof[i,1], planform.prof[i,3]], [planform.prof[i,2], planform.prof[i,4]],'r', linewidth = 0.8)

for j in range(0, 165840, 100):
    plt.plot(results.costas_x[j,:], results.costas_y[j,:],'-', linewidth = 0.1)

plt.savefig('./results/Planform.png',dpi = 300)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

########################################     EBSEM     ########################################
fig, ax = plt.subplots(2 , 1, figsize=(10, 2), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5, 3.5]})
ylim_lower = np.floor(np.min([np.nanmin(model.cross.Obs), np.nanmin(model.cross_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.cross.Obs), np.nanmax(model.cross_run)]) / 2) * 2
ax[0].scatter(model.cross.time_obs, model.cross.Obs,s = 1, c = 'grey', label = 'Observed data')
ax[0].plot(model.cross.time, model.cross_run, color='red',linestyle='solid', label= 'EBSEM-Cross_shore')
ax[0].fill([model.cross.start_date, model.cross.end_date, model.cross.end_date, model.cross.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax[0].set_ylim([ylim_lower,ylim_upper])
# ax[0].set_ylim([0,150])
ax[0].set_xlim([model.cross.time[0], model.cross.time[-1]])
ax[0].set_ylabel('S [m]', fontdict=font)
ax[0].legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax[0].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

ylim_lower = np.floor(np.min([np.nanmin(model.long.Obs), np.nanmin(model.long_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.long.Obs), np.nanmax(model.long_run)]) / 2) * 2
ax[1].scatter(model.long.time_obs, model.long.Obs,s = 1, c = 'grey', label = 'Observed data')
ax[1].plot(model.long.time, model.long_run, color='red',linestyle='solid', label= 'EBSEM-longshore')
ax[1].fill([model.long.start_date, model.long.end_date, model.long.end_date, model.long.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax[1].set_ylim([ylim_lower,ylim_upper])
ax[1].set_xlim([model.long.time[0], model.long.time[-1]])
ax[1].set_ylabel('alpha [deg]', fontdict=font)
ax[1].legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax[1].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/EBSEM_Best_modelrun'+'.png',dpi=300)

########################################     IH-MOOSE Resutls (Validation)     ########################################
fig, ax = plt.subplots(results.ntrs , 1, figsize=(10, results.ntrs*1.5), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5] * results.ntrs})
rmse = np.zeros([results.ntrs,1])
nsse = np.zeros([results.ntrs,1])
mss = np.zeros([results.ntrs,1])
rp = np.zeros([results.ntrs,1])
bias = np.zeros([results.ntrs,1])

for i in range(results.ntrs):
        ylim_lower = np.floor(np.min([np.nanmin(results.Obs[:,i]), np.nanmin(results.SS[:,i])]) / 2) * 2
        ylim_upper = np.ceil(np.max([np.nanmax(results.Obs[:,i]), np.nanmax(results.SS[:,i])]) / 2) * 2
        ax[i].plot(model.cross.time, results.SS[:,i], color='red',linestyle='solid', label= 'Model')
        ax[i].scatter(model.cross.time_obs, results.Obs[:,i], s = 1, c = 'grey', label = 'Observations')
        ax[i].set_ylim([ylim_lower,ylim_upper])
        ax[i].set_xlim([model.cross.time[0], model.cross.time[-1]])
        ax[i].set_ylabel('S [m]', fontdict=font)
        ax[i].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
        ax[i].set_title(f"Profile {i + 1}", fontdict=font)
        
        Observations = results.Obs[:,i]
        run = results.SS[results.idx_obs,i]
        rmse[i] = spt.objectivefunctions.rmse(Observations, run)
        nsse[i] = spt.objectivefunctions.nashsutcliffe(Observations, run)
        mss[i] = mielke_skill_score(Observations, run)
        rp[i] = spt.objectivefunctions.rsquared(Observations, run)
        bias[i] = spt.objectivefunctions.bias(Observations, run)

        print('**********************************************************')
        print('Hybrid model (', planform.parabola_num, 'Parabolic )')
        print('Metrics - Profile', i+1, '           | Validation |')
        print('RMSE [m]                       | %-5.2f      |' % (rmse[i][0]))
        print('Nash-Sutcliffe coefficient [-] | %-5.2f      |' % (nsse[i][0]))
        print('Mielke Skill Score [-]         | %-5.2f      |' % (mss[i][0]))
        print('R2 [-]                         | %-5.2f      |' % (rp[i][0]))
        print('Bias [m]                       | %-5.2f      |' % (bias[i][0]))
ax[0].legend(ncol = 6, prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.55))

plt.subplots_adjust(hspace=0.6)
fig.savefig('./results/IH-MOOSE_Results'+'.png',dpi=300)
