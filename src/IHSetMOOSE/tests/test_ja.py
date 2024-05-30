from IHSetMOOSE import calibration, ih_moose
from IHSetCalibration import setup_spotpy, mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np
import pandas as pd
from IHSetUtils import waves
from scipy.io import savemat
import datetime
import matplotlib.dates as mdates

# Avaliable methods: NSGAII, mle, mc, dds, mcmc, sa, abc, lhs, rope, sceua, demcz, padds, fscabc

# config = xr.Dataset(coords={'dt': 1,                # [hours]
#                             'switch_Yini': 0,       # Calibrate the initial S position? (0: No, 1: Yes)
#                             'crossshore' : 'YA',    # EBSEM for cross-shore (MD, YA, SF, JA, LIM)
#                             'switch_alpha_ini': 0,  # Calibrate the initial alpha position? (0: No, 1: Yes)
#                             'longshore' : 'JA',     # EBSEM for longshore (TU, JA)
#                             'Ysi': 2017,              # Initial year for calibration
#                             'Msi': 4,                 # Initial month for calibration
#                             'Dsi': 6,                 # Initial day for calibration
#                             'Ysf': 2020,              # Final year for calibration
#                             'Msf': 1,                 # Final month for calibration
#                             'Dsf': 26,                 # Final day for calibration
#                             'cal_alg': 'NSGAII',    # Avaliable methods: NSGAII
#                             'metrics': 'mss_rmse',  # Metrics to be minimized (mss_rmse, mss_rho, mss_rmse_rho)
#                             'n_pop': 50,            # Number of individuals in the population
#                             'n_obj': 2,             # Number of objectives to be minimized 
#                             'generations': 1000,    # Number of generations for the calibration algorithm
#                             })              

config = xr.Dataset(coords={'dt': 1,                  # [hours]
                            'switch_Yini': 1,         # Calibrate the initial S position? (0: No, 1: Yes)
                            'crossshore' : 'YA',      # EBSEM for cross-shore (MD, YA, SF, JA, LIM)
                            'switch_alpha_ini': 0,    # Calibrate the initial alpha position? (0: No, 1: Yes)
                            'longshore' : 'JA',       # EBSEM for longshore (TU, JA)
                            
                            'depth': 10,              # Water depth [m] (MD, TU)
                            'D50': .3e-3,             # Median grain size [m] (MD, TU)
                            'bathy_angle': -30,      # Bathymetry mean orientation [deg N] (MD, TU)
                            'break_type': 'spectral', # Breaking type (spectral or linear) (MD)
                            'Hberm': 1,               # Berm height [m] (MD)
                            'flagP': 1,               # Parameter Proportionality (MD)
                            'BeachL': 3200,           # Beach Length [m] (TU)
                            
                            'Ysi': 2017,              # Initial year for calibration
                            'Msi': 4,                 # Initial month for calibration
                            'Dsi': 6,                 # Initial day for calibration
                            'Ysf': 2020,              # Final year for calibration
                            'Msf': 1,                 # Final month for calibration
                            'Dsf': 26,                 # Final day for calibration
                            'cal_alg': 'sceua',       # Avaliable methods: sceua
                            'metrics': 'rmse',        # Metrics to be minimized (mss, RP, rmse, nsse)
                            'repetitions': 10000000      # Number of repetitions for the calibration algorithm
                            })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/src/IHSetMOOSE/tests/Data_SRM/config.nc', engine='netcdf4')

model = calibration.cal_IH_MOOSE(wrkDir+'/src/IHSetMOOSE/tests/Data_SRM/', wrkDir+'/src/IHSetMOOSE/tests/Data_SRM/')

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

fig, ax = plt.subplots(1 , 1, figsize=(10, 2), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5]})

ylim_lower = np.floor(np.min([np.nanmin(model.long.Obs), np.nanmin(model.long.full_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.long.Obs), np.nanmax(model.long.full_run)]) / 2) * 2
ax.scatter(model.long.time_obs, model.long.Obs,s = 1, c = 'grey', label = 'Observed data')
ax.plot(model.long.time, model.long.full_run, color='red',linestyle='solid', label= 'EBSEM-longshore')
ax.fill([model.long.start_date, model.long.end_date, model.long.end_date, model.long.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax.set_ylim([ylim_lower,ylim_upper])
ax.set_xlim([model.long.time[0], model.long.time[-1]])
ax.set_ylabel('alpha [deg]', fontdict=font)
ax.legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/EBSEM_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

# Calibration:
rmse_l = spt.objectivefunctions.rmse(model.long.observations, model.long.best_simulation)
nsse_l = spt.objectivefunctions.nashsutcliffe(model.long.observations, model.long.best_simulation)
mss_l = mielke_skill_score(model.long.observations, model.long.best_simulation)
rp_l = spt.objectivefunctions.rsquared(model.long.observations, model.long.best_simulation)
bias_l = spt.objectivefunctions.bias(model.long.observations, model.long.best_simulation)

# Validation:
run_cut_l = model.long.full_run[model.long.idx_validation]
rmse_v_l = spt.objectivefunctions.rmse(model.long.Obs[model.long.idx_validation_obs], run_cut_l[model.long.idx_validation_for_obs])
nsse_v_l = spt.objectivefunctions.nashsutcliffe(model.long.Obs[model.long.idx_validation_obs], run_cut_l[model.long.idx_validation_for_obs])
mss_v_l = mielke_skill_score(model.long.Obs[model.long.idx_validation_obs], run_cut_l[model.long.idx_validation_for_obs])
rp_v_l = spt.objectivefunctions.rsquared(model.long.Obs[model.long.idx_validation_obs], run_cut_l[model.long.idx_validation_for_obs])
bias_v_l = spt.objectivefunctions.bias(model.long.Obs[model.long.idx_validation_obs], run_cut_l[model.long.idx_validation_for_obs])

def matlab_datenum(dt):
    dt = np.atleast_1d(dt)
    ordinal_dates = np.array([d.toordinal() for d in dt])
    frac_of_day = np.array([d.hour / 24.0 + d.minute / (24.0 * 60) + d.second / (24.0 * 3600) for d in dt])
    return ordinal_dates + frac_of_day + 366

matlab_datenums = matlab_datenum(model.long.time)

data_dict = {
    'TimeJA': matlab_datenums,
    'JA': model.long.full_run
}

savemat('output.mat', data_dict)

print('**********************************************************')
print('EBSEM - Longshore')
print('Metrics                       | Calibration  | Validation|')
print('RMSE [m]                      | %-5.2f        | %-5.2f     |' % (rmse_l, rmse_v_l))
print('Nash-Sutcliffe coefficient [-]| %-5.2f        | %-5.2f     |' % (nsse_l, nsse_v_l))
print('Mielke Skill Score [-]        | %-5.2f        | %-5.2f     |' % (mss_l, mss_v_l))
print('R2 [-]                        | %-5.2f        | %-5.2f     |' % (rp_l, rp_v_l))
print('Bias [m]                      | %-5.2f        | %-5.2f     |' % (bias_l, bias_v_l))

config.close()
