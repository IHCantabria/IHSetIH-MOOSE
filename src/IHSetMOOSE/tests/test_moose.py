from IHSetMOOSE import calibration, ih_moose
from IHSetCalibration import mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np
import time
    
# Avaliable methods: NSGAII, mle, mc, dds, mcmc, sa, abc, lhs, rope, sceua, demcz, padds, fscabc

# config = xr.Dataset(coords={'dt': 1,                  # [hours]
#                             'switch_Yini': 1,         # Calibrate the initial S position? (0: No, 1: Yes)
#                             'crossshore' : 'YA',      # EBSEM for cross-shore (MD, YA, SF, JR, JA, LIM)
#                             'switch_alpha_ini': 1,    # Calibrate the initial alpha position? (0: No, 1: Yes)
#                             'longshore' : 'JA',       # EBSEM for longshore (TU, JA)
                            
#                             'depth': 20,              # Water depth [m] (MD, TU, JR, LIM)
#                             'D50': .3e-3,             # Median grain size [m] (MD, TU, SF, JR, LIM)
#                             'bathy_angle': 7.2,    # Bathymetry mean orientation [deg N] (MD, TU, JR, LIM)
#                             'break_type': 'spectral', # Breaking type (spectral or linear) (MD, TU)
#                             'xc' : 255,               # Cross-shore distance from shore to closure depth [m] (JR)
#                             'hc' : 6,                 # Depth of closure [m] (JR)
#                             'Hberm': 1,               # Berm height [m] (MD, JR)
#                             'flagP': 3,               # Parameter Proportionality (MD)
#                             'switch_D': 0,            # Calibrate D independently? (0: No, 1: Yes) (SF)
#                             'switch_r': 0,            # Calibrate r independently? (0: No, 1: Yes) (SF)
#                             'mf' : 0.02,              # Proflie slope (LIM)
#                             'switch_vlt': 0,          # Calibrate the longterm trend? (0: No, 1: Yes) (JA-cross)
#                             'vlt': 0,                 # Longterm trend [m] (JA-cross)
#                             'BeachL': 3200,           # Beach Length [m] (TU)
                                                        
#                             'Ysi': 1974,              # Initial year for calibration
#                             'Msi': 1,                 # Initial month for calibration
#                             'Dsi': 1,                 # Initial day for calibration
#                             'Ysf': 2016,              # Final year for calibration
#                             'Msf': 1,                 # Final month for calibration
#                             'Dsf': 1,                 # Final day for calibration
#                             'cal_alg': 'NSGAII',      # Avaliable methods: NSGAII
#                             'metrics': 'mss_rmse',    # Metrics to be minimized (mss_rmse, mss_rho, mss_rmse_rho)
#                             'n_pop': 50,              # Number of individuals in the population
#                             'n_obj': 2,               # Number of objectives to be minimized 
#                             'generations': 500,      # Number of generations for the calibration algorithm
#                             })              

config = xr.Dataset(coords={'dt': 1,                  # [hours]
                            'switch_Yini': 1,         # Calibrate the initial S position? (0: No, 1: Yes)
                            'crossshore' : 'YA',      # EBSEM for cross-shore (MD, YA, SF, JR, JA, LIM)
                            'switch_alpha_ini': 1,    # Calibrate the initial alpha position? (0: No, 1: Yes)
                            'longshore' : 'JA',       # EBSEM for longshore (TU, JA)
                            
                            'depth': 20,              # Water depth [m] (MD, TU, JR, LIM)
                            'D50': .3e-3,             # Median grain size [m] (MD, TU, SF, JR, LIM)
                            'bathy_angle': 7.2,    # Bathymetry mean orientation [deg N] (MD, TU, JR, LIM)
                            'break_type': 'spectral', # Breaking type (spectral or linear) (MD, TU)
                            'xc' : 255,               # Cross-shore distance from shore to closure depth [m] (JR)
                            'hc' : 6,                 # Depth of closure [m] (JR)
                            'Hberm': 1,               # Berm height [m] (MD, JR)
                            'flagP': 3,               # Parameter Proportionality (MD)
                            'switch_D': 0,            # Calibrate D independently? (0: No, 1: Yes) (SF)
                            'switch_r': 0,            # Calibrate r independently? (0: No, 1: Yes) (SF)
                            'mf' : 0.02,              # Proflie slope (LIM)
                            'switch_vlt': 0,          # Calibrate the longterm trend? (0: No, 1: Yes) (JA-cross)
                            'vlt': 0,                 # Longterm trend [m] (JA-cross)
                            'BeachL': 3200,           # Beach Length [m] (TU)
                            
                            'Ysi': 1975,              # Initial year for calibration
                            'Msi': 1,                 # Initial month for calibration
                            'Dsi': 1,                 # Initial day for calibration
                            'Ysf': 2015,              # Final year for calibration
                            'Msf': 1,                 # Final month for calibration
                            'Dsf': 1,                 # Final day for calibration
                            'cal_alg': 'sceua',       # Avaliable methods: sceua
                            'metrics': 'nsse',        # Metrics to be minimized (mss, RP, rmse, nsse)
                            'repetitions': 50000      # Number of repetitions for the calibration algorithm
                            })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/data_hybrid/Cross_shore/config.nc', engine='netcdf4')
config.to_netcdf(wrkDir+'/data_hybrid/Longshore/config.nc', engine='netcdf4')

# Calibration EBSEM
# model = calibration.cal_IH_MOOSE(wrkDir+'/data_hybrid/Cross_shore/', wrkDir+'/data_hybrid/Longshore/', prof_orgin=[342451.3627, 6267913.117], DirN=[100.26])
model = calibration.cal_IH_MOOSE(wrkDir+'/data_hybrid/Cross_shore/', wrkDir+'/data_hybrid/Longshore/')

# IH-MOOSE for one planform
results = ih_moose.ih_moose(wrkDir+'/data_hybrid/Profiles/', model, Fmean=109.2900, profN=[0, 1, 2, 3, 4], pivotNi=2,
                            Cp=[344915.384, 6266136.216], T=10, depth=20, Lr=1800, gamd=0, parabola_num=1)
# IH-MOOSE for two planforms
# results = ih_moose.ih_moose(wrkDir+'/data_hybrid/Profiles/', model, Fmean=109.2900, profN=[0, 1, 2, 3, 4], pivotNi=2,
#                             Cp1=[343537.569816416, 6269248.981867335], Cp2=[344915.384, 6266136.216], T=10, depth=20, gamd=10, parabola_num=2)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

######################################################################     EBSEM     ######################################################################
fig, ax = plt.subplots(2 , 1, figsize=(10, 2), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5, 3.5]})
ylim_lower = np.floor(np.min([np.nanmin(model.cross.Obs), np.nanmin(model.cross.full_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.cross.Obs), np.nanmax(model.cross.full_run)]) / 2) * 2
ax[0].scatter(model.cross.time_obs, model.cross.Obs,s = 1, c = 'grey', label = 'Observed data')
ax[0].plot(model.cross.time, model.cross.full_run, color='red',linestyle='solid', label= 'EBSEM-Cross_shore')
ax[0].fill([model.cross.start_date, model.cross.end_date, model.cross.end_date, model.cross.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax[0].set_ylim([ylim_lower,ylim_upper])
ax[0].set_xlim([model.cross.time[0], model.cross.time[-1]])
ax[0].set_ylabel('S [m]', fontdict=font)
ax[0].legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax[0].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

ylim_lower = np.floor(np.min([np.nanmin(model.long.Obs), np.nanmin(model.long.full_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.long.Obs), np.nanmax(model.long.full_run)]) / 2) * 2
ax[1].scatter(model.long.time_obs, model.long.Obs,s = 1, c = 'grey', label = 'Observed data')
ax[1].plot(model.long.time, model.long.full_run, color='red',linestyle='solid', label= 'EBSEM-longshore')
ax[1].fill([model.long.start_date, model.long.end_date, model.long.end_date, model.long.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax[1].set_ylim([ylim_lower,ylim_upper])
ax[1].set_xlim([model.long.time[0], model.long.time[-1]])
ax[1].set_ylabel('alpha [deg]', fontdict=font)
ax[1].legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax[1].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/EBSEM_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

# Calibration:
rmse_c = spt.objectivefunctions.rmse(model.cross.observations, model.cross.best_simulation)
nsse_c = spt.objectivefunctions.nashsutcliffe(model.cross.observations, model.cross.best_simulation)
mss_c = mielke_skill_score(model.cross.observations, model.cross.best_simulation)
rp_c = spt.objectivefunctions.rsquared(model.cross.observations, model.cross.best_simulation)
bias_c = spt.objectivefunctions.bias(model.cross.observations, model.cross.best_simulation)

# Validation:
run_cut_c = model.cross.full_run[model.cross.idx_validation]
rmse_v_c = spt.objectivefunctions.rmse(model.cross.Obs[model.cross.idx_validation_obs], run_cut_c[model.cross.idx_validation_for_obs])
nsse_v_c = spt.objectivefunctions.nashsutcliffe(model.cross.Obs[model.cross.idx_validation_obs], run_cut_c[model.cross.idx_validation_for_obs])
mss_v_c = mielke_skill_score(model.cross.Obs[model.cross.idx_validation_obs], run_cut_c[model.cross.idx_validation_for_obs])
rp_v_c = spt.objectivefunctions.rsquared(model.cross.Obs[model.cross.idx_validation_obs], run_cut_c[model.cross.idx_validation_for_obs])
bias_v_c = spt.objectivefunctions.bias(model.cross.Obs[model.cross.idx_validation_obs], run_cut_c[model.cross.idx_validation_for_obs])

print('**********************************************************')
print('EBSEM - Cross_shore')
print('Metrics                       | Calibration  | Validation|')
print('RMSE [m]                      | %-5.2f        | %-5.2f     |' % (rmse_c, rmse_v_c))
print('Nash-Sutcliffe coefficient [-]| %-5.2f        | %-5.2f     |' % (nsse_c, nsse_v_c))
print('Mielke Skill Score [-]        | %-5.2f        | %-5.2f     |' % (mss_c, mss_v_c))
print('R2 [-]                        | %-5.2f        | %-5.2f     |' % (rp_c, rp_v_c))
print('Bias [m]                      | %-5.2f        | %-5.2f     |' % (bias_c, bias_v_c))

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

print('**********************************************************')
print('EBSEM - Longshore')
print('Metrics                       | Calibration  | Validation|')
print('RMSE [m]                      | %-5.2f        | %-5.2f     |' % (rmse_l, rmse_v_l))
print('Nash-Sutcliffe coefficient [-]| %-5.2f        | %-5.2f     |' % (nsse_l, nsse_v_l))
print('Mielke Skill Score [-]        | %-5.2f        | %-5.2f     |' % (mss_l, mss_v_l))
print('R2 [-]                        | %-5.2f        | %-5.2f     |' % (rp_l, rp_v_l))
print('Bias [m]                      | %-5.2f        | %-5.2f     |' % (bias_l, bias_v_l))

######################################################################     IH-MOOSE Results     ######################################################################
fig, ax = plt.subplots(results.npro , 1, figsize=(10, results.npro*1.5), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5] * results.npro})
rmse = np.zeros([results.npro,1])
nsse = np.zeros([results.npro,1])
mss = np.zeros([results.npro,1])
rp = np.zeros([results.npro,1])
bias = np.zeros([results.npro,1])

for i in range(results.npro):
        ylim_lower = np.floor(np.min([np.nanmin(results.Obs[i]), np.nanmin(results.S_PF[:,i])]) / 2) * 2
        ylim_upper = np.ceil(np.max([np.nanmax(results.Obs[i]), np.nanmax(results.S_PF[:,i])]) / 2) * 2
        ax[i].scatter(results.time_obs[i], results.Obs[i], s = 1, c = 'grey', label = 'Observed data')
        ax[i].plot(model.cross.time, results.S_PF[:,i], color='red',linestyle='solid', label= 'IH-MOOSE')
        ax[i].set_ylim([ylim_lower,ylim_upper])
        ax[i].set_xlim([model.cross.time[0], model.cross.time[-1]])
        ax[i].set_ylabel('S [m]', fontdict=font)
        ax[i].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
        ax[i].set_title(f"Profile {results.profN[i] + 1}", fontdict=font)
        
        Observations = results.Obs[i]
        run = results.S_PF[:,i]
        run_cut = run[results.idx_validation[i]]
        rmse[i] = spt.objectivefunctions.rmse(Observations[results.idx_validation_obs[i]], run_cut[results.idx_validation_for_obs[i]])
        nsse[i] = spt.objectivefunctions.nashsutcliffe(Observations[results.idx_validation_obs[i]], run_cut[results.idx_validation_for_obs[i]])
        mss[i] = mielke_skill_score(Observations[results.idx_validation_obs[i]], run_cut[results.idx_validation_for_obs[i]])
        rp[i] = spt.objectivefunctions.rsquared(Observations[results.idx_validation_obs[i]], run_cut[results.idx_validation_for_obs[i]])
        bias[i] = spt.objectivefunctions.bias(Observations[results.idx_validation_obs[i]], run_cut[results.idx_validation_for_obs[i]])
ax[0].legend(ncol = 6, prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.55))

for i in range(results.npro):
        print('**********************************************************')
        print('Hybrid model (', results.parabola_num, 'Parabolic )')
        print('Metrics - Profile', results.profN[i]+1, '           | Validation |')
        print('RMSE [m]                       | %-5.2f      |' % (rmse[i][0]))
        print('Nash-Sutcliffe coefficient [-] | %-5.2f      |' % (nsse[i][0]))
        print('Mielke Skill Score [-]         | %-5.2f      |' % (mss[i][0]))
        print('R2 [-]                         | %-5.2f      |' % (rp[i][0]))
        print('Bias [m]                       | %-5.2f      |' % (bias[i][0]))

plt.subplots_adjust(hspace=0.6)
fig.savefig('./results/IH-MOOSE_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

config.close()

