import os
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
from .ih_moose_jit import ih_moose_jit

class ih_moose(object):
    
    def __init__(self, path_prof, model, Fmean, pivotN, Cp, Cl, T, depth, Lr):
        """
        Jaramillo et al. 2021 model
        """
        S = model.cross.full_run
        alp = model.long.full_run
    
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
        prof = pd.read_csv(path_prof+'prof.csv')
        prof = prof.values
        self.npro = prof.shape[0]
        pivotDir = prof[pivotN,0]
        Dif = np.abs(pivotDir - Fmean)

        dX =  - (S * np.cos(np.radians(Dif)))
        delta_alpha = alp - np.mean(alp)
        delta_alpha = delta_alpha * np.pi / 180
    
        self.Obs = []
        self.Obs_valiation = []
        self.time_obs = []
        self.idx_validation = []
        self.idx_validation_obs = []
        self.idx_validation_for_obs = []
        for i in range(self.npro):
            ens = xr.open_dataset(os.path.join(path_prof, 'ens_prof' + str(i+1) + '.nc'))
            Obs_o = ens['Obs'].values
            time_obs_o = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)
            self.Obs.append(Obs_o)
            self.time_obs.append(time_obs_o)
            
            idx = np.where((model.cross.time < model.cross.start_date) | (model.cross.time > model.cross.end_date))[0]
            self.idx_validation.append(idx)
            idx_validation_obs_o = np.where((time_obs_o < model.cross.start_date) | (time_obs_o > model.cross.end_date))[0]
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(model.cross.time[idx] - t)))
            idx_validation_for_obs_o = mkIdx(time_obs_o[idx_validation_obs_o])
            self.idx_validation_obs.append(idx_validation_obs_o)
            self.idx_validation_for_obs.append(idx_validation_for_obs_o)
        
        self.S_PF = ih_moose_jit(prof, pivotN, Fmean, Cp, Cl, T, depth, Lr, dX, delta_alpha)
        