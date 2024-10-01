import os
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
from .ih_moose_jit import ih_moose_jit_par1, ih_moose_jit_par2

class ih_moose(object):
    
    def __init__(self, path_prof, model, **kwargs):
        """
        Jaramillo et al. 2021 model
        """
        S = model.cross.full_run
        alp = model.long.full_run

        Fmean = kwargs['Fmean']
        profN = kwargs['profN']
        pivotNi = kwargs['pivotNi']
        T = kwargs['T']
        depth = kwargs['depth']
                    
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))
        prof = pd.read_csv(path_prof+'prof.csv')
        prof = prof.values
        prof = prof[profN,:]
        self.profN = profN
        self.npro = prof.shape[0]
        
        indice = np.where(np.array(profN) == pivotNi)
        pivotN = indice[0][0]
        DirN = model.DirN[0]
        Dif = np.abs(DirN - Fmean)
        
        dX = - (S * np.cos(np.radians(Dif)))
        delta_alpha = alp - np.mean(alp)
        delta_alpha = delta_alpha * np.pi / 180
    
        self.Obs = []
        self.Obs_valiation = []
        self.time_obs = []
        self.idx_validation = []
        self.idx_validation_obs = []
        self.idx_validation_for_obs = []
        
        for i in range(self.npro):
            ens = xr.open_dataset(os.path.join(path_prof, 'ens_prof' + str(profN[i]+1) + '.nc'))
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
        

        parabola_num = kwargs['parabola_num']        
        self.parabola_num = parabola_num
        if parabola_num == 1:
            Cp = kwargs['Cp']
            Cl = model.prof_orgin
            Lr = kwargs['Lr']
            self.S_PF = ih_moose_jit_par1(prof, pivotN, Fmean, Cp, Cl, T, depth, Lr, dX, delta_alpha, gamd)
        
        if parabola_num == 2:
            Cp1 = kwargs['Cp1']
            Cp2 = kwargs['Cp2']
            Cl = model.prof_orgin
            Lr = 0
            self.S_PF = ih_moose_jit_par2(prof, pivotN, Fmean, Cp1, Cp2, Cl, T, depth, Lr, dX, delta_alpha, gamd)
