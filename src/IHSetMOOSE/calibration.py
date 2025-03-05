# import numpy as np
# import xarray as xr
# import spotpy as spt
# from IHSetCalibration import setup_spotpy
# from IHSetUtils import BreakingPropagation
# from IHSetMillerDean import calibration as cal_MD
# from IHSetMillerDean import millerDean
# from IHSetYates09 import calibration as cal_YA
# from IHSetYates09 import yates09
# from IHSetShoreFor import calibration as cal_SF
# from IHSetShoreFor import shoreFor
# from IHSetJara import calibration as cal_JR
# from IHSetJara import jara
# from IHSetJaramillo20 import calibration as cal_JA20
# from IHSetJaramillo20 import jaramillo20
# from IHSetLim import calibration as cal_LIM
# from IHSetLim import lim
# from IHSetTurki import calibration as cal_TU
# from IHSetTurki import turki
# from IHSetJaramillo21a import calibration as cal_JA21a
# from IHSetJaramillo21a import jaramillo21a

# class cal_IH_MOOSE(object):
#     """
#     cal_IH_MOOSE
    
#     Configuration to calibfalse,and run the Jaramillo et al. (2020) Hybrid Model.
    
#     This class reads input datasets, performs its calibration.
#     """
#     def __init__(self, path_cross, path_long, **kwargs):
#         self.path_cross = path_cross
#         self.path_long = path_long
#         cfg = xr.open_dataset(path_cross+'config.nc')
        
#         # self.prof_orgin = kwargs['prof_orgin']
#         # self.DirN = kwargs['DirN']
        
#         self.crossshore = cfg['crossshore'].values
#         self.switch_Yini = cfg['switch_Yini'].values
#         self.switch_alpha_ini = cfg['switch_alpha_ini'].values
        
#         if self.crossshore == 'MD':
#             # wav = xr.open_dataset(path_cross+'wav.nc')
#             # Hb, Dirb, depthb = BreakingPropagation(wav['Hs'].values,
#             #                            wav['Tp'].values,
#             #                            wav['Dir'].values,
#             #                            np.full_like(wav['Hs'].values, cfg['depth'].values),
#             #                            np.full_like(wav['Hs'].values, cfg['bathy_angle'].values),
#             #                            cfg['break_type'].values)
#             # wav['Hb'] = xr.DataArray(Hb, dims = 'Y', coords = {'Y': wav['Y']})
#             # wav['Dirb'] = xr.DataArray(Dirb, dims = 'Y', coords = {'Y': wav['Y']})
#             # wav['depthb'] = xr.DataArray(depthb, dims = 'Y', coords = {'Y': wav['Y']})
#             # wav.to_netcdf(path_cross+'wavb.nc', engine='netcdf4')
#             # wav.close()
            
#             self.cross = cal_MD.cal_MillerDean(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
            
#             self.cross.kero = self.cross.best_model_run['parkero']
#             self.cross.kacr = self.cross.best_model_run['parkacr']
#             self.cross.Y0 = self.cross.best_model_run['parY0']
#             if self.switch_Yini == 0:
#                 self.cross.full_run, _ = millerDean(self.cross.Hb, self.cross.depthb, self.cross.sl, self.cross.wast, self.cross.dt, self.cross.Hberm,
#                          self.cross.Y0, self.cross.kero, self.cross.kacr, self.cross.Obs[0], self.cross.flagP, self.cross.Omega)
#             if self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.full_run, _ = millerDean(self.cross.Hb, self.cross.depthb, self.cross.sl, self.cross.wast, self.cross.dt, self.cross.Hberm,
#                          self.cross.Y0, self.cross.kero, self.cross.kacr, self.cross.Yini, self.cross.flagP, self.cross.Omega)
            
#         if self.crossshore == 'YA':
#             self.cross = cal_YA.cal_Yates09(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
                
#             self.cross.a = -self.cross.best_model_run['para']
#             self.cross.b = self.cross.best_model_run['parb']
#             self.cross.cacr = -self.cross.best_model_run['parcacr']
#             self.cross.cero = -self.cross.best_model_run['parcero']
#             if self.switch_Yini == 0:
#                 self.cross.full_run, _ = yates09(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Obs[0])
#             if self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.full_run, _ = yates09(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Yini)

#         if self.crossshore == 'SF':
#             self.switch_D = cfg['switch_D'].values
#             self.switch_r = cfg['switch_r'].values
#             self.cross = cal_SF.cal_ShoreFor(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
        
#             if self.switch_Yini == 0:
#                 if self.switch_D == 0 and self.switch_r == 0:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       2*self.cross.phi, self.cross.Obs[0], self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 1 and self.switch_r == 0:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.D = self.cross.best_model_run['parD']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       self.cross.D, self.cross.Obs[0], self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 0 and self.switch_r == 1:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       2*self.cross.phi, self.cross.Obs[0], self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 1 and self.switch_r == 1:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.D = self.cross.best_model_run['parD']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       self.cross.D, self.cross.Obs[0], self.cross.cp, self.cross.cm)
                                
#             if self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 if self.switch_D == 0 and self.switch_r == 0:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       2*self.cross.phi, self.cross.Yini, self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 1 and self.switch_r == 0:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.D = self.cross.best_model_run['parD']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       self.cross.D, self.cross.Yini, self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 0 and self.switch_r == 1:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       2*self.cross.phi, self.cross.Yini, self.cross.cp, self.cross.cm)
#                 elif self.switch_D == 1 and self.switch_r == 1:
#                     self.cross.phi = self.cross.best_model_run['parphi']
#                     self.cross.cp = self.cross.best_model_run['parcp']
#                     self.cross.cm = self.cross.best_model_run['parcm']
#                     self.cross.D = self.cross.best_model_run['parD']
#                     self.cross.full_run, _ = shoreFor(self.cross.P, self.cross.Omega, self.cross.dt, self.cross.phi,
#                                                       self.cross.D, self.cross.Yini, self.cross.cp, self.cross.cm)

#         if self.crossshore == 'JR':
#             self.cross = cal_JR.cal_Jara(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
            
#             self.cross.Ca = self.cross.best_model_run['parCa']
#             self.cross.Ce = self.cross.best_model_run['parCe']
#             if self.switch_Yini == 0:
#                 self.cross.full_run = jara.jara(self.cross.Hb, self.cross.Hcr, self.cross.Obs[0], self.cross.dt, self.cross.gamma, self.cross.xc, self.cross.hc,
#                                                 self.cross.B, self.cross.Ar, self.cross.hb_, self.cross.xre_, self.cross.pol, self.cross.Vol, self.cross.Ca, self.cross.Ce)
#             elif self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.full_run = jara.jara(self.cross.Hb, self.cross.Hcr, self.cross.Yini, self.cross.dt, self.cross.gamma, self.cross.xc, self.cross.hc,
#                                                 self.cross.B, self.cross.Ar, self.cross.hb_, self.cross.xre_, self.cross.pol, self.cross.Vol, self.cross.Ca, self.cross.Ce)        

#         if self.crossshore == 'JA':
#             self.switch_vlt = cfg['switch_vlt'].values
#             if self.switch_vlt == 0:
#                 self.vlt = cfg['vlt'].values
#             self.cross = cal_JA20.cal_Jaramillo20(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
            
#             self.cross.a = -self.cross.best_model_run['para']
#             self.cross.b = self.cross.best_model_run['parb']
#             self.cross.cacr = -self.cross.best_model_run['parcacr']
#             self.cross.cero = -self.cross.best_model_run['parcero']
#             if self.switch_vlt == 0 and self.switch_Yini == 0:
#                 self.cross.full_run, _ = jaramillo20(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Obs[0], self.vlt)
#             elif self.switch_vlt == 0 and self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.full_run, _ = jaramillo20(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Yini, self.vlt)
#             elif self.switch_vlt == 1 and self.switch_Yini == 0:
#                 self.cross.vlt = self.cross.best_model_run['parvlt']
#                 self.cross.full_run, _ = jaramillo20(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Obs[0], self.cross.vlt)
#             elif self.switch_vlt == 1 and self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.vlt = self.cross.best_model_run['parvlt']
#                 self.cross.full_run, _ = jaramillo20(self.cross.E, self.cross.dt, self.cross.a, self.cross.b, self.cross.cacr, self.cross.cero, self.cross.Yini, self.cross.vlt)
        
#         if self.crossshore == 'LIM':
#             self.cross = cal_LIM.cal_Lim(self.path_cross)
#             setup = setup_spotpy(self.cross)
#             results = setup.setup()
#             bestindex, self.cross.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.cross.best_model_run = results[bestindex]
#             fields=[word for word in self.cross.best_model_run.dtype.names if word.startswith('sim')]
#             self.cross.best_simulation = list(self.cross.best_model_run[fields])
            
#             self.cross.kr = self.cross.best_model_run['parkr']
#             self.cross.mu = self.cross.best_model_run['parmu']
#             if self.switch_Yini == 0:
#                 self.cross.full_run = lim.lim(self.cross.Hb, self.cross.dt, self.cross.A, self.cross.mf, self.cross.kr, self.cross.mu, self.cross.Sm, self.cross.Obs[0])
#             elif self.switch_Yini == 1:
#                 self.cross.Yini = self.cross.best_model_run['parYini']
#                 self.cross.full_run = lim.lim(self.cross.Hb, self.cross.dt, self.cross.A, self.cross.mf, self.cross.kr, self.cross.mu, self.cross.Sm, self.cross.Yini)
                        
#         self.longshore = cfg['longshore'].values
        
#         if self.longshore == 'TU':
#             self.long = cal_TU.cal_Turki(self.path_long)
#             setup = setup_spotpy(self.long)
#             results = setup.setup()
#             bestindex, self.long.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.long.best_model_run = results[bestindex]
#             fields=[word for word in self.long.best_model_run.dtype.names if word.startswith('sim')]
#             self.long.best_simulation = list(self.long.best_model_run[fields])
            
#             self.long.kk = self.long.best_model_run['parkk']
#             if self.switch_alpha_ini == 0:
#                 self.long.full_run, _ = turki.turki(self.long.EF, self.long.Hb, self.long.theb, self.long.BeachL, self.long.dt, self.long.kk, self.long.Obs[0])
#             if self.switch_alpha_ini == 1:
#                 self.long.alp0 = self.long.best_model_run['paralp0']
#                 self.long.full_run, _ = turki.turki(self.long.EF, self.long.Hb, self.long.theb, self.long.BeachL, self.long.dt, self.long.kk, self.long.alp0)    
        
#         if self.longshore == 'JA':
#             self.long = cal_JA21a.cal_Jaramillo21a(self.path_long)
#             setup = setup_spotpy(self.long)
#             results = setup.setup()
#             bestindex, self.long.bestobjf = spt.analyser.get_minlikeindex(results)
#             self.long.best_model_run = results[bestindex]
#             fields=[word for word in self.long.best_model_run.dtype.names if word.startswith('sim')]
#             self.long.best_simulation = list(self.long.best_model_run[fields])
                        
#             self.long.a = self.long.best_model_run['para']
#             self.long.b = self.long.best_model_run['parb']
#             self.long.Lcw = self.long.best_model_run['parLcw']
#             self.long.Lccw = self.long.best_model_run['parLccw']
#             if self.switch_alpha_ini == 0:
#                 self.long.full_run, _ = jaramillo21a(self.long.P, self.long.Dir, self.long.dt, self.long.a, self.long.b, self.long.Lcw, self.long.Lccw, self.long.Obs[0])
#             if self.switch_alpha_ini == 1:
#                 self.long.alp0 = self.long.best_model_run['paralpha_ini']
#                 self.long.full_run, _ = jaramillo21a(self.long.P, self.long.Dir, self.long.dt, self.long.a, self.long.b, self.long.Lcw, self.long.Lccw, self.long.alp0)
            
