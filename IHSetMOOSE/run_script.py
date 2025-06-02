import xarray as xr
from IHSetMOOSE import ih_moose
import numpy as np
import json
from pyproj import Transformer
import time as t_
from IHSetYates09 import cal_Yates09_2, Yates09_run
from IHSetShoreFor import cal_ShoreFor_2, ShoreFor_run
from IHSetJaramillo20 import cal_Jaramillo20_2, Jaramillo20_run
from IHSetLim import cal_Lim_2, Lim_run
from IHSetMillerDean import cal_MillerDean_2, MillerDean_run
from IHSetJaramillo21a import cal_Jaramillo21a_2, Jaramillo21a_run
from IHSetTurki import cal_Turki_2, Turki_run
from IHSetJara import cal_Jara_2, Jara_run
import traceback
import datetime


def run_script(self, path, current_params=None):
        
        self.path = path
        data = xr.open_dataset(path)
        cfg = json.loads(data.attrs['IH-MOOSE'])
        self.crossshore_selected = cfg['cross']['name']
        self.longshore_selected = cfg['long']['name']

        #crear variable con la ruta del archivo Narrabeen8_md.nc
        self.trs = ih_moose.find_min_distance(data.x_pivotal.values, data.y_pivotal.values, data.xi.values, data.yi.values, data.xf.values, data.yf.values)


        starting_time = t_.time()
        print("Running ", self.selected_crossshore, " model...")
        print('Starting JIT compilation...')
        #CORRER MODELO CROSSSHORE
        if self.selected_crossshore == "Yates": 
            if cfg['mode']== "calibrate":
                self.model_cross = cal_Yates09_2(path)
                self.model_cross.calibrate()
            else:
                self.model_cross = Yates09_run(path)
                self.model_cross.run(current_params)
                print("--------Model standalone finished-------")

        elif self.selected_crossshore == "Davidson":
            if cfg['mode']== "calibrate":
                self.model_cross = cal_ShoreFor_2(path)
                self.model_cross.calibrate()
            else:                 
                self.model_cross = ShoreFor_run(path)
                self.model_cross.run(current_params)

        elif self.selected_crossshore == "Jaramillo20":
            if cfg['mode']== "calibrate":
                self.model_cross = cal_Jaramillo20_2(path)
                self.model_cross.calibrate()

            else:
                
                self.model_cross = Jaramillo20_run(path)
                self.model_cross.run(current_params)

        elif self.selected_crossshore == "Lim":
            if cfg['mode']== "calibrate":
                self.model_cross = cal_Lim_2(path)
                self.model_cross.calibrate()

            else:
                
                self.model_cross = Lim_run(path)
                self.model_cross.run(current_params)

        elif self.selected_crossshore == "Miller and Dean":
            if cfg['mode']== "calibrate":
                self.model_cross = cal_MillerDean_2(path)
                self.model_cross.calibrate()

            else:
            
                # Ejecuta el modelo
                self.model_cross = MillerDean_run(path)
                self.model_cross.run(current_params)
                    
        elif self.selected_crossshore == "Jara":
            if cfg['mode']== "calibrate":
                
                self.model_cross = cal_Jara_2(path)
                self.model_cross.calibrate()

            else:
                self.model_cross = Jara_run(path)
                self.model_cross.run(current_params)

        print(f"Model {self.selected_crossshore} finished in {t_.time() - starting_time:.2f} seconds.")

        self.full_run_cross = self.model_cross.full_run
        par_names = self.model_cross.par_names
        par_values = self.model_cross.par_values

        
        for name, value in zip(par_names, par_values):
            print(f"{name}: {value}")

        #CORRER MODELO ROTATION

        if self.selected_longshore == "Jaramillo21a":
            if self.radio_calibrate_long.isChecked():
                
                # Verifica se o arquivo existe
                model_long = cal_Jaramillo21a_2(path)

                model_long.calibrate()
                
            else:
                
                self.model_long = Jaramillo21a_run(path)
                self.model_long.run(current_params)
                self.full_run_long= self.model_cross.full_run


        elif self.selected_longshore == "Turki":
            if self.radio_calibrate_long.isChecked():
                
                # Verifica se o arquivo existe
                model_long = cal_Turki_2(path)

                model_long.calibrate()
                
            else :
                
                # Ejecuta el modelo
                self.model_long = Turki_run(path)
                self.model_long.run(current_params)

        print(f"Model {self.selected_longshore} finished in {t_.time() - starting_time:.2f} seconds.")
        self.full_run_long = self.model_long.full_run
        # Imprimir los parámetros
        par_names = self.model_long.par_names
        par_values = self.model_long.par_values            

        for name, value in zip(par_names, par_values):
            print(f"{name}: {value}")
        
        # Now we create the IH-MOOSE object
        print("Creating IH-MOOSE object...")

        ### Step 1: Create planform and profile
        print("Creating planform and profile...")
        planform = ih_moose.equilibrium_planform(path)

        ### Step 3: Simulate IH-MOOSE
        print("Running IH-MOOSE...")
        self.results = ih_moose.ih_moose(path, planform, self.full_run_cross, self.full_run_long)

        print('Time elapsed: %.2f seconds' % (t_.time() - starting_time))

