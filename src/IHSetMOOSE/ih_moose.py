import xarray as xr
import numpy as np
import json
import pandas as pd
from IHSetUtils import hunt
from IHSetUtils.geometry import nauticalDir2cartesianDirP as n2c

from IHSetMillerDean import calibration_2 as cal_MD
from IHSetMillerDean import direct_run as MD_run

from IHSetYates09 import calibration_2 as cal_YA
from IHSetYates09 import direct_run as YA_run

from IHSetShoreFor import calibration_2 as cal_SF
from IHSetShoreFor import direct_run as SF_run

from IHSetJara import calibration_2 as cal_JR
from IHSetJara import direct_run as JR_run

from IHSetJaramillo20 import calibration_2 as cal_JA20
from IHSetJaramillo20 import direct_run as JA20_run

from IHSetLim import calibration_2 as cal_LIM
from IHSetLim import direct_run as LIM_run

from IHSetTurki import calibration_2 as cal_TU
from IHSetTurki import direct_run as TU_run

from IHSetJaramillo21a import calibration_2 as cal_JA21a
from IHSetJaramillo21a import direct_run as JA21a_run

from .ih_moose_jit import ih_moose_jit_par1, ih_moose_jit_par2, gonzalez_ih_moose, initialize_array, intersect_with_min_distance

class ih_moose(object):
    
    def __init__(self, path, planform, model):
        """
        Jaramillo et al. 2021 model
        """
        self.path = path
        data = xr.open_dataset(path)
        
        self.trs = model.trs        
        S = model.cross_run
        alp = model.long_run
        
        if self.trs == 'Average':
            print('Please select the specific transect')
        else:
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.ntrs_xi = data.xi.values
            self.ntrs_yi =  data.yi.values
            self.ntrs_xf =  data.xf.values
            self.ntrs_yf =  data.yf.values
            ntrs =  data.ntrs.values
            self.ntrs = ntrs[-1] + 1
        
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)
                
        DirN = 90 - np.arctan((self.ntrs_yi[self.trs] - self.ntrs_yf[self.trs]) / (self.ntrs_xi[self.trs] - self.ntrs_xf[self.trs])) * 180 / np.pi
        self.Cl = [self.ntrs_xi[self.trs], self.ntrs_yi[self.trs]]
        
        Dif = np.abs(DirN - planform.Fmean)
        dX = - (S * np.cos(np.radians(Dif)))
        
        delta_alpha = alp - np.mean(alp)
        delta_alpha = delta_alpha * np.pi / 180
    
        if planform.parabola_num == 1:
            print('Start Simulating IH-MOOSE...')                           
            xyi = np.vstack((planform.prof[:, 1], planform.prof[:, 2]))
            pivot_point = np.array([data.x_pivotal.values, data.x_pivotal.values]).reshape(2, 1)
            pivotN = np.argmin(np.linalg.norm(xyi.T - pivot_point.T, axis=1))
            
            self.S_PF, self.costas_x, self.costas_y = ih_moose_jit_par1(planform.prof, pivotN, planform.Fmean, planform.Cp, self.Cl, planform.T, planform.depth, planform.Lr, dX, delta_alpha, 0)
        
        if planform.parabola_num == 2:
            print('Start Simulating IH-MOOSE...')   
            xyi = np.vstack((planform.prof[:, 1], planform.prof[:, 2]))
            pivot_point = np.array([data.x_pivotal.values, data.x_pivotal.values]).reshape(2, 1)
            pivotN = np.argmin(np.linalg.norm(xyi.T - pivot_point.T, axis=1))
            
            self.S_PF, self.costas_x, self.costas_y = ih_moose_jit_par2(planform.prof, pivotN, planform.Fmean, planform.Cp1, planform.Cp2, self.Cl, planform.T, planform.depth, 0, dX, delta_alpha, 0)
                
        self.SS = initialize_array((len(dX), self.ntrs), np.float64)
        
        for j in range(self.ntrs):
            print('Calculating Transect', j+1, 'Shoreline Position...')   
            prof = [self.ntrs_xi[j], self.ntrs_xi[j], self.ntrs_yi[j], self.ntrs_xf[j], self.ntrs_yf[j]]
            for i in range(len(dX)):
                xf, yf = intersect_with_min_distance(self.costas_x[i,:], self.costas_y[i,:], prof)
                squared_distance = ((xf - self.ntrs_xi[j])**2 + (yf - self.ntrs_yi[j])**2)
                self.SS[i, j] = np.sqrt(squared_distance)
            
class equilibrium_planform(object):
    
    def __init__(self, path):
        """
        Jaramillo et al. 2021 model
        """
        self.path = path
        data = xr.open_dataset(path)
        cfg = json.loads(data.attrs['IH_MOOSE'])
        
        # self.lim = cfg['lim']
        self.Fmean = cfg['Fmean']
        self.T = cfg['T']
        self.depth = cfg['hd']
        self.parabola_num = cfg['parabola_num']
        
        # self.npro = cfg['npro']
        self.lpro = cfg['lpro']
        # self.prof = np.zeros((self.npro, 5))
        
        self.Cl = cfg['Cl']
        
        if self.parabola_num == 1:
            self.Cp = cfg['Cp']
            self.Lr = cfg['Lr']
            self.costa_xe, self.costa_ye, self.alpha_curve = gonzalez_ih_moose(self.Fmean, self.Cp, self.Cl, self.T, self.depth, self.Lr, 0, 0)
            self.costa_xi, self.costa_yi, _ = gonzalez_ih_moose(self.Fmean, self.Cp, self.Cl, self.T, self.depth, self.Lr, 0, self.lpro/2)
            self.costa_xf, self.costa_yf, _ = gonzalez_ih_moose(self.Fmean, self.Cp, self.Cl, self.T, self.depth, self.Lr, 0, -self.lpro/2)
            
        if self.parabola_num == 2:
            self.Cp1 = cfg['Cp1']
            self.Cp2 = cfg['Cp2']
            costa_xe1, costa_ye1, self.alpha_curve1 = gonzalez_ih_moose(self.Fmean, self.Cp1, self.Cl, self.T, self.depth, 0, 0, 0)
            costa_xe2, costa_ye2, self.alpha_curve2 = gonzalez_ih_moose(self.Fmean, self.Cp2, self.Cl, self.T, self.depth, 0, 0, 0)
            self.costa_xe = combine_arrays(costa_xe1, costa_xe2)
            self.costa_ye = combine_arrays(costa_ye1, costa_ye2)
            
            costa_xi1, costa_yi1, _ = gonzalez_ih_moose(self.Fmean, self.Cp1, self.Cl, self.T, self.depth, 0, 0, self.lpro/2)
            costa_xi2, costa_yi2, _ = gonzalez_ih_moose(self.Fmean, self.Cp2, self.Cl, self.T, self.depth, 0, 0, self.lpro/2)
            self.costa_xi = combine_arrays(costa_xi1, costa_xi2)
            self.costa_yi = combine_arrays(costa_yi1, costa_yi2)
        
            costa_xf1, costa_yf1, _ = gonzalez_ih_moose(self.Fmean, self.Cp1, self.Cl, self.T, self.depth, 0, 0, -self.lpro/2)
            costa_xf2, costa_yf2, _ = gonzalez_ih_moose(self.Fmean, self.Cp2, self.Cl, self.T, self.depth, 0, 0, -self.lpro/2)
            self.costa_xf = combine_arrays(costa_xf1, costa_xf2)
            self.costa_yf = combine_arrays(costa_yf1, costa_yf2)

        original_points_i = np.vstack((self.costa_xi, self.costa_yi))
        original_points_f = np.vstack((self.costa_xf, self.costa_yf))    
        self.xi, self.yi = original_points_i
        self.xf, self.yf = original_points_f
        self.prof = np.zeros((len(self.xi), 5))
        
        self.prof[:, 0] = 90 - np.arctan((self.yi - self.yf) / (self.xi - self.xf)) * 180 / np.pi
        self.prof[:, 1] = self.xi
        self.prof[:, 2] = self.yi
        self.prof[:, 3] = self.xf
        self.prof[:, 4] = self.yf
        
class plotting_planform(object):
    
    def __init__(self, **kwargs):
        """
        Jaramillo et al. 2021 model
        """
        
        self.Fmean = kwargs['Fmean']
        self.T = kwargs['T']
        self.depth = kwargs['hd']
        self.parabola_num = kwargs['parabola_num']
        self.Cl = kwargs['Cl']
        self.Scale = kwargs['Scale']
        
        if self.parabola_num == 1:
            self.Cp = kwargs['Cp']
            self.Lr = kwargs['Lr']
            self.costa_xe, self.costa_ye, self.alpha_curve = parabolic_planform(self.Fmean, self.Cp, self.Cl, self.T, self.depth, self.Lr, 0, 0, self.Scale)
            
        if self.parabola_num == 2:
            self.Cp1 = kwargs['Cp1']
            self.Cp2 = kwargs['Cp2']
            costa_xe1, costa_ye1, self.alpha_curve1 = parabolic_planform(self.Fmean, self.Cp1, self.Cl, self.T, self.depth, 0, 0, 0, self.Scale)
            costa_xe2, costa_ye2, self.alpha_curve2 = parabolic_planform(self.Fmean, self.Cp2, self.Cl, self.T, self.depth, 0, 0, 0, self.Scale)
            self.costa_xe = combine_arrays(costa_xe1, costa_xe2)
            self.costa_ye = combine_arrays(costa_ye1, costa_ye2)
            
        self.coast = np.vstack((self.costa_xe, self.costa_ye))

class cal_IH_MOOSE(object):
    """
    cal_IH_MOOSE
    
    Configuration to calibfalse, and run the EBSEM.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        
        self.path = path
        data = xr.open_dataset(path)
        cfg = json.loads(data.attrs['IH_MOOSE'])
        
        self.crossshore = cfg['crossshore']
        self.longshore = cfg['longshore']
        
        if self.crossshore == 'MD':
            cfg_c = json.loads(data.attrs['MillerDean'])
            self.trs = cfg_c['trs']
            self.cross = cal_MD.cal_MillerDean_2(self.path)
        if self.crossshore == 'YA':
            cfg_c = json.loads(data.attrs['Yates09'])
            self.trs = cfg_c['trs']
            self.cross = cal_YA.cal_Yates09_2(self.path)
        if self.crossshore == 'SF':
            cfg_c = json.loads(data.attrs['ShoreFor'])
            self.trs = cfg_c['trs']
            self.cross = cal_SF.cal_ShoreFor_2(self.path)
        if self.crossshore == 'JR':
            cfg_c = json.loads(data.attrs['Jara'])
            self.trs = cfg_c['trs']
            self.cross = cal_JR.cal_Jara_2(self.path)
        if self.crossshore == 'JA':
            cfg_c = json.loads(data.attrs['Jaramillo20'])
            self.trs = cfg_c['trs']
            self.cross = cal_JA20.cal_Jaramillo20_2(self.path)
        if self.crossshore == 'LIM':
            cfg_c = json.loads(data.attrs['Lim'])
            self.trs = cfg_c['trs']
            self.cross = cal_LIM.cal_Lim_2(self.path)
        
        if self.longshore == 'TU':
            self.long = cal_TU.cal_Turki_2(self.path)
        if self.longshore == 'JA':
            self.long = cal_JA21a.cal_Jaramillo21a_2(self.path)

        self.cross.calibrate()
        self.cross_run = self.cross.run_model(self.cross.solution)
        
        self.long.calibrate()
        self.long_run = self.long.run_model(self.long.solution)


class run_IH_MOOSE(object):
    """
    run_IH_MOOSE
    
    Run the EBSEM.
    """

    def __init__(self, path, params_cross, params_long):
        
        self.path = path
        data = xr.open_dataset(path)
        cfg = json.loads(data.attrs['IH_MOOSE'])
        
        self.crossshore = cfg['crossshore']
        self.longshore = cfg['longshore']
        
        if self.crossshore == 'MD':
            cfg_c = json.loads(data.attrs['run_MillerDean'])
            self.trs = cfg_c['trs']
            self.cross = MD_run.MillerDean_run(self.path)
        if self.crossshore == 'YA':
            cfg_c = json.loads(data.attrs['run_Yates09'])
            self.trs = cfg_c['trs']
            self.cross = YA_run.Yates09_run(self.path)
        if self.crossshore == 'SF':
            cfg_c = json.loads(data.attrs['run_ShoreFor'])
            self.trs = cfg_c['trs']
            self.cross = SF_run.ShoreFor_run(self.path)
        if self.crossshore == 'JR':
            cfg_c = json.loads(data.attrs['run_Jara'])
            self.trs = cfg_c['trs']
            self.cross = JR_run.Jara_run(self.path)
        if self.crossshore == 'JA':
            cfg_c = json.loads(data.attrs['run_Jaramillo20'])
            self.trs = cfg_c['trs']
            self.cross = JA20_run.Jaramillo20_run(self.path)
        if self.crossshore == 'LIM':
            cfg_c = json.loads(data.attrs['run_Lim'])
            self.trs = cfg_c['trs']
            self.cross = LIM_run.Lim_run(self.path)
        
        if self.longshore == 'TU':
            self.long = TU_run.Turki_run(self.path)
        if self.longshore == 'JA':
            self.long = JA21a_run.Jaramillo21a_run(self.path)

        self.cross_run = self.cross.run_model(params_cross)
        self.long_run = self.long.run_model(params_long)

def combine_arrays(x1, x2):
    x_bt = np.linspace(x1[-1], x2[-1])
    combined_length = len(x1) + len(x_bt) + len(x2)
    combined = np.empty(combined_length)
    combined[:len(x1)] = x1
    combined[len(x1):len(x1)+len(x_bt)] = x_bt
    combined[len(x1)+len(x_bt):] = np.flipud(x2)
        
    return combined

def parabolic_planform(Fmean, Cp, Cl, T, depth, Lr, gamd, dX): 
    Fmean = getAwayLims(Fmean)
    Fmean_o = Fmean
    Ld = hunt(T, depth)
    Xd, Yd = Cp # Difraction Point
    Xc, Yc = Cl # Coast Point
    Fmean = n2c(Fmean) + 90

    if Fmean < 0:
        Fmean = 360 + Fmean

    xc, yc = (Xd + 100 * np.cos(np.deg2rad(Fmean-90)), Yd + 100 * np.sin(np.deg2rad(Fmean-90)))
    m = (yc - Yd) / (xc - Xd)
    b = Yd - m * Xd
    b2 = Yc - m * Xc

    if b2 > b:
        flag_dir = 1
    else:
        flag_dir = -1
    
    Rl = ((Xd - Xc)**2 + (Yd - Yc)**2)**0.5
    the = np.arctan2(Yc - Yd, Xc - Xd)
    X = np.abs(Rl * np.sin(the - (Fmean)*np.pi/180)) + dX
    
    beta_r=2.13
    alpha_min=(np.arctan((((((beta_r)**4)/16)+(((beta_r)**2)/2)*(X/Ld))**(1/2))/(X/Ld))*180/np.pi)
    beta=90-alpha_min
    if beta <= 10:
        C0=0.0707-0.0047*10+0.000349*(10**2)-0.00000875*(10**3)+0.00000004765*(10**4)
        C1=0.9536+0.0078*10-0.0004879*(10**2)+0.0000182*(10**3)-0.0000001281*(10**4)
    else:
        C0=0.0707-0.0047*beta+0.000349*(beta**2)-0.00000875*(beta**3)+0.00000004765*(beta**4)
        C1=0.9536+0.0078*beta-0.0004879*(beta**2)+0.0000182*(beta**3)-0.0000001281*(beta**4)
    C2=1-C0-C1

    Ro=(X/Ld)/(np.sin(beta*np.pi/180))
    Ro=Ro*Ld
    
    theta0=np.arange(np.ceil(beta), 181, 1)
    theta=np.zeros(len(theta0)+1)
    theta[0]=beta
    theta[1:]=theta0

    R=Ro*(C0+C1*(beta/theta)+C2*((beta/theta)**2))
    dist = np.abs((-m *  Xc) + Yc - b) / np.sqrt(m**2 + 1)
    y_alpha = np.linspace(0, X, 250)

    x_alpha = (np.sqrt(((beta_r**4)/16)+(((beta_r)**2)/2)*(y_alpha/Ld)))*Ld
    rho, phi = np.sqrt(x_alpha**2 + y_alpha**2), np.arctan2(y_alpha, x_alpha)
    x_alpha, y_alpha = Xd + rho * np.cos(phi+(Fmean)*np.pi/180), Yd + rho * np.sin(phi+(Fmean)*np.pi/180)
    theta_rad = np.deg2rad(theta+Fmean)
    Lrr = np.linspace(0, Lr, 180 - len(theta))

    if Fmean_o > 0 and Fmean_o <= 180:
        x = Xd + R * np.cos(theta_rad)
        y = Yd + R * np.sin(theta_rad)
        costa_x = np.flipud(x)
        costa_y = np.flipud(y)
        if flag_dir == -1:
            for i in range(len(x_alpha)):
                x_alpha[i], y_alpha[i] = reflect_point(x_alpha[i], y_alpha[i], m, b)
            for i in range(len(costa_x)):
                costa_x[i], costa_y[i] = reflect_point(costa_x[i], costa_y[i], m, b)
            x2 = [costa_x, costa_x[-1] - Lrr * np.cos(Fmean * np.pi / 180)]
            y2 = [costa_y, costa_y[-1] - Lrr * np.sin(Fmean * np.pi / 180)]
        else:
            x2 = [costa_x, costa_x[-1] + Lrr * np.cos(Fmean * np.pi / 180)]
            y2 = [costa_y, costa_y[-1] + Lrr * np.sin(Fmean * np.pi / 180)]
    elif Fmean_o > 180 and Fmean_o <= 360:
        x = Xd + R * np.cos(theta_rad)
        y = Yd + R * np.sin(theta_rad)
        costa_x = np.flipud(x)
        costa_y = np.flipud(y)
        if flag_dir == 1:
            for i in range(len(x_alpha)):
                x_alpha[i], y_alpha[i] = reflect_point(x_alpha[i], y_alpha[i], m, b)
            for i in range(len(costa_x)):
                costa_x[i], costa_y[i] = reflect_point(costa_x[i], costa_y[i], m, b)
            x2 = [costa_x, costa_x[-1] - Lrr * np.cos(Fmean * np.pi / 180)]
            y2 = [costa_y, costa_y[-1] - Lrr * np.sin(Fmean * np.pi / 180)]
        else:
            x2 = [costa_x, costa_x[-1] + Lrr * np.cos(Fmean * np.pi / 180)]
            y2 = [costa_y, costa_y[-1] + Lrr * np.sin(Fmean * np.pi / 180)]
    
    costa_xx = [item for sublist in x2 for item in sublist]
    costa_yy = [item for sublist in y2 for item in sublist]

    alpha_curve = np.column_stack((x_alpha, y_alpha))
    
    return costa_xx, costa_yy, alpha_curve

def reflect_point(x, y, m, b):
    """
    Reflects a point (x, y) over a line y = mx + b.

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        m (float): Slope of the line.
        b (float): y-intercept of the line.

    Returns:
        tuple: (x_reflected, y_reflected) coordinates of the reflected point.
    """

    # Find the equation of the perpendicular line that passes through (x, y)
    perp_m = -1/m  # Negative reciprocal of the slope
    perp_b = y - perp_m * x

    # Find the intersection point of the perpendicular line and the original line
    x_intersect = (perp_b - b) / (m - perp_m)
    y_intersect = m * x_intersect + b

    # The reflected point is the mirror image of (x, y) across the intersection point

    x_reflected = 2 * x_intersect - x
    y_reflected = 2 * y_intersect - y

    return x_reflected, y_reflected

def getAwayLims(Fmean):
    if Fmean < 0.2:
        Fmean = 0.2
    elif Fmean > 359.8:
        Fmean = 359.8
    elif Fmean > 179.8 and Fmean <= 180:
        Fmean = 179.8
    elif Fmean > 180 and Fmean < 180.2:
        Fmean = 180.2
    elif Fmean > 89.8 and Fmean <= 90:
        Fmean = 89.8
    elif Fmean > 90 and Fmean < 90.2:
        Fmean = 90.2
    elif Fmean > 269.8 and Fmean <= 270:
        Fmean = 269.8
    elif Fmean > 270 and Fmean < 270.2:
        Fmean = 270.2
        
    return Fmean








