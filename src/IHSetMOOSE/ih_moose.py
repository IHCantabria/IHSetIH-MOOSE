import os
import xarray as xr
import numpy as np
from .ih_moose_jit import ih_moose_jit_par1, ih_moose_jit_par2, gonzalez_ih_moose, initialize_array, intersect_with_min_distance

class ih_moose(object):
    
    def __init__(self, path_prof, model, **kwargs):
        """
        Jaramillo et al. 2021 model
        """
        S = model.cross.full_run
        alp = model.long.full_run

        lim = kwargs['lim']
        Fmean = kwargs['Fmean']
        T = kwargs['T']
        depth = kwargs['depth']
        parabola_num = kwargs['parabola_num']
        trs_S = kwargs['trs_S']
        self.parabola_num = parabola_num
        
        ens = xr.open_dataset(os.path.join(path_prof))
        self.Obs = ens['obs'].values
        self.time_obs = ens['time_obs'].values
        self.ntrs_xi = ens['xi'].values
        self.ntrs_yi = ens['yi'].values
        self.ntrs_xf = ens['xf'].values
        self.ntrs_yf = ens['yf'].values
        ntrs = ens['ntrs'].values
        self.ntrs = ntrs[-1] + 1
        
        DirN = 90 - np.arctan((self.ntrs_yi[trs_S-1] - self.ntrs_yf[trs_S-1]) / (self.ntrs_xi[trs_S-1] - self.ntrs_xf[trs_S-1])) * 180 / np.pi
        self.Cl = [self.ntrs_xi[trs_S-1], self.ntrs_yi[trs_S-1]]
        
        Dif = np.abs(DirN - Fmean)
        dX = - (S * np.cos(np.radians(Dif)))
        
        delta_alpha = alp - np.mean(alp)
        delta_alpha = delta_alpha * np.pi / 180
    
        if parabola_num == 1:
            self.Cp = kwargs['Cp']
            self.Lr = kwargs['Lr']
            self.npro = kwargs['npro']
            
            self.xi, self.yi, self.xf, self.yf = ih_moose.transects(self.npro, lim[0], lim[1], self.Cl, Fmean, self.Cp, 0, 0, self.Lr, T, depth, parabola_num)
            prof = np.zeros((self.npro, 5))
            
            prof[:, 0] = 90 - np.arctan((self.yi - self.yf) / (self.xi - self.xf)) * 180 / np.pi
            prof[:, 1] = self.xi
            prof[:, 2] = self.yi
            prof[:, 3] = self.xf
            prof[:, 4] = self.yf
            
            xyi = np.vstack((self.xi, self.yi))
            pivot_point = np.array([ens['x_pivotal'], ens['y_pivotal']]).reshape(2, 1)
            pivotN = np.argmin(np.linalg.norm(xyi.T - pivot_point.T, axis=1))
            
            self.prof = prof
            self.S_PF, self.costas_x, self.costas_y = ih_moose_jit_par1(prof, pivotN, Fmean, self.Cp, self.Cl, T, depth, self.Lr, dX, delta_alpha, 0)
        
        if parabola_num == 2:
            self.Cp1 = kwargs['Cp1']
            self.Cp2 = kwargs['Cp2']
            self.npro = kwargs['npro']
            
            self.xi, self.yi, self.xf, self.yf = ih_moose.transects(self.npro, lim[0], lim[1], self.Cl, Fmean, 0, self.Cp1, self.Cp2, 0, T, depth, parabola_num)
            prof = np.zeros((self.npro, 5))
            
            prof[:, 0] = 90 - np.arctan((self.yi - self.yf) / (self.xi - self.xf)) * 180 / np.pi
            prof[:, 1] = self.xi
            prof[:, 2] = self.yi
            prof[:, 3] = self.xf
            prof[:, 4] = self.yf
            
            xyi = np.vstack((self.xi, self.yi))
            pivot_point = np.array([ens['x_pivotal'], ens['y_pivotal']]).reshape(2, 1)
            pivotN = np.argmin(np.linalg.norm(xyi.T - pivot_point.T, axis=1))
            
            self.prof = prof
            self.S_PF, self.costas_x, self.costas_y = ih_moose_jit_par2(prof, pivotN, Fmean, self.Cp1, self.Cp2, self.Cl, T, depth, 0, dX, delta_alpha, 0)
                
        self.SS = initialize_array((len(dX), self.ntrs), np.float64)
        
        for j in range(self.ntrs):
            m = (self.ntrs_yf[j] - self.ntrs_yi[j]) / (self.ntrs_xf[j] - self.ntrs_xi[j])
            b = self.ntrs_yi[j] - m * self.ntrs_xi[j]
            for i in range(len(dX)):
                xf, yf = intersect_with_min_distance(m, b, self.costas_x[i,:], self.costas_y[i,:])
                squared_distance = ((xf - self.ntrs_xi[j])**2 + (yf - self.ntrs_yi[j])**2)
                self.SS[i, j] = np.sqrt(squared_distance)
        
    def transects(nTrs, pt1, pt2, xyi, Fmean, Cp, Cp1, Cp2, Lr, T, depth, parabola_num):
        
        if parabola_num == 1:
            Cl = xyi
            costa_xi, costa_yi = gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, 0, 300)
            costa_xf, costa_yf = gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, 0, -300)
            
        if parabola_num == 2:
            Cl = xyi
            Lr = 0
            costa_xi1, costa_yi1 = gonzalez_ih_moose(Fmean, Cp1, Cl, T, depth, Lr, 0, 300)
            costa_xi2, costa_yi2 = gonzalez_ih_moose(Fmean, Cp2, Cl, T, depth, Lr, 0, 300)
            costa_xi = combine_arrays(costa_xi1, costa_xi2)
            costa_yi = combine_arrays(costa_yi1, costa_yi2)
        
            costa_xf1, costa_yf1 = gonzalez_ih_moose(Fmean, Cp1, Cl, T, depth, Lr, 0, -300)
            costa_xf2, costa_yf2 = gonzalez_ih_moose(Fmean, Cp2, Cl, T, depth, Lr, 0, -300)
            costa_xf = combine_arrays(costa_xf1, costa_xf2)
            costa_yf = combine_arrays(costa_yf1, costa_yf2)
    
        Fmeand = Fmean * np.pi/180
        rotation_matrix = np.array([[np.cos(Fmeand), -np.sin(Fmeand)],
                                [np.sin(Fmeand), np.cos(Fmeand)]])
        reverse_rotation_matrix = np.array([[np.cos(-Fmeand), -np.sin(-Fmeand)],
                                        [np.sin(-Fmeand), np.cos(-Fmeand)]])
    
        # Process initial points (i)
        original_points_i = np.vstack((costa_xi, costa_yi))
        closestPt1i = np.argmin(np.linalg.norm(original_points_i.T - pt1, axis=1))
        closestPt2i = np.argmin(np.linalg.norm(original_points_i.T - pt2, axis=1))
    
        rotated_points_i = rotation_matrix.dot(original_points_i[:,min(closestPt1i,closestPt2i):max(closestPt1i,closestPt2i)])
        rot_xi, rot_yi = rotated_points_i
        sorted_indices_i = np.argsort(rot_xi)
        xi = np.linspace(np.min(rot_xi), np.max(rot_xi), nTrs)
        yi = np.interp(xi, rot_xi[sorted_indices_i], rot_yi[sorted_indices_i])
        rotated_points_array_i = np.vstack((xi, yi))
        reverted_points_i = reverse_rotation_matrix.dot(rotated_points_array_i)
        xi, yi = reverted_points_i

        # Process final points (f)
        original_points_f = np.vstack((costa_xf, costa_yf))    
        closestPt1f = np.argmin(np.linalg.norm(original_points_f.T - pt1, axis=1))
        closestPt2f = np.argmin(np.linalg.norm(original_points_f.T - pt2, axis=1))
    
        rotated_points_f = rotation_matrix.dot(original_points_f[:,min(closestPt1f,closestPt2f):max(closestPt1f,closestPt2f)])
        rot_xf, rot_yf = rotated_points_f
        sorted_indices_f = np.argsort(rot_xf)
        xf = np.linspace(np.min(rot_xf), np.max(rot_xf), nTrs)
        yf = np.interp(xf, rot_xf[sorted_indices_f], rot_yf[sorted_indices_f])
        rotated_points_array_f = np.vstack((xf, yf))
        reverted_points_f = reverse_rotation_matrix.dot(rotated_points_array_f)
        xf, yf = reverted_points_f
    
        return xi, yi, xf, yf
        
def combine_arrays(x1, x2):
    x_bt = np.linspace(x1[-1], x2[-1])
    combined_length = len(x1) + len(x_bt) + len(x2)
    combined = np.empty(combined_length)
    combined[:len(x1)] = x1
    combined[len(x1):len(x1)+len(x_bt)] = x_bt
    combined[len(x1)+len(x_bt):] = np.flipud(x2) 
        
    return combined
