import numpy as np
from numba import jit

@jit
def ih_moose_jit_par1(prof, pivotN, Fmean, Cp, Cl, T, depth, Lr, dX, delta_alpha, gamd):
    npro = prof.shape[0]
    pivotDir = prof[pivotN,0]
    pivotXY = [prof[pivotN,1], prof[pivotN,2]]
    
    S_PFo = initialize_array((len(dX), npro), np.float64)
    S_PF = initialize_array((len(dX), npro), np.float64)
    pivotS = initialize_array((len(dX), 1), np.float64)
    X_PF = initialize_array((len(dX), 1), np.float64)
    Y_PF = initialize_array((len(dX), 1), np.float64)
    
    for i in range(len(dX)):
        
        costa_x, costa_y = gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, gamd, dX[i])
        m = (prof[pivotN,4] - prof[pivotN,2]) / (prof[pivotN,3] - prof[pivotN,1])
        b = prof[pivotN,2] - m * prof[pivotN,1]        
        xf, yf = intersect_with_min_distance(m, b, costa_x, costa_y)
        squared_distance = ((xf - prof[pivotN, 1])**2 + (yf - prof[pivotN, 2])**2)
        S_PFo = np.sqrt(squared_distance)
        
        pivotS[i] = S_PFo
        X_PF[i] = pivotXY[0] + pivotS[i] * np.cos(np.radians(pivotDir - 90))
        Y_PF[i] = pivotXY[1] - pivotS[i] * np.sin(np.radians(pivotDir - 90))
        centro = [X_PF[i], Y_PF[i]]
                
        xg = np.array(costa_x) - centro[0]
        yg = np.array(costa_y) - centro[1]

        theta0, rho = np.arctan2(yg, xg), np.sqrt(xg**2 + yg**2)
        theta = theta0 - delta_alpha[i]
        
        costa_xf, costa_yf = rho * np.cos(theta), rho * np.sin(theta)
        costa_xf += centro[0]
        costa_yf += centro[1]
        
        for j in range(npro):
            m = (prof[j,4] - prof[j,2]) / (prof[j,3] - prof[j,1])
            b = prof[j,2] - m * prof[j,1]
            xf, yf = intersect_with_min_distance(m, b, costa_xf, costa_yf)
            squared_distance = ((xf - prof[j, 1])**2 + (yf - prof[j, 2])**2)
            S_PF[i, j] = np.sqrt(squared_distance)
            
    return S_PF

@jit
def ih_moose_jit_par2(prof, pivotN, Fmean, Cp1, Cp2, Cl, T, depth, Lr, dX, delta_alpha, gamd):
    npro = prof.shape[0]
    pivotDir = prof[pivotN,0]
    pivotXY = [prof[pivotN,1], prof[pivotN,2]]
    
    S_PFo = initialize_array((len(dX), npro), np.float64)
    S_PF = initialize_array((len(dX), npro), np.float64)
    pivotS = initialize_array((len(dX), 1), np.float64)
    X_PF = initialize_array((len(dX), 1), np.float64)
    Y_PF = initialize_array((len(dX), 1), np.float64)
    
    for i in range(len(dX)):
        costa_x1, costa_y1 = gonzalez_ih_moose(Fmean, Cp1, Cl, T, depth, Lr, gamd, dX[i])
        costa_x2, costa_y2 = gonzalez_ih_moose(Fmean, Cp2, Cl, T, depth, Lr, gamd, dX[i])
        costa_x = combine_arrays(costa_x1, costa_x2)
        costa_y = combine_arrays(costa_y1, costa_y2)
        
        m = (prof[pivotN,4] - prof[pivotN,2]) / (prof[pivotN,3] - prof[pivotN,1])
        b = prof[pivotN,2] - m * prof[pivotN,1]        
        xf, yf = intersect_with_min_distance(m, b, costa_x, costa_y)
        squared_distance = ((xf - prof[pivotN, 1])**2 + (yf - prof[pivotN, 2])**2)
        S_PFo = np.sqrt(squared_distance)
        
        pivotS[i] = S_PFo
        X_PF[i] = pivotXY[0] + pivotS[i] * np.cos(np.radians(pivotDir - 90))
        Y_PF[i] = pivotXY[1] - pivotS[i] * np.sin(np.radians(pivotDir - 90))
        centro = [X_PF[i], Y_PF[i]]
            
        xg = costa_x - centro[0]
        yg = costa_y - centro[1]

        theta0, rho = np.arctan2(yg, xg), np.sqrt(xg**2 + yg**2)
        theta = theta0 - delta_alpha[i]
        
        costa_xf, costa_yf = rho * np.cos(theta), rho * np.sin(theta)
        costa_xf += centro[0]
        costa_yf += centro[1]
        
        for j in range(npro):
            m = (prof[j,4] - prof[j,2]) / (prof[j,3] - prof[j,1])
            b = prof[j,2] - m * prof[j,1]
            xf, yf = intersect_with_min_distance(m, b, costa_xf, costa_yf)
            squared_distance = ((xf - prof[j, 1])**2 + (yf - prof[j, 2])**2)
            S_PF[i, j] = np.sqrt(squared_distance)
            
    return S_PF

@jit
def initialize_array(shape, dtype):
    return np.zeros(shape, dtype)

@jit
def intersect_with_min_distance(m, b, x1, y1):
    min_distance = np.inf
    for i in range(len(x1)):
        distance = abs(m * x1[i] - y1[i] + b) / (m**2 + 1)**0.5
        if distance < min_distance:
            min_distance = distance
            intersection_x = x1[i]
            intersection_y = y1[i]
    
    return intersection_x, intersection_y

@jit
def gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, gamd, dX): 
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
    # Elshinnawy et al. (2022) - Static Equilibrium
    if beta <= 10:
        C0=0.0707-0.0047*10+0.000349*(10**2)-0.00000875*(10**3)+0.00000004765*(10**4)
        C1=0.9536+0.0078*10-0.0004879*(10**2)+0.0000182*(10**3)-0.0000001281*(10**4)
    else:
        C0=0.0707-0.0047*beta+0.000349*(beta**2)-0.00000875*(beta**3)+0.00000004765*(beta**4)
        C1=0.9536+0.0078*beta-0.0004879*(beta**2)+0.0000182*(beta**3)-0.0000001281*(beta**4)
    
    # Elshinnawy et al. (2022) - Dynamic Equilibrium
    bt = beta*np.pi/180
    if flag_dir == 1:
        gamd = -gamd
    if gamd != 0.0:
        alp_st = 0.277 - 0.0785 * 10**(bt)
        psi = (bt * np.cos(bt) + bt * np.sin(bt) * np.tan(gamd*np.pi/180)) / (np.sin(bt) - np.cos(bt) * np.tan(gamd*np.pi/180))
        C0 = 1 - psi + alp_st
        C1 = psi - 2 * alp_st
    C2=1-C0-C1

    thed = np.arctan2(Yd - Yc, Xd - Xc) * 180 / np.pi
    thed = 90 - thed
    if thed < 0:
        thed = 360 + thed
    if flag_dir == 1:
        if Fmean_o >= 270 and thed <= 90:
            thed = 360 + thed
        if Fmean_o <= 90 and thed >= 270:
            thed = thed - 360
        bt_ref = 90 - abs(thed - Fmean_o)
    if flag_dir == -1:
        if Fmean_o >= 270 and thed <= 90:
            thed = 360 + thed
        if Fmean_o <= 90 and thed >= 270:
            thed = thed - 360
        bt_ref = 90 - abs(Fmean_o - thed)
    if bt_ref >= beta:
        beta = bt_ref

    # Elshinnawy et al. (2022) - Pocket Beach
    if bt_ref > 70:
        THE = 19 / ((1+np.exp(-0.3*(bt_ref-45)))**8270)
        alp_gap = 0.277 - 0.0785 * 10**((bt_ref-THE)*np.pi/180)
        C0 = 1 - bt_ref*np.pi/180 / np.tan(bt_ref*np.pi/180) + alp_gap
        C1 = bt_ref*np.pi/180 / np.tan(bt_ref*np.pi/180)-2 * alp_gap
        C2 = alp_gap

    Ro=(X/Ld)/(np.sin(beta*np.pi/180))
    Ro=Ro*Ld
    
    theta0=np.arange(np.ceil(beta), 181, 1)
    theta=np.zeros(len(theta0)+1)
    theta[0]=beta
    theta[1:]=theta0

    R=Ro*(C0+C1*(beta/theta)+C2*((beta/theta)**2))
    dist = np.abs((-m *  Xc) + Yc - b) / np.sqrt(m**2 + 1)
    y_alpha = np.linspace(0, dist, 250)

    x_alpha = (np.sqrt(((beta_r**4)/16)+(((beta_r)**2)/2)*(y_alpha/Ld)))*Ld
    rho, phi = np.sqrt(x_alpha**2 + y_alpha**2), np.arctan2(y_alpha, x_alpha)
    x_alpha, y_alpha = Xd + rho * np.cos(phi+(Fmean)*np.pi/180), Yd + rho * np.sin(phi+(Fmean)*np.pi/180)
    theta_rad = np.deg2rad(theta+Fmean)
    Lrr = np.linspace(0, Lr)

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

    return costa_xx, costa_yy

@jit
def reflect_point(x, y, m, b):
    perp_m = -1/m
    perp_b = y - perp_m * x

    x_intersect = (perp_b - b) / (m - perp_m)
    y_intersect = m * x_intersect + b

    x_reflected = 2 * x_intersect - x
    y_reflected = 2 * y_intersect - y

    return x_reflected, y_reflected

@jit
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

@jit  
def n2c(nDir):
    cDir = 90.0 - nDir
    if cDir < -180.0:
        cDir += 360.0
    
    return cDir

@jit    
def hunt(T, d):   
    g = 9.81
    G = (2 * np.pi / T) ** 2 * (d / g)
    F = G + 1.0 / (1 + 0.6522*G + 0.4622*G**2 + 0.0864*G**4 + 0.0675*G**5)
    L = T * (g * d / F) ** 0.5
    
    return L

@jit
def combine_arrays(x1, x2):
    x_bt = np.linspace(x1[-1], x2[-1])
    combined_length = len(x1) + len(x_bt) + len(x2)
    combined = np.empty(combined_length)
    combined[:len(x1)] = x1
    combined[len(x1):len(x1)+len(x_bt)] = x_bt
    combined[len(x1)+len(x_bt):] = np.flipud(x2)
    
    return combined
