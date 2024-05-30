# from IHSetGonzalez import gonzalez
from IHSetUtils import waves
from IHSetUtils.geometry import nauticalDir2cartesianDirP as n2c
import numpy as np
from numba import jit

@jit
def ih_moose_jit(npro, prof, pivotN, pivotDir, pivotXY, Fmean, Cp, Cl, T, depth, Lr, dX, delta_alpha):
    S_PFo = np.zeros([len(dX), npro])
    pivotS = np.zeros([len(dX), 1])
    S_PF = np.zeros([len(dX), npro])
    X_PF = np.zeros([len(dX), 1])
    Y_PF = np.zeros([len(dX), 1])

    for i in range(len(dX)):
    # for i in range(int(np.round(len(dX)/100))):
        if i % 1000 == 0:
            print(i)
        
        costa = gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, dX[i])

        costa_x = costa[:,0]
        costa_y = costa[:,1]
                
            # line_coords1 = list(zip(costa_x, costa_y))
            # polygon1 = LineString(line_coords1)

        for j in range(npro):
            m = (prof[j,4] - prof[j,2]) / (prof[j,3] - prof[j,1])
            b = prof[j,2] - m * prof[j,1]
            xx = np.linspace(np.min(costa_x)*0.01,np.max(costa_x)*100)
            yy = m * xx + b
                # line_coords2 = list(zip(xx, yy))
                # polygon2 = LineString(line_coords2)
                # intersection = polygon1.intersection(polygon2)
            intersection = intersect_with_min_distance(costa_x, costa_y, xx, yy, 1e-5)
            xf = intersection[0]
            yf = intersection[1]
            S_PFo[i,j] = ((xf - prof[j,1])**2 + (yf - prof[j,2])**2)**0.5
        
            # del polygon1, polygon2, line_coords1, line_coords2, m, b, xx, yy, xf, yf

        pivotS[i] = S_PFo[i,pivotN]
        X_PF[i] = pivotXY[0] + pivotS[i] * np.cos(np.radians(pivotDir - 90)) # warning
        Y_PF[i] = pivotXY[1] - pivotS[i] * np.sin(np.radians(pivotDir - 90)) # warning
        centro = [X_PF[i], Y_PF[i]]
        
        xg = costa_x - centro[0]
        yg = costa_y - centro[1]

        theta0, rho = np.arctan2(yg, xg), np.sqrt(xg**2 + yg**2)
        theta = theta0 - delta_alpha[i]
        
        costa_xf, costa_yf = rho * np.cos(theta), rho * np.sin(theta)
        costa_xf += centro[0]
        costa_yf += centro[1]
        
        # line_coords1 = list(zip(costa_xf, costa_yf))
        # polygon1 = LineString(line_coords1)
        for j in range(npro):
            m = (prof[j,4] - prof[j,2]) / (prof[j,3] - prof[j,1])
            b = prof[j,2] - m * prof[j,1]
            xx = np.linspace(np.min(costa_x)*0.01,np.max(costa_x)*100)
            yy = m * xx + b
            # line_coords2 = list(zip(xx, yy))
            # polygon2 = LineString(line_coords2)
            # intersection = polygon1.intersection(polygon2)
            # xf = np.array(intersection.xy).T[0][0]
            # yf = np.array(intersection.xy).T[0][1] 
            intersection = intersect_with_min_distance(costa_x, costa_y, xx, yy, 1e-5)
            xf = intersection[0]
            yf = intersection[1]                               
            S_PF[i,j] = ((xf - prof[j,1])**2 + (yf - prof[j,2])**2)**0.5
            
    return S_PF

@jit(nopython=True)
def intersect_with_min_distance(x1, y1, x2, y2):
    arr1 = np.column_stack((x1, y1))
    arr2 = np.column_stack((x2, y2))
    
    intersection = []
    
    for i in range(arr1.shape[0]):
        min_distance = float('inf')
        min_j = -1
        for j in range(arr2.shape[0]):
            distance = np.sqrt((arr1[i, 0] - arr2[j, 0]) ** 2 + (arr1[i, 1] - arr2[j, 1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_j = j
    
        intersection.append(arr2[min_j,:])
    
    return np.array(intersection)

@jit
def gonzalez_ih_moose(Fmean, Cp, Cl, T, depth, Lr, dX): ## Lim
    
    Fmean = getAwayLims(Fmean)
    Fmean_o = Fmean
    Ld = waves.hunt(T, depth)
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
    
    X = np.abs(Rl * np.sin(the - (Fmean)*np.pi/180)) + dX ## Lim
    
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
    y_alpha = np.linspace(0, dist, 100)

    x_alpha = (np.sqrt(((beta_r**4)/16)+(((beta_r)**2)/2)*(y_alpha/Ld)))*Ld

    rho, phi = np.sqrt(x_alpha**2 + y_alpha**2), np.arctan2(y_alpha, x_alpha)

    x_alpha, y_alpha = Xd + rho * np.cos(phi+(Fmean)*np.pi/180), Yd + rho * np.sin(phi+(Fmean)*np.pi/180)

    theta_rad = np.deg2rad(theta+Fmean)

    if Fmean_o > 0 and Fmean_o <= 180:
        x = Xd + R * np.cos(theta_rad)
        y = Yd + R * np.sin(theta_rad)
        costa_x = np.flipud(x)
        costa_y = np.flipud(y)
        if flag_dir == -1:
            x2=np.vstack((costa_x[-1], costa_x[-1]-(Lr*np.cos((Fmean)*np.pi/180)))) ## Lim
            y2=np.vstack((costa_y[-1], costa_y[-1]-(Lr*np.sin((Fmean)*np.pi/180)))) ## Lim
            for i in range(len(x_alpha)):
                x_alpha[i], y_alpha[i] = reflect_point(x_alpha[i], y_alpha[i], m, b)

            for i in range(len(costa_x)):
                costa_x[i], costa_y[i] = reflect_point(costa_x[i], costa_y[i], m, b)
        else:
            x2=np.vstack((costa_x[-1], costa_x[-1]+(Lr*np.cos((Fmean)*np.pi/180)))) ## Lim
            y2=np.vstack((costa_y[-1], costa_y[-1]+(Lr*np.sin((Fmean)*np.pi/180)))) ## Lim
    elif Fmean_o > 180 and Fmean_o <= 360:
        x = Xd + R * np.cos(theta_rad)
        y = Yd + R * np.sin(theta_rad)
        costa_x = np.flipud(x)
        costa_y = np.flipud(y)
        if flag_dir == 1:
            x2=np.vstack((costa_x[-1], costa_x[-1]-(Lr*np.cos((Fmean)*np.pi/180)))) ## Lim
            y2=np.vstack((costa_y[-1], costa_y[-1]-(Lr*np.sin((Fmean)*np.pi/180)))) ## Lim
            for i in range(len(x_alpha)):
                x_alpha[i], y_alpha[i] = reflect_point(x_alpha[i], y_alpha[i], m, b)

            for i in range(len(costa_x)):
                costa_x[i], costa_y[i] = reflect_point(costa_x[i], costa_y[i], m, b)
        else:
            x2=np.vstack((costa_x[-1], costa_x[-1]+(Lr*np.cos((Fmean)*np.pi/180)))) ## Lim
            y2=np.vstack((costa_y[-1], costa_y[-1]+(Lr*np.sin((Fmean)*np.pi/180)))) ## Lim
    
    costa_x = np.append(costa_x, x2[:,0])
    costa_y = np.append(costa_y, y2[:,0])

    alpha_curve = np.column_stack((x_alpha, y_alpha))

    costa = np.column_stack((costa_x, costa_y))


    return costa

@jit
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
