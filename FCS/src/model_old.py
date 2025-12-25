import numpy as np
from scipy.optimize import curve_fit

def fcs_standard_model(tau, N, D, w0, wz, T, tau_trip, y0):
    """
    基本モデル + オフセット(y0)
    G(tau) = (1/N) * Diff_XY * Diff_Z * Triplet + y0
    """
    # 拡散項 (横)
    term_xy = (1 + (4 * D * tau) / (w0**2))**(-1)
    
    # 拡散項 (縦)
    term_z  = (1 + (4 * D * tau) / (wz**2))**(-0.5)
    
    # Triplet項
    term_trip = 1 + (T / (1 - T)) * np.exp(-tau / tau_trip)
    
    # 全体 + オフセット
    G = (1 / N) * term_xy * term_z * term_trip + y0
    
    return G

def fit_standard_data(lags, G_data, params, fix_flags, param_bounds):
    """
    y0 を含む基本フィッティング関数
    """
    target_vars = ['N', 'D', 'w0', 'wz', 'T', 'tau_trip', 'y0']
    
    p0 = []
    vary_keys = []
    bounds_min = []
    bounds_max = []
    
    for key in target_vars:
        if not fix_flags.get(key, False):
            p0.append(params[key])
            vary_keys.append(key)
            if key in param_bounds:
                bounds_min.append(param_bounds[key][0])
                bounds_max.append(param_bounds[key][1])
            else:
                bounds_min.append(-np.inf)
                bounds_max.append(np.inf)
    
    bounds = (bounds_min, bounds_max)

    def wrapper_func(t, *args):
        current = params.copy()
        for k, val in zip(vary_keys, args):
            current[k] = val
        
        return fcs_standard_model(t, 
                            current['N'], current['D'], 
                            current['w0'], current['wz'], 
                            current['T'], current['tau_trip'],
                            current['y0'])
    
    try:
        popt, pcov = curve_fit(wrapper_func, lags, G_data, p0=p0, bounds=bounds, maxfev=10000)
        
        result_params = params.copy()
        for k, val in zip(vary_keys, popt):
            result_params[k] = val
            
        return result_params
    except RuntimeError:
        print("Fitting failed.")
        return params