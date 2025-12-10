import numpy as np
from scipy.optimize import curve_fit

def fcs_calibration_model(tau, N, D, w0, wz, T, tau_trip):
    term_xy = (1 + (4 * D * tau) / (w0**2))**(-1)
    term_z  = (1 + (4 * D * tau) / (wz**2))**(-0.5)
    term_trip = 1 + (T / (1 - T)) * np.exp(-tau / tau_trip)
    G = (1 / N) * term_xy * term_z * term_trip
    return G

def fit_calibration_data(lags, G_data, params, fix_flags, param_bounds):
    target_vars = ['N', 'D', 'w0', 'wz', 'T', 'tau_trip']
    p0 = []
    vary_keys = []
    
    # Boundsの準備 (Freeなパラメータ分だけ抽出)
    bounds_min = []
    bounds_max = []
    
    # 全パラメータのBounds定義 (辞書からリストへ)
    # param_bounds = {'N': (0, inf), ...}
    
    for key in target_vars:
        if not fix_flags.get(key, False):
            p0.append(params[key])
            vary_keys.append(key)
            
            # Boundsの追加
            if key in param_bounds:
                bounds_min.append(param_bounds[key][0])
                bounds_max.append(param_bounds[key][1])
            else:
                bounds_min.append(-np.inf)
                bounds_max.append(np.inf)
    
    # curve_fit用のBounds形式
    bounds = (bounds_min, bounds_max)

    def wrapper_func(t, *args):
        current = params.copy()
        for k, val in zip(vary_keys, args):
            current[k] = val
        return fcs_calibration_model(t, current['N'], current['D'], current['w0'], current['wz'], current['T'], current['tau_trip'])
    
    try:
        # bounds引数を追加
        popt, pcov = curve_fit(wrapper_func, lags, G_data, p0=p0, bounds=bounds, maxfev=5000)
        
        result_params = params.copy()
        for k, val in zip(vary_keys, popt):
            result_params[k] = val
        return result_params
    except RuntimeError:
        print("Fitting failed.")
        return params