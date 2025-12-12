import numpy as np
from scipy.signal import fftconvolve

def calculate_1d_acf(trace):
    """
    基本的な1次元自己相関計算 (FFT使用)
    G(tau) = <I(t)I(t+tau)> / <I>^2 - 1
    """
    n = len(trace)
    mean_i = np.mean(trace)
    
    # FFT畳み込み
    corr_full = fftconvolve(trace, trace[::-1], mode='full')
    corr_positive = corr_full[n-1:]
    
    # 正規化
    lags = np.arange(len(corr_positive))
    weights = n - lags
    valid = weights > 0
    
    G = np.zeros_like(corr_positive)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        G[valid] = (corr_positive[valid] / weights[valid]) / (mean_i**2) - 1
        
    return lags, G

def calculate_segmented_acf(trace, n_segments=10):
    """
    【分割平均法】
    データを n_segments 個に分割し、それぞれのACFを計算して平均を取る。
    標準誤差(SEM)も同時に計算する。
    """
    total_len = len(trace)
    seg_len = total_len // n_segments
    
    accumulated_G = None
    accumulated_G_sq = None # 二乗和（分散計算用）
    valid_lags = None
    
    # print(f"Splitting trace into {n_segments} segments of {seg_len} points.")
    
    for i in range(n_segments):
        # セグメント切り出し
        segment = trace[i*seg_len : (i+1)*seg_len]
        
        # 基本ACF計算
        lags, G = calculate_1d_acf(segment)
        
        # 初回初期化
        if accumulated_G is None:
            accumulated_G = np.zeros_like(G)
            accumulated_G_sq = np.zeros_like(G)
            valid_lags = lags
            
        # 長さが合う部分だけ加算
        min_len = min(len(G), len(accumulated_G))
        accumulated_G[:min_len] += G[:min_len]
        accumulated_G_sq[:min_len] += G[:min_len]**2
        
    # 平均と標準誤差(SEM)の計算
    mean_G = accumulated_G / n_segments
    
    # 分散 V = E[X^2] - (E[X])^2
    mean_sq = accumulated_G_sq / n_segments
    variance = mean_sq - mean_G**2
    variance[variance < 0] = 0 # 計算誤差対策
    
    std_G = np.sqrt(variance)
    sem_G = std_G / np.sqrt(n_segments) # 標準誤差
    
    return valid_lags, mean_G, sem_G

def log_binning(lags, G, points_per_decade=15):
    """
    通常の対数ビニング
    """
    valid_indices = lags > 0
    lags = lags[valid_indices]
    G = G[valid_indices]
    
    if len(lags) == 0: return np.array([]), np.array([])

    min_lag = np.log10(lags[0])
    max_lag = np.log10(lags[-1])
    n_bins = int((max_lag - min_lag) * points_per_decade)
    if n_bins < 10: n_bins = 10
    
    log_bins = np.logspace(min_lag, max_lag, n_bins)
    indices = np.digitize(lags, log_bins)
    
    new_lags = []
    new_G = []
    
    for i in range(1, len(log_bins)):
        mask = (indices == i)
        if np.any(mask):
            new_lags.append(np.mean(lags[mask]))
            new_G.append(np.mean(G[mask]))
            
    return np.array(new_lags), np.array(new_G)

def log_binning_weighted(lags, G, sigma, points_per_decade=15):
    """
    重み(sigma)付きの対数ビニング
    """
    valid_indices = lags > 0
    lags = lags[valid_indices]
    G = G[valid_indices]
    sigma = sigma[valid_indices]
    
    if len(lags) == 0: return np.array([]), np.array([]), np.array([])

    min_lag = np.log10(lags[0])
    max_lag = np.log10(lags[-1])
    n_bins = int((max_lag - min_lag) * points_per_decade)
    if n_bins < 10: n_bins = 10
    
    log_bins = np.logspace(min_lag, max_lag, n_bins)
    indices = np.digitize(lags, log_bins)
    
    new_lags = []
    new_G = []
    new_sigma = []
    
    for i in range(1, len(log_bins)):
        mask = (indices == i)
        if np.any(mask):
            new_lags.append(np.mean(lags[mask]))
            new_G.append(np.mean(G[mask]))
            # 誤差伝播: sqrt(sum(sigma^2)) / N
            sigma_subset = sigma[mask]
            new_sigma.append(np.sqrt(np.sum(sigma_subset**2)) / len(sigma_subset))
            
    return np.array(new_lags), np.array(new_G), np.array(new_sigma)