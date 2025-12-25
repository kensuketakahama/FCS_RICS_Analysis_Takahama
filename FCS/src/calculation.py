import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def _correlate_1d(a, b):
    """高速相関計算 (FFT使用)"""
    n = len(a)
    n_fft = 2 ** int(np.ceil(np.log2(2*n - 1)))
    fa = np.fft.fft(a, n_fft)
    fb = np.fft.fft(b, n_fft)
    acf = np.fft.ifft(fa * np.conj(fb))
    acf = np.real(acf)[:n]
    return acf

def calculate_segmented_acf(trace, n_segments=10):
    """
    データを分割してACFを計算し、平均と標準誤差(SEM)を返す
    """
    trace = np.array(trace)
    total_len = len(trace)
    seg_len = total_len // n_segments
    
    if seg_len < 100: 
        n_segments = 1
        seg_len = total_len
        
    G_segments = []
    
    for i in range(n_segments):
        segment = trace[i*seg_len : (i+1)*seg_len]
        mean_i = np.mean(segment)
        if mean_i == 0: continue
        
        fluctuation = segment - mean_i
        corr = _correlate_1d(fluctuation, fluctuation)
        norm_factor = np.arange(len(corr), 0, -1)
        
        # 正規化
        g_seg = (corr / norm_factor) / (mean_i**2)
        G_segments.append(g_seg)
        
    min_len = min([len(g) for g in G_segments])
    G_matrix = np.array([g[:min_len] for g in G_segments])
    
    mean_G = np.mean(G_matrix, axis=0)
    # 標準誤差
    if n_segments > 1:
        sem_G = np.std(G_matrix, axis=0, ddof=1) / np.sqrt(n_segments)
    else:
        sem_G = np.ones_like(mean_G) * 1e-5
        
    lags = np.arange(min_len)
    return lags, mean_G, sem_G

def log_binning_weighted(lags, G, sigma, points_per_decade=10):
    """
    対数等間隔ビニング (重み付き平均)
    """
    mask = lags > 0
    lags = lags[mask]
    G = G[mask]
    sigma = sigma[mask]
    
    if len(lags) == 0: return np.array([]), np.array([]), np.array([])
    
    min_lag = np.min(lags)
    max_lag = np.max(lags)
    
    n_decades = np.log10(max_lag / min_lag)
    n_bins = int(n_decades * points_per_decade)
    if n_bins < 5: n_bins = 10
    
    bins = np.logspace(np.log10(min_lag), np.log10(max_lag), n_bins+1)
    
    bin_lags = []
    bin_G = []
    bin_sigma = []
    
    indices = np.digitize(lags, bins)
    
    for i in range(1, len(bins)):
        in_bin = (indices == i)
        if not np.any(in_bin): continue
        
        g_vals = G[in_bin]
        l_vals = lags[in_bin]
        s_vals = sigma[in_bin]
        
        weights = 1.0 / (s_vals**2 + 1e-12)
        w_sum = np.sum(weights)
        if w_sum == 0: continue
        
        avg_l = np.sum(l_vals * weights) / w_sum
        avg_g = np.sum(g_vals * weights) / w_sum
        avg_s = np.sqrt(1.0 / w_sum)
        
        bin_lags.append(avg_l)
        bin_G.append(avg_g)
        bin_sigma.append(avg_s)
        
    return np.array(bin_lags), np.array(bin_G), np.array(bin_sigma)