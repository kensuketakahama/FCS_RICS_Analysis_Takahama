import numpy as np
from scipy.signal import correlate

def calculate_1d_acf(trace):
    """
    1次元時系列データの自己相関関数 G(tau) を計算する
    G(tau) = <I(t)I(t+tau)> / <I>^2 - 1
    """
    n = len(trace)
    mean_i = np.mean(trace)
    
    # FFTを使った高速自己相関 (scipy.signal.correlateを使用)
    # mode='full'で計算し、後半半分（正のラグ）を取り出す
    corr_full = correlate(trace - mean_i, trace - mean_i, mode='full', method='fft')
    corr_positive = corr_full[n-1:]
    
    # 正規化: データ点数で割って共分散にし、平均の二乗で割る
    # ただし、ラグが大きくなると重なり(n - tau)が減るための補正が必要
    lags = np.arange(len(corr_positive))
    normalization = (n - lags) * (mean_i ** 2)
    
    # ゼロ除算回避
    with np.errstate(divide='ignore', invalid='ignore'):
        G = corr_positive / normalization
    
    G[normalization == 0] = 0
    return lags, G

def log_binning(lags, G, points_per_decade=10):
    """
    FCSは対数グラフで見るため、データを対数等間隔に間引く（Log Binning）
    """
    # ラグ0は除外（ショットノイズの影響が大きいため）
    valid_indices = lags > 0
    lags = lags[valid_indices]
    G = G[valid_indices]
    
    if len(lags) == 0:
        return np.array([]), np.array([])

    min_lag = np.log10(lags[0])
    max_lag = np.log10(lags[-1])
    
    # 対数空間でのビンを作成
    n_bins = int((max_lag - min_lag) * points_per_decade)
    if n_bins < 10: n_bins = 10
    
    log_bins = np.logspace(min_lag, max_lag, n_bins)
    
    # digitizeでビン分けして平均を取る
    indices = np.digitize(lags, log_bins)
    
    new_lags = []
    new_G = []
    
    for i in range(1, len(log_bins)):
        mask = (indices == i)
        if np.any(mask):
            new_lags.append(np.mean(lags[mask]))
            new_G.append(np.mean(G[mask]))
            
    return np.array(new_lags), np.array(new_G)