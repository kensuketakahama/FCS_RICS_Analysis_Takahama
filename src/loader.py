import numpy as np
import tifffile
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import iirnotch, filtfilt

def load_fcs_trace(filepath):
    try:
        data = tifffile.imread(filepath)
        trace = data.flatten().astype(np.float64)
        print(f"Loaded trace length: {len(trace)} points")
        return trace
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_noise_spectrum(trace, sampling_interval):
    """
    パワースペクトル密度(PSD)を計算して、支配的なノイズ周波数を探す
    """
    n = len(trace)
    sample_rate = 1.0 / sampling_interval
    
    # FFT計算
    yf = rfft(trace - np.mean(trace))
    xf = rfftfreq(n, sampling_interval)
    
    power = np.abs(yf)**2
    
    # ピーク周波数の検出 (DC成分除く)
    # 最もパワーが強い周波数を返す
    peak_idx = np.argmax(power[1:]) + 1
    peak_freq = xf[peak_idx]
    
    return xf, power, peak_freq

def remove_periodic_noise(trace, sampling_interval, target_freq, Q=30.0):
    """
    ノッチフィルタを用いて特定の周波数(target_freq)を除去する
    Q: フィルタの鋭さ (高いほどピンポイントで消す)
    """
    fs = 1.0 / sampling_interval
    
    # ナイキスト周波数チェック
    if target_freq >= fs/2:
        print(f"Warning: Target frequency {target_freq}Hz is above Nyquist. Skipping filter.")
        return trace

    print(f"Applying Notch Filter at {target_freq:.1f} Hz...")
    
    # フィルタ設計
    b, a = iirnotch(target_freq, Q, fs)
    
    # フィルタ適用 (位相ズレを防ぐためfiltfiltを使用)
    filtered_trace = filtfilt(b, a, trace)
    
    return filtered_trace

def process_trace(trace, window_size_ratio=0.01):
    from scipy.ndimage import uniform_filter1d
    n = len(trace)
    w_size = int(n * window_size_ratio)
    if w_size < 100: w_size = 100
    smoothed = uniform_filter1d(trace, size=w_size)
    global_mean = np.mean(trace)
    return trace - smoothed + global_mean