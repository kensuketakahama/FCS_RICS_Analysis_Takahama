import numpy as np
import tifffile
from scipy.fft import rfft, rfftfreq
from scipy.signal import iirnotch, filtfilt
from scipy.ndimage import uniform_filter1d

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
    n = len(trace)
    yf = rfft(trace - np.mean(trace))
    xf = rfftfreq(n, sampling_interval)
    power = np.abs(yf)**2
    peak_idx = np.argmax(power[1:]) + 1
    peak_freq = xf[peak_idx]
    return xf, power, peak_freq

def remove_periodic_noise(trace, sampling_interval, target_freq, Q=30.0):
    fs = 1.0 / sampling_interval
    if target_freq >= fs/2:
        print(f"Warning: Target frequency {target_freq}Hz is above Nyquist. Skipping filter.")
        return trace

    print(f"Applying Notch Filter at {target_freq:.1f} Hz...")
    b, a = iirnotch(target_freq, Q, fs)
    filtered_trace = filtfilt(b, a, trace)
    return filtered_trace

def process_trace_detrend(trace, sampling_interval, cutoff_time=0.01):
    """
    【ここを確認！】退色補正関数
    """
    # 窓サイズ(ポイント数)の計算
    window_size = int(cutoff_time / sampling_interval)
    if window_size < 10: window_size = 10
    
    print(f"Detrending: Removing fluctuations slower than {cutoff_time*1000:.1f} ms (Window: {window_size} pts)")
    
    # 移動平均
    trend = uniform_filter1d(trace, size=window_size)
    
    # 補正
    global_mean = np.mean(trace)
    corrected_trace = trace - trend + global_mean
    
    # ★★★ ここが重要！ ★★★
    # この return が無いと、main.py の trace が None になりエラーが出ます
    return corrected_trace