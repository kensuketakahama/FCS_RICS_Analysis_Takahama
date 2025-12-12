import matplotlib.pyplot as plt
import numpy as np
import sys

# モジュール読み込み
try:
    from src.loader import load_fcs_trace, analyze_noise_spectrum, remove_periodic_noise, process_trace_detrend
    from src.correlation import calculate_segmented_acf, log_binning_weighted
    print("[1] Modules loaded successfully.")
except ImportError as e:
    print(f"Error loading modules: {e}")
    sys.exit(1)

def main():
    FILE_PATH = 'Data/FCS1_TAMRA1000nM.tif' 
    SAMPLING_INTERVAL = 2.0e-6 
    
    USE_DETREND = True       
    DETREND_CUTOFF = 0.005   

    # ==========================================
    # 1. データ読み込み
    # ==========================================
    print(f"[2] Loading {FILE_PATH}...")
    raw_trace = load_fcs_trace(FILE_PATH)
    
    if raw_trace is None:
        print("Error: Trace is None. Check file path.")
        return

    # 平均強度の表示
    mean_cps = np.mean(raw_trace) / SAMPLING_INTERVAL
    print(f"Mean Count Rate: {mean_cps/1000:.1f} kHz")

    # ==========================================
    # 2. 前処理
    # ==========================================
    print("[3] Pre-processing...")
    freqs, power, peak_freq = analyze_noise_spectrum(raw_trace, SAMPLING_INTERVAL)
    if peak_freq > 10: 
        print(f" - Removing periodic noise at {peak_freq:.1f} Hz")
        trace_filtered = remove_periodic_noise(raw_trace, SAMPLING_INTERVAL, peak_freq)
    else:
        trace_filtered = raw_trace

    if USE_DETREND:
        print(f" - Applying Detrend filter (Cutoff: {DETREND_CUTOFF*1000:.1f} ms)")
        trace = process_trace_detrend(trace_filtered, sampling_interval=SAMPLING_INTERVAL, cutoff_time=DETREND_CUTOFF)
    else:
        print(" - Using RAW TRACE (No Detrending)")
        trace = trace_filtered

    if trace is None:
        print("Error: Trace became None after processing. Check src/loader.py return statement.")
        return

    # ==========================================
    # 3. 相関計算
    # ==========================================
    print("[4] Calculating ACF...")
    try:
        raw_lags, raw_G, sem_G = calculate_segmented_acf(trace, n_segments=10)
        time_lags = raw_lags * SAMPLING_INTERVAL
        
        lags_bin, G_bin, sigma_bin = log_binning_weighted(time_lags, raw_G, sem_G)
        print(" - ACF Calculation finished.")
    except Exception as e:
        print(f"Error during ACF calculation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # 4. グラフ保存 (Save instead of Show)
    # ==========================================
    print(f"[5] Plotting data... (Max G: {np.max(G_bin):.6f})")

    plt.figure(figsize=(8, 6))
    
    plt.errorbar(lags_bin, G_bin, yerr=sigma_bin, fmt='bo', alpha=0.8, label='Raw Data (ACF)', capsize=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.xscale('log')
    plt.xlabel('Lag Time [s]')
    plt.ylabel('Autocorrelation G(tau)')
    
    title_str = "Raw Data Check"
    if USE_DETREND:
        title_str += f" (Detrended: {DETREND_CUTOFF*1000:.1f}ms)"
    else:
        title_str += " (No Detrend)"
        
    plt.title(title_str)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # ファイルとして保存
    save_filename = 'raw_data_check.png'
    print(f"[6] Saving figure to {save_filename} ...")
    plt.savefig(save_filename)
    print("Done.")

if __name__ == "__main__":
    main()