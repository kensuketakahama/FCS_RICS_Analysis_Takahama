import matplotlib.pyplot as plt
import numpy as np

# 既存のモジュールを使用
from src.loader import load_fcs_trace, analyze_noise_spectrum, remove_periodic_noise, process_trace_detrend
from src.correlation import calculate_segmented_acf, log_binning_weighted

def main():
    FILE_PATH = 'Data/FCS1_TAMRA1000nM.tif' 
    SAMPLING_INTERVAL = 2.0e-6 
    
    # ==========================================
    # 設定エリア
    # ==========================================
    # ここを変えるだけで、データの前処理が変わります
    USE_DETREND = True       # True: 退色補正あり / False: 完全生データ
    DETREND_CUTOFF = 0.005   # 補正する場合のカットオフ時間 [秒]

    # ==========================================
    # 1. データ読み込み
    # ==========================================
    print(f"Loading {FILE_PATH}...")
    raw_trace = load_fcs_trace(FILE_PATH)
    if raw_trace is None: return

    # 平均強度の表示 (kHz)
    mean_cps = np.mean(raw_trace) / SAMPLING_INTERVAL
    print(f"Mean Count Rate: {mean_cps/1000:.1f} kHz")

    # ==========================================
    # 2. ノイズ除去・退色補正 (前処理)
    # ==========================================
    # ノイズ除去 (定常波)
    freqs, power, peak_freq = analyze_noise_spectrum(raw_trace, SAMPLING_INTERVAL)
    if peak_freq > 10: 
        print(f"Removing periodic noise at {peak_freq:.1f} Hz")
        trace_filtered = remove_periodic_noise(raw_trace, SAMPLING_INTERVAL, peak_freq)
    else:
        trace_filtered = raw_trace

    # 退色補正の分岐
    if USE_DETREND:
        print(f"Applying Detrend filter (Cutoff: {DETREND_CUTOFF*1000:.1f} ms)")
        trace = process_trace_detrend(trace_filtered, sampling_interval=SAMPLING_INTERVAL, cutoff_time=DETREND_CUTOFF)
    else:
        print("!!! Using RAW TRACE (No Detrending) !!!")
        trace = trace_filtered

    if trace is None: return

    # ==========================================
    # 3. 相関計算 (ACF) - ここで青い点が決まる
    # ==========================================
    print("Calculating Segmented ACF...")
    # 分割平均法で計算
    raw_lags, raw_G, sem_G = calculate_segmented_acf(trace, n_segments=10)
    time_lags = raw_lags * SAMPLING_INTERVAL
    
    # 対数ビニング (表示用)
    lags_bin, G_bin, sigma_bin = log_binning_weighted(time_lags, raw_G, sem_G)
    
    # ==========================================
    # 4. 単純プロット (フィッティングなし)
    # ==========================================
    print("-" * 30)
    print(" Data Statistics")
    print("-" * 30)
    print(f" Max G(tau) : {np.max(G_bin):.6f}")
    print(f" Min G(tau) : {np.min(G_bin):.6f}")
    print(f" Last Point : {G_bin[-1]:.6f}")
    print("-" * 30)

    plt.figure(figsize=(8, 6))
    
    # エラーバー付きで生データを表示
    plt.errorbar(lags_bin, G_bin, yerr=sigma_bin, fmt='bo', alpha=0.8, label='Raw Data (ACF)', capsize=2)
    
    # ゼロライン（基準線）
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
    
    # グラフ表示
    plt.show()

if __name__ == "__main__":
    main()