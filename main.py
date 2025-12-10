import matplotlib.pyplot as plt
import numpy as np
from src.loader import load_fcs_trace, process_trace, analyze_noise_spectrum, remove_periodic_noise
from src.correlation import calculate_1d_acf, log_binning
from src.models import fit_calibration_data, fcs_calibration_model

def main():
    FILE_PATH = 'Data/test.tif'
    SAMPLING_INTERVAL = 2.0e-6 
    
    # ==========================================
    # 1. パラメータ設定 (Boundsを追加)
    # ==========================================
    PARAMS = {
        'N': 10.0,
        'D': 414e-12,     # Fix
        'w0': 0.25e-6,
        'wz': 1.5e-6,
        'T': 0.1,
        'tau_trip': 5e-6
    }

    FIX_FLAGS = {
        'N': False, 'D': True, 'w0': False, 'wz': False, 'T': False, 'tau_trip': False
    }

    # 【新規】パラメータの境界 (最小値, 最大値)
    # wzが負になったり、Tripletが100%超えたりするのを防ぐ
    BOUNDS = {
        'N': (0, np.inf),
        'D': (0, np.inf),
        'w0': (0.1e-6, 1.0e-6),  # 0.1um ~ 1um の範囲に限定
        'wz': (0.3e-6, 10.0e-6), # 0.3um ~ 10um (S>=3を想定して下限を高めに)
        'T': (0, 0.5),           # Triplet率は最大50%まで
        'tau_trip': (0, 100e-6)
    }

    # ==========================================
    # 2. データの読み込みと診断
    # ==========================================
    print(f"Loading {FILE_PATH}...")
    raw_trace = load_fcs_trace(FILE_PATH)
    if raw_trace is None: return

    # --- ノイズ解析 ---
    print("Analyzing Noise Spectrum...")
    freqs, power, peak_freq = analyze_noise_spectrum(raw_trace, SAMPLING_INTERVAL)
    print(f" >> Detected Dominant Frequency: {peak_freq:.1f} Hz")

    # ノイズ確認用プロット (FFT)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power)
    plt.xlim(0, 1000) # 0-1000Hzを表示
    plt.title(f"Power Spectrum (Peak at {peak_freq:.1f} Hz)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.grid(True)
    plt.show() # ここで変なピーク(50Hzや60Hzなど)がないか確認

    # --- ノイズ除去 (定常波がある場合) ---
    # もしピーク周波数が明確ならフィルタをかける
    # 例: 50Hz, 60Hz, またはその高調波
    if peak_freq > 10: # 10Hz以上の明確なノイズがある場合
        trace_filtered = remove_periodic_noise(raw_trace, SAMPLING_INTERVAL, peak_freq)
    else:
        trace_filtered = raw_trace

    # --- 退色補正 ---
    trace = process_trace(trace_filtered, window_size_ratio=0.01)

    # ==========================================
    # 3. 相関計算とフィッティング
    # ==========================================
    print("Calculating ACF...")
    raw_lags, raw_G = calculate_1d_acf(trace)
    time_lags = raw_lags * SAMPLING_INTERVAL
    
    # 範囲制限
    lags_bin, G_bin = log_binning(time_lags, raw_G)
    start_time = 10e-6
    mask = (lags_bin > start_time) & (lags_bin < 1.0)
    fit_lags = lags_bin[mask]
    fit_G = G_bin[mask]

    print("Fitting with Bounds...")
    # fit_calibration_dataにBOUNDSを渡す
    res = fit_calibration_data(fit_lags, fit_G, PARAMS, FIX_FLAGS, BOUNDS)

    # 結果表示
    w0_um = res['w0'] * 1e6
    wz_um = res['wz'] * 1e6
    S_ratio = wz_um / w0_um

    print("-" * 35)
    print(" Calibration Results (Corrected)")
    print("-" * 35)
    print(f" w0 (XY)   : {w0_um:.4f} [um]")
    print(f" wz (Z)    : {wz_um:.4f} [um]")
    print(f" S (Ratio) : {S_ratio:.2f}")
    print(f" N         : {res['N']:.2f}")
    print(f" Triplet   : {res['T']*100:.1f} [%]")
    print("-" * 35)

    # プロット
    model_G = fcs_calibration_model(fit_lags, res['N'], res['D'], res['w0'], res['wz'], res['T'], res['tau_trip'])

    plt.figure(figsize=(8, 6))
    plt.semilogx(fit_lags, fit_G, 'bo', label='Filtered Data')
    plt.semilogx(fit_lags, model_G, 'r-', linewidth=2, label='Fit Model')
    plt.xlabel('Lag Time [s]')
    plt.ylabel('Autocorrelation G(tau)')
    plt.title(f'Calibration: w0={w0_um:.3f} um')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()