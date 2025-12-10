import matplotlib.pyplot as plt
import numpy as np
from src.loader import load_fcs_trace, process_trace, analyze_noise_spectrum, remove_periodic_noise
from src.correlation import calculate_1d_acf, log_binning
from src.models import fit_calibration_data, fcs_calibration_model

def main():
    # パスの入力
    FILE_PATH = 'hondasan/3.tif'
    SAMPLING_INTERVAL = 2.0e-6 
    
    # ==========================================
    # 1. パラメータ設定(ここは手動で入力)
    # ==========================================
    PARAMS = {
        'N': 10.0,
        'D': 414e-12,   
        'w0': 0.20e-6,   
        'wz': 1.0e-6,    
        'T': 0.1,
        'tau_trip': 5e-6
    }
    # ==========================================
    # 2. 固定するパラメータの選択
    # ==========================================
    FIX_FLAGS = {
        'N': False, 'D': True, 'w0': False, 'wz': False, 'T': False, 'tau_trip': False
    }

    BOUNDS = {
        'N': (0, np.inf),
        'D': (0, np.inf),
        'w0': (0.1e-6, 1.0e-6),
        'wz': (0.3e-6, 10.0e-6),
        'T': (0, 0.5),
        'tau_trip': (0, 100e-6)
    }

    print(f"Loading {FILE_PATH}...")
    raw_trace = load_fcs_trace(FILE_PATH)
    if raw_trace is None: return

    # ノイズ解析・除去・退色補正
    freqs, power, peak_freq = analyze_noise_spectrum(raw_trace, SAMPLING_INTERVAL)
    if peak_freq > 10:
        trace_filtered = remove_periodic_noise(raw_trace, SAMPLING_INTERVAL, peak_freq)
    else:
        trace_filtered = raw_trace
        
    trace = process_trace(trace_filtered, window_size_ratio=0.01)

    print("Calculating ACF...")
    raw_lags, raw_G = calculate_1d_acf(trace)
    time_lags = raw_lags * SAMPLING_INTERVAL
    
    # Log Binning (データ間引き)
    lags_bin, G_bin = log_binning(time_lags, raw_G)
    
    # マスク作成 (使用する範囲の指定)
    start_time = 10e-6  # 10us未満はAfterpulseノイズとしてカット
    end_time = 1.0      # 1秒以上は統計不足としてカット
    mask = (lags_bin > start_time) & (lags_bin < end_time)
    
    fit_lags = lags_bin[mask]
    fit_G = G_bin[mask]

    total_bins = len(lags_bin)       # Binningで作られた全点数
    used_bins = len(fit_lags)        # 実際にFittingに使った点数
    excluded_bins = total_bins - used_bins
    exclusion_rate = (excluded_bins / total_bins) * 100
    
    print("-" * 35)
    print(" Data Usage Statistics")
    print("-" * 35)
    print(f" Raw Data Points : {len(raw_trace)}")
    print(f" Binned Points   : {total_bins} (Compressed for Log Scale)")
    print(f" Used for Fit    : {used_bins}")
    print(f" Excluded        : {excluded_bins} points")
    print(f" Exclusion Rate  : {exclusion_rate:.1f} % (Filtered out)")
    print("-" * 35)
    # ------------------------------------

    print("Fitting with Bounds...")
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
    plt.semilogx(fit_lags, fit_G, 'bo', label=f'Used Data (n={used_bins})')

    inverse_mask = ~mask
    if np.any(inverse_mask):
        plt.semilogx(lags_bin[inverse_mask], G_bin[inverse_mask], 'x', color='gray', alpha=0.5, label='Excluded')

    plt.semilogx(fit_lags, model_G, 'r-', linewidth=2, label='Fit Model')
    plt.xlabel('Lag Time [s]')
    plt.ylabel('Autocorrelation G(tau)')
    plt.title(f'Calibration: w0={w0_um:.3f} um (Excluded: {exclusion_rate:.1f}%)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()