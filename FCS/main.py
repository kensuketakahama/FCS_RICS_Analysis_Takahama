import matplotlib.pyplot as plt
import numpy as np

# 必要なモジュールをインポート
from src.loader import load_fcs_trace, analyze_noise_spectrum, remove_periodic_noise, process_trace_detrend
from src.correlation import calculate_segmented_acf, log_binning_weighted
from src.models import fit_standard_data, fcs_standard_model

def main():
    FILE_PATH = 'Data/FCS1_TAMRA1000nM.tif' 
    SAMPLING_INTERVAL = 2.0e-6 
    
    # ==========================================
    # 0. 前処理設定
    # ==========================================
    USE_DETREND = False       # 退色補正ON
    DETREND_CUTOFF = 0.005   # 5ms

    # ==========================================
    # 1. パラメータ設定 (キャリブレーション用)
    # ==========================================
    PARAMS = {
        'N': 1000.0,      # 初期値
        'D': 414.0,       # TAMRA/Rh6Gの理論値 [μm^2/s]
        'w0': 0.2,       # 推定値 [μm]
        'wz': 1,        # 固定値 [μm] (w0の4~5倍)
        'T': 0.1,         
        'tau_trip': 1e-6,
        'y0': 0.0         
    }

    # 【重要】キャリブレーションのための固定設定
    # Dとwzを固定して、w0を正しく求めに行きます。
    FIX_FLAGS = {
        'N': False, 
        'D': True,       # ★固定 (414.0)
        'w0': False,     # Free (これを求めたい)
        'wz': False,      # ★固定 (1.0) -> S=4~5程度を維持させる
        'T': False,      
        'tau_trip': False,
        'y0': False     # Free (ベースライン補正)
    }

    BOUNDS = {
        'N': (0, np.inf),
        'D': (10.0, 2000.0),
        'w0': (0.1, 1.0),
        'wz': (0.5, 10.0),
        'T': (0, 0.5),
        'tau_trip': (0, 100e-6),
        'y0': (-0.1, 0.1)
    }

    # ==========================================
    # 2. データ読み込み & 前処理
    # ==========================================
    print(f"Loading {FILE_PATH}...")
    raw_trace = load_fcs_trace(FILE_PATH)
    if raw_trace is None: return

    # ノイズ除去
    freqs, power, peak_freq = analyze_noise_spectrum(raw_trace, SAMPLING_INTERVAL)
    if peak_freq > 10: 
        trace_filtered = remove_periodic_noise(raw_trace, SAMPLING_INTERVAL, peak_freq)
    else:
        trace_filtered = raw_trace

    # 退色補正 (Detrend)
    if USE_DETREND:
        print(f"Applying Detrend filter (Cutoff: {DETREND_CUTOFF*1000:.1f} ms)")
        trace = process_trace_detrend(trace_filtered, sampling_interval=SAMPLING_INTERVAL, cutoff_time=DETREND_CUTOFF)
    else:
        trace = trace_filtered

    if trace is None: return

    # ==========================================
    # 3. 相関計算
    # ==========================================
    print("Calculating Segmented ACF...")
    raw_lags, raw_G, sem_G = calculate_segmented_acf(trace, n_segments=10)
    time_lags = raw_lags * SAMPLING_INTERVAL
    
    lags_bin, G_bin, sigma_bin = log_binning_weighted(time_lags, raw_G, sem_G)
    
    # マスク処理
    start_time = 40e-6
    mask = (lags_bin > start_time) & (lags_bin < 1.0)
    
    fit_lags = lags_bin[mask]
    fit_G = G_bin[mask]
    
    # sigma (重み) の処理
    fit_sigma = sigma_bin[mask]
    if len(fit_sigma) > 0:
        mean_sigma = np.mean(fit_sigma[fit_sigma > 0]) if np.any(fit_sigma > 0) else 1.0
        fit_sigma[fit_sigma == 0] = mean_sigma
    else:
        print("Error: No data points left after masking.")
        return

    # ==========================================
    # 4. フィッティング
    # ==========================================
    print("Fitting...")
    res = fit_standard_data(fit_lags, fit_G, PARAMS, FIX_FLAGS, BOUNDS)

    # ==========================================
    # 5. 結果表示
    # ==========================================
    w0_val = res['w0']
    wz_val = res['wz']
    D_val  = res['D']
    N_val  = res['N']
    T_val  = res['T']
    y0_val = res['y0']
    
    ratio = w0_val / wz_val
    S_val = wz_val / w0_val

    # G(0)理論値
    if T_val < 1.0:
        G0_theoretical = ((1.0 / N_val) * (1.0 / (1.0 - T_val))) + y0_val
    else:
        G0_theoretical = 0

    print("=" * 40)
    print(" FINAL FITTING RESULTS")
    print(" (Standard Model + y0)")
    print("=" * 40)
    print(f" [Beam Parameters]")
    print(f"  w0 (Lateral)   : {w0_val:.4f} [μm]")
    print(f"  wz (Axial)     : {wz_val:.4f} [μm]")
    print(f"  S (wz / w0)    : {S_val:.4f}")
    print("-" * 40)
    print(f" [Molecular Parameters]")
    print(f"  D (Diffusion)  : {D_val:.2f} [μm^2/s]")
    print(f"  N (Molecules)  : {N_val:.2f}")
    print("-" * 40)
    print(f" [Amplitudes]")
    print(f"  G(0) Total     : {G0_theoretical:.6f}")
    print(f"  Background (y0): {y0_val:.6f}")
    print(f"  Triplet (T)    : {T_val*100:.1f} %")
    print("=" * 40)

    # プロット
    model_G = fcs_standard_model(fit_lags, 
                                res['N'], res['D'], res['w0'], res['wz'], 
                                res['T'], res['tau_trip'], res['y0'])

    plt.figure(figsize=(8, 6))
    plt.errorbar(fit_lags, fit_G, yerr=fit_sigma, fmt='bo', alpha=0.5, label='Averaged Data', capsize=2)
    plt.semilogx(fit_lags, model_G, 'r-', linewidth=2, label=f'Fit (w0={w0_val:.3f})')
    plt.axhline(y=y0_val, color='green', linestyle='--', alpha=0.5, label='Background')
    
    plt.xlabel('Lag Time [s]')
    plt.ylabel('Autocorrelation G(tau)')
    plt.title(f'Calibration: w0={w0_val:.3f} μm (D={D_val:.0f} fixed)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()