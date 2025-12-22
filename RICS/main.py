import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
import os

# 自作モジュールのインポート
import config as cfg
from src import preprocessing as prep
from src import calculation as calc
from src import model

def main():
    # 1. データの読み込み
    filename = "TAMRA1000nMRICS.tif" # dataフォルダに入れておく
    filepath = os.path.join("Data", filename)
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} が見つかりません。")
        # ダミーデータを作成して動作確認する場合
        print("ダミーデータを生成します...")
        data = np.random.rand(100, 256, 256).astype(np.float32)
    else:
        print(f"Loading {filename}...")
        data = prep.load_tiff(filepath)

    # 2. 前処理 (移動平均除去)
    print("Preprocessing (Moving Average Subtraction)...")
    processed_data = prep.subtract_moving_average(data, cfg.MOVING_AVG_WINDOW)

    # 3. 中心部分のROIを切り出し (テストとして画像中央を解析)
    _, H, W = processed_data.shape
    cy, cx = H // 2, W // 2
    r = cfg.ROI_SIZE // 2
    roi = processed_data[:, cy-r:cy+r, cx-r:cx+r]
    
    print(f"Calculating ACF for ROI size: {cfg.ROI_SIZE}x{cfg.ROI_SIZE}...")
    
    # 4. ACF計算
    acf_map = calc.calculate_2d_acf(roi)
    
    # 5. フィッティング準備
    # グリッド作成 (中心を0とするラグ座標)
    y_range = np.arange(-r, r)
    x_range = np.arange(-r, r)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # フィッティング用にデータを1次元化
    xdata = np.vstack((X_grid.ravel(), Y_grid.ravel()))
    ydata = acf_map.ravel()

    # フィッティング関数に固定パラメータを埋め込む
    # D と G0 以外を固定
    fit_func = partial(model.rics_3d_equation, 
                       w0=cfg.W0, wz=cfg.WZ,
                       pixel_size=cfg.PIXEL_SIZE,
                       pixel_dwell=cfg.PIXEL_DWELL_TIME,
                       line_time=cfg.LINE_TIME)
    
    # 6. フィッティング実行
    print("Fitting...")
    # 初期値 [D, G0] : D=10 (仮), G0=0.01 (仮)
    p0 = [10.0, 0.01]
    
    try:
        popt, pcov = curve_fit(fit_func, xdata, ydata, p0=p0, bounds=(0, np.inf))
        D_fit, G0_fit = popt
        print("-" * 30)
        print(f"解析結果:")
        print(f"拡散係数 D  = {D_fit:.2f} [um^2/s]")
        print(f"相関振幅 G0 = {G0_fit:.4f}")
        print("-" * 30)
        
        # 7. 結果の可視化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 実測ACF
        im1 = axes[0].imshow(acf_map, cmap='jet', extent=[-r, r, -r, r])
        axes[0].set_title('Calculated ACF')
        plt.colorbar(im1, ax=axes[0])
        
        # フィッティング結果の再構成
        fit_model_map = fit_func(xdata, D_fit, G0_fit).reshape(cfg.ROI_SIZE, cfg.ROI_SIZE)
        im2 = axes[1].imshow(fit_model_map, cmap='jet', extent=[-r, r, -r, r])
        axes[1].set_title(f'Fitted Model (D={D_fit:.1f})')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
    except RuntimeError:
        print("Fitting failed. Try adjusting initial parameters or checking data quality.")

if __name__ == "__main__":
    main()