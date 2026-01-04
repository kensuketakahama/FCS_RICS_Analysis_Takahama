# 顕微鏡と実験のパラメータ設定

# GUI上の初期値(変更不要)
W0 = 0.30               # radial beam waist (um)
WZ = 1.0                # axial beam waist (um)

# --- 解析パラメータ初期値(変更不要) ---
ROI_SIZE = 64           # 解析する領域のサイズ (2の乗数が望ましい: 64, 128, 256)
MOVING_AVG_WINDOW = 10  # 移動平均を引く際のフレーム幅

# --- 走査パラメータ (変更必要) ---
PIXEL_SIZE = 0.049       # 1ピクセルのサイズ (um)
PIXEL_DWELL_TIME = 2e-6 # 1ピクセルの滞留時間 (秒) 10us
LINE_TIME = 3.152e-3      # 1ラインの走査時間 (秒) 2ms (Retrace含む)