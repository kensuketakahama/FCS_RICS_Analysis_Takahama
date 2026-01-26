import numpy as np

def rics_3d_equation(xy, D, G0, w0, wz, pixel_size, pixel_dwell, line_time, 
                     T=0.0, tau_trip=0.0, use_triplet=False, y0=0.0):
    """
    RICS 3D diffusion model.
    Includes optional Triplet blinking term and Offset (y0).
    """
    # 1次元配列として受け取った座標を展開
    xi = xy[0] * pixel_size  # spatial lag x (m)
    psi = xy[1] * pixel_size # spatial lag y (m)
    
    # 時空間ラグの計算 (Time lag at xi, psi)
    tau = np.abs(xi * pixel_dwell / pixel_size + psi * line_time / pixel_size)
    
    # --- 拡散項 (Diffusion Term) ---
    # G_diff = (1 + 4D*tau/w0^2)^-1 * (1 + 4D*tau/wz^2)^-0.5
    w0_sq = w0**2
    wz_sq = wz**2
    
    term1 = (1 + (4 * D * tau) / w0_sq) ** -1
    term2 = (1 + (4 * D * tau) / wz_sq) ** -0.5
    G_diff = term1 * term2
    
    # --- トリプレット項 (Triplet Term) ---
    G_trip = 1.0
    if use_triplet and T > 0 and tau_trip > 0:
        # Tが1.0以上になると発散するためキャップする
        if T >= 1.0: T = 0.999
        G_trip = 1.0 + (T / (1.0 - T)) * np.exp(-tau / tau_trip)
    
    # --- スキャン空間相関項 (Spatial Scanning Term) ---
    w0_sq_t = w0_sq + 4 * D * tau
    
    # 係数部分: G0 / N 相当 (ここではG0としてまとめる)
    amplitude_factor = (w0_sq / w0_sq_t)
    
    spatial_exp = np.exp( - (xi**2 + psi**2) / w0_sq_t )
    
    # 標準的なRICS式（Digman et al.）
    # G_val = G0 * G_trip * G_diff * S_scan
    G_val = G0 * G_trip * amplitude_factor * term2 * spatial_exp
        
    # オフセットを加算して返す
    return G_val.ravel() + y0