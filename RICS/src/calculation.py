import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def calculate_2d_acf(roi_stack):
    """
    時系列画像スタックから空間2D自己相関関数を計算する
    RICSでは各フレームで空間相関を取り、それを平均する
    """
    T, H, W = roi_stack.shape
    G_sum = np.zeros((H, W), dtype=np.float64)
    valid_frames = 0

    for t in range(T):
        img = roi_stack[t, :, :]
        mean_I = np.mean(img)
        
        if mean_I == 0:
            continue

        # FFTによる相関計算
        # パワースペクトル = FFT * FFT_conjugate
        F = fft2(img)
        P = F * np.conj(F)
        acf = np.real(ifft2(P))
        
        # 原点移動 (中心にピークが来るように)
        acf = fftshift(acf)
        
        # 正規化: <I(r)I(r+dr)> / <I>^2 - 1
        # H*W で割るのはFFTの性質上の補正
        norm_acf = acf / (mean_I**2 * H * W) - 1
        
        G_sum += norm_acf
        valid_frames += 1

    if valid_frames == 0:
        return np.zeros((H, W))
        
    return G_sum / valid_frames