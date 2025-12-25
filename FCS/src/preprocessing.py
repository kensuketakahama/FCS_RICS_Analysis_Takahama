import numpy as np
import tifffile

def load_tiff(filepath):
    """
    Load Multi-page TIFF as numpy array (T, H, W) or (H, W)
    """
    return tifffile.imread(filepath)

def subtract_moving_average(stack, window_size):
    """
    [RICS用] 2D画像スタック用の移動平均減算
    """
    T, H, W = stack.shape
    if T < window_size:
        return stack # データが短すぎる場合は何もしない
        
    # 移動平均を計算 (Uniform filter)
    # scipy.ndimage.uniform_filter1d だと早いが、ここでは簡易実装
    kernel = np.ones(window_size) / window_size
    
    # 時間軸(axis=0)方向に移動平均
    # メモリ節約のためループ処理、あるいはscipyを使用
    from scipy.ndimage import uniform_filter1d
    stack_mean = uniform_filter1d(stack, size=window_size, axis=0, mode='nearest')
    
    # 差分 + 全体平均 (負の値を防ぐためオフセットを戻すのが一般的だが、
    # RICS計算では平均0の揺らぎ成分がほしいので、平均を引いた状態にする)
    return stack - stack_mean

def detrend_1d_trace(trace, pixel_time, cutoff_time=0.005):
    """
    [FCS用] 1次元時系列データの退色補正 (Detrend)
    移動平均成分を差し引いて、低周波ノイズ（退色）を除去する。
    
    cutoff_time: 移動平均の窓幅 (秒)。これより遅い変動を除去する。
    """
    n_points = len(trace)
    # 窓幅 (データ点数)
    window_size = int(cutoff_time / pixel_time)
    
    if window_size < 3:
        return trace # 窓が小さすぎる場合は何もしない
    if window_size > n_points:
        window_size = n_points

    # 移動平均 (平滑化トレンド) を計算
    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(trace, size=window_size, mode='nearest')
    
    # 元データ - トレンド + 全体平均
    # (FCS計算では平均値で正規化するため、DC成分(平均値)を残した形で出力する)
    global_mean = np.mean(trace)
    detrended = (trace - trend) + global_mean
    
    return detrended