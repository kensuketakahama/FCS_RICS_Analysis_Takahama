import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

def load_tiff(filepath):
    """TIFファイルを読み込み、numpy配列(T, Y, X)で返す"""
    return tifffile.imread(filepath).astype(np.float32)

def subtract_moving_average(image_stack, window_size):
    """
    移動平均減算処理 (不動成分の除去)
    MIAの基本処理に相当
    """
    # 時間軸方向(axis=0)に対して移動平均を計算
    # mode='nearest' は端の処理
    moving_avg = uniform_filter(image_stack, size=(window_size, 1, 1), mode='nearest')
    
    # 元画像 - 移動平均 + 全体平均
    # これにより平均輝度レベルを維持しつつ、ゆっくり動く構造を除去
    global_mean = np.mean(image_stack)
    corrected_stack = image_stack - moving_avg + global_mean
    
    return corrected_stack