# FCS Analysis App (GUI)

Pythonによる蛍光相関分光法 (Fluorescence Correlation Spectroscopy, FCS) 解析用GUIアプリケーションです。
時系列の蛍光強度データ（Trace）から自己相関関数（ACF）を計算し、3次元拡散モデルを用いて拡散係数や粒子数を算出します。

## 主な機能

* **GUI操作:** Tkinterを用いた直感的な操作画面。
* **データ読み込み:** Multi-page TIFF形式のPoint FCSデータに対応。
* **前処理 (Preprocessing):**
    * **Bleach Correction (Detrend):** 移動平均差し引き法による退色補正。
* **ACF計算:**
    * **Segmented ACF:** データを分割して計算・平均化することでノイズを低減。
    * **Log Binning:** 対数等間隔ビニングによるデータ点の間引きと重み付け（Fitting精度の向上）。
    * **All Points Mode:** ビニングなしの全データ点を用いた解析も可能。
* **フィッティング (Fitting):**
    * **Standard Model:** 1成分3次元拡散 + トリプレット + オフセット項を用いた非線形最小二乗法。
    * **Fix/Free制御:** 各パラメータ（$D, N, w_0, \tau_{trip}$ 等）の固定/変動を個別に設定可能。

## 環境構築 (Requirements)

以下のライブラリが必要です。プロジェクトルートの仮想環境で実行することを推奨します。

* Python 3.10+
* numpy
* scipy
* matplotlib
* tifffile

## 使い方 (Usage)

1.  **起動:**
    ```bash
    python fcs_gui.py
    ```
2.  **データ読み込み:**
    * "Load Data" ボタンから `.tif` ファイルを選択します。
3.  **解析設定 (Config & Preprocessing):**
    * **Pixel Time (us):** サンプリング間隔（滞留時間）を入力します。非常に重要です。
    * **Bleach Correction:** 必要に応じて "Apply Detrend" にチェックを入れ、Cutoff時間（ms）を設定します。
    * **ACF Calculation Mode:** 通常は "Use Log Binning" をONにすることを推奨します。全データを見たい場合はOFFにします。
4.  **ビームパラメータ設定:**
    * **w0 (Lateral):** 焦点半径。キャリブレーション等で得られた値を入力し、基本的には "Fix" します。
    * **wz (Axial):** 軸方向半径。$S = w_z/w_0$ (構造因子) の決定に使われます。
5.  **フィッティング (Fitting):**
    * 初期値 (Initial Guess) を入力し、固定したいパラメータの "Fix" にチェックを入れます。
    * "Run Fitting" で解析を実行します。
6.  **結果保存:**
    * "Save Graph Image" でグラフを保存できます。

## フィッティングモデル (Fitting Model)

本アプリケーションでは、以下の**1成分3次元拡散モデル（トリプレット項・オフセット付き）**を使用しています。

### 自己相関関数 (ACF) の定義
$$
G(\tau) = \frac{1}{N} \cdot G_{diff}(\tau) \cdot G_{trip}(\tau) + y_0
$$

各項の詳細は以下の通りです。

1.  **拡散項 (Diffusion Term):** 3次元ガウス型焦点領域における自由拡散
    $$
    G_{diff}(\tau) = \left( 1 + \frac{\tau}{\tau_D} \right)^{-1} \left( 1 + \frac{\tau}{S^2 \tau_D} \right)^{-0.5}
    $$
    * ここで、拡散時間 $\tau_D$ と構造因子 $S$ は以下の関係にあります。
        $$
        \tau_D = \frac{w_0^2}{4D}, \quad S = \frac{w_z}{w_0}
        $$

2.  **トリプレット項 (Triplet Term):** 暗状態への遷移による明滅成分
    $$
    G_{trip}(\tau) = 1 + \frac{T}{1-T} \exp\left( -\frac{\tau}{\tau_{trip}} \right)
    $$

### パラメータの意味
| パラメータ | 記号 | 説明 | 単位 |
| :--- | :---: | :--- | :--- |
| **Diffusion Coeff** | $D$ | 拡散係数 (求めたい値) | $\mu m^2/s$ |
| **Number of Molecules** | $N$ | 焦点領域内の平均粒子数 | - |
| **Beam Waist (Lateral)** | $w_0$ | 焦点半径 (横方向) | $\mu m$ |
| **Beam Waist (Axial)** | $w_z$ | 焦点半径 (縦方向) | $\mu m$ |
| **Triplet Fraction** | $T$ | トリプレット(暗)状態の割合 ($0 \le T < 1$) | - |
| **Triplet Time** | $\tau_{trip}$ | トリプレット緩和時間 | $s$ |
| **Offset** | $y_0$ | ベースラインオフセット (背景光など) | - |

## Author
Kensuke Takahama