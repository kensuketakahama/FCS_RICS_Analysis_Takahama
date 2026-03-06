# FCS Analysis App (GUI)

Pythonによる蛍光相関分光法 (Fluorescence Correlation Spectroscopy, FCS) 解析用GUIアプリケーションです。
時系列の蛍光強度データ（Trace）から自己相関関数（ACF）を計算し、3次元拡散モデルを用いて拡散係数や粒子数を算出します。

## 主な機能

- **GUI操作:** Tkinterを用いた直感的な操作画面。
- **データ読み込み:** Multi-page TIFF形式のPoint FCSデータに対応。
- **前処理 (Preprocessing):**
  - **Bleach Correction (Detrend):** 移動平均差し引き法による退色補正。
- **ACF計算:**
  - **Segmented ACF:** データを分割して計算・平均化することでノイズを低減。
  - **Log Binning:** 対数等間隔ビニングによるデータ点の間引きと重み付け（Fitting精度の向上）。
  - **All Points Mode:** ビニングなしの全データ点を用いた解析も可能。
- **フィッティング (Fitting):**
  - **Standard Model:** 1成分3次元拡散 + トリプレット + オフセット項を用いた非線形最小二乗法。
  - **Fix/Free制御:** 各パラメータ（$`D, N, w_0, \tau_{trip}`$ 等）の固定/変動を個別に設定可能。

## 環境構築 (Requirements)

以下のライブラリが必要です。プロジェクトルートの仮想環境で実行することを推奨します。

- Python 3.10+
- numpy
- scipy
- matplotlib
- tifffile

## 使い方 (Usage)

1.  **起動**:
    ```bash
    python3 fcs_gui.py
    ```
2.  **データ読み込み:**
    - "Load Data" ボタンから `.tif` ファイルを選択します。
3.  **解析設定 (Config & Preprocessing):**
    - **Pixel Time (us):** 顕微鏡で設定しているピクセルタイムを入力して下さい。
    - **Bleach Correction:** 必要に応じて "Apply Detrend" にチェックを入れて下さい。褪色補正が行われます。
    - **Trace Analysis Range (s):** 解析範囲の設定です。画面上部のintensityを見て、どの範囲を解析範囲として用いるべきかを決定して下さい。
    - **Log Binning:** データをいくつかのセグメントに分けて、それぞれのACFを計算し、平均値を取る機能です。従って、この機能をonにして、segmentを1より大きな値にすると各プロットにエラーバーが現れます。通常は Log BinningをONにすることを推奨します。(例えば38000点のデータプロットがあり、segmentを10とすると3800点ずつを一つの塊としてACFを計算します。ACFの計算は、各セグメントに対してpixel timeずつずらしていった時の自己相関を見ます。その後、各時間ラグに対するACFの値に関して平均値を取るということをします。)
4.  **Fitting Parametersの入力:**
    各パラメータの入力・fix・limit設定を行って下さい。
5.  **フィッティング (Fitting):**
    "Run Fitting" で解析を実行します。
6.  **結果保存:**
    "Save Graph Image" でグラフを保存できます。

## Appendix

### フィッティングモデル (Fitting Model)

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

ここで、拡散時間 $`\tau_D`$ と構造因子 $`S`$ は以下の関係にあります。

$$
\tau_D = \frac{w_0^2}{4D}, \quad S = \frac{w_z}{w_0}
$$

2.  **トリプレット項 (Triplet Term):** 暗状態への遷移による明滅成分

$$
G_{trip}(\tau) = 1 + \frac{T}{1-T} \exp\left( -\frac{\tau}{\tau_{trip}} \right)
$$

### パラメータの意味

| パラメータ               |     記号      | 説明                                       | 単位        |
| :----------------------- | :-----------: | :----------------------------------------- | :---------- |
| **Diffusion Coeff**      |      $D$      | 拡散係数 (求めたい値)                      | $\mu m^2/s$ |
| **Number of Molecules**  |      $N$      | 焦点領域内の平均粒子数                     | -           |
| **Beam Waist (Lateral)** |     $w_0$     | 焦点半径 (横方向)                          | $\mu m$     |
| **Beam Waist (Axial)**   |     $w_z$     | 焦点半径 (縦方向)                          | $\mu m$     |
| **Triplet Fraction**     |      $T$      | トリプレット(暗)状態の割合 ($0 \le T < 1$) | -           |
| **Triplet Time**         | $\tau_{trip}$ | トリプレット緩和時間                       | $s$         |
| **Offset**               |     $y_0$     | ベースラインオフセット (背景光など)        | -           |

### Detrend処理の仕様について

本ソフトウェアの "Bleach Correction (Detrend)" 機能は、長時間の測定による蛍光分子の退色や装置のドリフトなど、分子拡散よりも遅い強度の変動を除去するために使用されます。内部では以下の仕様に基づいて処理が行われています。

#### 1. 補正計算式 (加法的な補正)

本ソフトウェアでは、局所平均に対する比率をとる乗法的な補正ではなく、生の時系列データからトレンド成分を直接引き算し、全体の平均強度を足し戻す加法的な補正を採用しています。

$$I_{\text{corrected}}(t) = I(t) - I_{\text{trend}}(t) + \langle I \rangle_{\text{global}}$$

- $I_{\text{corrected}}(t)$: 補正後の蛍光強度
- $I(t)$: 生の蛍光強度
- $I_{\text{trend}}(t)$: 移動平均によって抽出されたトレンド成分
- $\langle I \rangle_{\text{global}}$: 生データ全体の平均強度

#### 2. トレンド成分の算出アルゴリズム

- **移動平均の計算:** 指定されたウィンドウサイズに基づく単純移動平均（`scipy.ndimage.uniform_filter1d`）を用いて $I_{\text{trend}}(t)$ を算出しています。
- **端の処理:** データの開始時および終了時においてウィンドウがはみ出す部分については、最も近い端のデータ値をそのまま延長して計算する処理（`mode='nearest'`）を適用しています。

#### 3. Cutoff Time (カットオフ時間)

- **デフォルト設定:** **5.0 ms**です。これは`fcs_gui.py`内の`self.detrend_cutoff_var`で設定されています。変更が必要な場合は値をコード内で変更して下さい。
- **ウィンドウサイズの決定:** 移動平均の窓幅（データ点数）は `Cutoff Time / Pixel Time` で自動算出されます。このカットオフ時間より遅い（周期が長い）変動がトレンドとして除去され、速いゆらぎ成分のみが後段のACF計算へと送られます。

## Author

Kensuke Takahama
