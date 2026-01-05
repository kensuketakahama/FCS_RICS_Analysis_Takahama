# RICS Analysis App (Python)

Pythonによる Raster Image Correlation Spectroscopy (RICS) 解析用GUIアプリケーションです。
顕微鏡で取得したMulti-tifデータから、拡散係数（Diffusion Coefficient）を算出・マッピングします。

## 主な機能 (v15.0)

* **画像ビューア:**
    * **Frame Slider:** 全フレームの平均画像だけでなく、スライダー操作で個別のフレーム（経時変化）を確認可能。
    * **ROI選択:** ドラッグによる任意領域指定、または "Select Full Image" ボタンによる全画素指定。
* **前処理:** 移動平均（Moving Average）による背景（不動成分）除去。
* **ACF計算:** 高速フーリエ変換（FFT）を用いた2D自己相関関数の計算。
* **フィッティング:** 3D拡散モデルを用いたカーブフィッティング。
    * **Auto-Detect Range:** 相関関数の形状（単調減少区間）を解析し、フィッティング範囲を自動決定する機能。
    * **Pixel Jump (Omit Center):** ショットノイズ除去のため中心点を除外。
    * **Fix/Free制御:** 各パラメータの固定/変動を個別に設定可能。
* **ヒートマップ作成:** Scanning RICSによる拡散係数のマッピング。
    * **Live Plot:** 計算中のピクセルのフィッティング状況（グラフと曲線）をリアルタイムでアニメーション表示。
    * **Fixed Scale:** ライブ表示時のグラフスケールを固定し、変化を見やすくする機能。
    * **表示設定:** 補間（Interpolation）の切り替え、パーセンタイルによるオートスケール、閾値設定など。
* **画像保存:** 解析結果グラフやヒートマップのJPEG/PNG保存。

## 環境構築 (Requirements)

以下のライブラリが必要です。プロジェクトルートの仮想環境で実行することを推奨します。

* Python 3.10+
* numpy
* scipy
* matplotlib
* tifffile

### macOS でのセットアップ手順

1.  **Homebrew で Python と Tkinter をインストール**
    システム標準のPythonではなく、最新のPython環境を整えます。
    ```bash
    brew install python
    brew install python-tk
    ```
    ※ `python-tk` が重要です。これがないとGUIが正しく描画されません。

2.  **リポジトリのクローン（またはダウンロード）**
    ```bash
    git clone [https://github.com/kensuketakahama/FCS_RICS_Analysis_Takahama.git](https://github.com/kensuketakahama/FCS_RICS_Analysis_Takahama.git)
    cd FCS_RICS_Analysis_Takahama
    ```

3.  **仮想環境 (venv) の作成**
    ```bash
    python3 -m venv venv
    ```

4.  **仮想環境の有効化**
    ```bash
    source venv/bin/activate
    ```

5.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```
    ※ `requirements.txt` がない場合は以下を実行:
    `pip install numpy scipy matplotlib tifffile`

### Windows でのセットアップ手順

1.  Python公式サイトからPythonをインストール（"tcl/tk and IDLE" にチェックを入れる）。
2.  PowerShellなどでプロジェクトフォルダに移動。
3.  仮想環境作成: `python -m venv venv`
4.  有効化: `.\venv\Scripts\Activate`
5.  ライブラリインストール: `pip install -r requirements.txt`

## 使い方 (Usage)

### 1. アプリケーションの起動
仮想環境を有効化した状態で実行してください。

```bash
python gui_app.py
```
### 2. 解析パラメータの設定 (`config.py`)
実験条件に合わせて、`config.py` 内の以下の値を編集してください。**ここが合っていないと拡散係数が正しい桁になりません。**

* `PIXEL_SIZE`: 1ピクセルのサイズ (um)
* `PIXEL_DWELL_TIME`: 1ピクセルの滞留時間 (s)
* `LINE_TIME`: 1ラインの走査時間 (s)
* `W0`, `WZ`: レーザーのビームウェスト径 (um)

### 3. GUI操作フロー

1.  **Load Data:**
    * TIFファイル（Multi-stack）を読み込みます。
    * **Frame Viewer:** スライダーを動かすと個別のフレームを表示できます。"Show Average Image" で平均画像に戻ります。

2.  **ROI & Preprocessing:**
    * 画像上でドラッグしてROI（解析領域）を選択します。画像全体を解析したい場合は "Select Full Image" を押します。
    * `Mov.Avg` で背景除去の強度（フレーム数）を調整し "Update" します。

3.  **Range & Omit:**
    * `Omit Radius` を `1` 以上に設定して中心ノイズを除去します。
    * **Auto-Detect Fit Range:** チェックを入れると、ピクセルごとに最適なフィッティング範囲（単調減少区間）を自動判定します。
    * チェックを外している場合は、グラフ上の赤い点線をドラッグして手動で範囲を設定します。

4.  **Single Point Fitting (プレビュー):**
    * "Run Fitting" を押すと、現在選択されているROI全体の平均ACFに対してフィッティングを行い、パラメータ（Dなど）の目安を確認できます。
    * ここでグラフのスケールが決まり、後のLive Plotで使用されます。

5.  **Heatmap Analysis:**
    * `Win Size` (局所領域サイズ) と `Step` (計算間隔) を設定します。
    * **Generate Heatmap (Live Plot):** 計算を開始します。右側のグラフに解析中のピクセルのフィッティング状況がリアルタイム表示されます。
    * **Display Settings:**
        * **Scale by Percentile:** チェックを入れると、外れ値を除外してコントラストを自動調整します（例: 95%）。
        * **Interpolation:** `nearest`（ドット表示）、`bicubic`（滑らか表示）などを切り替えられます。
        * "Re-draw Map Only" ボタンで、計算をやり直さずに表示設定だけを変更できます。

## トラブルシューティング

### Q. アプリを起動しても画面が真っ白 (macOS)
**A.** macOS標準のPython (`/usr/bin/python`) ではなく、Homebrew等で入れたPython (`/opt/homebrew/bin/python3` 等) を使用してください。

### Q. ヒートマップ作成が遅い
**A.** `Step` の値を大きくしてください（例: 4や8）。`Step=1` は全画素計算となるため時間がかかります。また、Live Plotの描画負荷を抑えるため、画面更新は一定間隔に制限されています。

### Q. フィッティングがうまくいかない / 青一色になる
**A.**
1.  `config.py` の時間パラメータを確認してください。
2.  `Omit Radius` を適切に設定してください。
3.  **Auto-Detect Fit Range** をONにしてみてください。
4.  ヒートマップの "Scale by Percentile" をONにするか、適切な "Max D" を設定して表示レンジを調整してください。

# Appendix: 技術詳細

## 1. RICS フィッティングモデル式

本アプリケーションでは、3次元自由拡散モデルに走査型顕微鏡（LSM）の空間走査特性を組み込んだ標準的なRICS式を使用しています。
ある空間ラグ $(\xi, \psi)$ における自己相関関数 $G(\xi, \psi)$ は以下の式で表されます。

$$
G(\xi, \psi) = \frac{\gamma}{N} \cdot G_{diff}(\xi, \psi) \cdot S(\xi, \psi) + G_0
$$

ここで、各項の定義は以下の通りです。

### 拡散項 (Diffusion Term)
ブラウン運動による粒子の移動を表す項です。

$$
G_{diff}(\xi, \psi) = \left( 1 + \frac{4 D \tau}{w_0^2} \right)^{-1} \left( 1 + \frac{4 D \tau}{w_z^2} \right)^{-1/2}
$$

### 走査項 (Scanning Term)
レーザービームの空間プロファイル（ガウス分布）と走査ラグの相関を表す項です。

$$
S(\xi, \psi) = \exp \left( -\frac{\delta r^2}{w_0^2 \left( 1 + \frac{4 D \tau}{w_0^2} \right)} \right)
$$

### 変数定義
* **ラグ時間 $\tau$**: 空間的な距離を、走査速度に基づいて時間に換算したものです。

$$
\tau = |\xi| \cdot \tau_p + |\psi| \cdot \tau_l
$$

* **空間距離 $\delta r^2$**:

$$
\delta r^2 = (\xi \cdot \delta_{pixel})^2 + (\psi \cdot \delta_{pixel})^2
$$

| 変数 | 説明 | 設定箇所 |
| :--- | :--- | :--- |
| $D$ | 拡散係数 (Diffusion Coefficient) | フィッティングパラメータ |
| $N$ | 焦点領域内の平均粒子数 | フィッティングパラメータ |
| $G_0$ | ベースラインオフセット | フィッティングパラメータ |
| $w_0$ | 焦点半径 (Radial Beam Waist) | `config.py` (W0) |
| $w_z$ | 軸方向半径 (Axial Beam Waist) | `config.py` (WZ) |
| $\tau_p$ | ピクセル滞留時間 (Pixel Dwell Time) | `config.py` |
| $\tau_l$ | ライン走査時間 (Line Time) | `config.py` |
| $\delta_{pixel}$ | ピクセルサイズ | `config.py` |

---

## 2. ヒートマップ作成アルゴリズム

Scanning RICSによる拡散係数マップの生成プロセスは以下の通りです。

1.  **パラメータ設定**
    * **Window Size ($W_{in}$):** 局所解析を行う正方形領域のサイズ（例: 32 px）。
    * **Step ($S$):** 次の解析ポイントへ移動するピクセル幅（例: 4 px）。

2.  **走査ループ (Iterative Calculation)**
    画像全体に対して、左上から右下へ $S$ ピクセルずつ移動しながら以下の処理を繰り返します。

    * **Step 2-1: ROI抽出**
        中心座標 $(x, y)$ に対し、 $W_{in} \times W_{in}$ サイズの時系列スタックデータを切り出します。
    * **Step 2-2: 2D-ACF計算**
        FFT（高速フーリエ変換）を用いて、抽出したスタックデータの2次元自己相関関数を計算します。
    * **Step 2-3: マスキング (Data Selection)**
        計算されたACF曲面に対し、以下の領域を除外（マスク）します。
        * **Omit Center:** 中心 $(\xi=0, \psi=0)$ 付近のショットノイズ領域。
        * **Fit Range:** 信号が減衰しきった後のノイズ領域（手動設定 または Auto-Detect）。
    * **Step 2-4: 非線形最小二乗法**
        有効なデータ点に対して上記のRICS式を適用し、`scipy.optimize.curve_fit` を用いて最適な $D$ を推定します。
    * **Step 2-5: マッピング**
        得られた $D$ の値を、座標 $(x, y)$ のピクセル値としてヒートマップ配列に格納します。

3.  **可視化と補間**
    計算された離散的な $D$ の値を、指定された補間方法（Nearest / Bicubic等）でカラーマップとして表示します。

---

## 3. Auto-Detect Fit Range アルゴリズム

各ピクセルにおいて、フィッティングに使用する最適なラグ範囲（Fit Range）を自動決定するロジックです。
ACFは理想的には中心から離れるほど単調に減衰しますが、遠方ではノイズにより値が振動したり再上昇したりします。このアルゴリズムは「単調減少が維持されている区間」のみを抽出します。

1.  **プロファイル抽出**
    計算された2D-ACFから、中心を通る **X軸断面** と **Y軸断面** のデータを取得します。

2.  **平滑化 (Smoothing)**
    局所的なノイズによる誤検知を防ぐため、3点移動平均フィルタを適用します。
    
$$ 
S_i = \frac{v_{i-1} + v_i + v_{i+1}}{3}
$$

3.  **微分と閾値判定 (Gradient Check)**
    平滑化したデータの隣接差分（勾配）を計算します。
    理想的な減衰局面では $\Delta < 0$ となります。
   中心から外側へ探索し、初めて **$\Delta > 0$ （値が増加に転じた、または底を打った）** となる点を特定します。

$$
\Delta = S_{i+1} - S_i
$$

4.  **範囲の決定**
    特定された位置をそのピクセルにおけるフィッティング限界点とし、それより外側のデータをフィッティング計算から除外します。

## Author
Kensuke Takahama
