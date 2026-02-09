# RICS Analysis App (Python)

Pythonによる Raster Image Correlation Spectroscopy (RICS) 解析用GUIアプリケーションです。
顕微鏡で取得したMulti-tifデータから、拡散係数（Diffusion Coefficient）を算出・マッピングします。

## 主な機能

* **単一解析:**
    * デフォルトの画面です。Multi-tiffファイルに対して、拡散係数の計算、ビームウェストの決定、ヒートマップの作成が可能です。
* **Batch Config Window:** 
    * 複数のMulti-tiffファイルに対して、拡散係数ヒートマップを作成する機能です。指定したディレクトリ内のMulti-tiffファイル全てに対して、ヒートマップの作成を実行します。
* **ROI Analysis Tool:** 
    * 作成済みの単一のヒートマップ(csv形式)に対して、さらなる解析を行うためのツールです。ヒートマップとして表示する拡散係数マップの最大値の変更、拡散係数のヒストグラムの作成、任意のROIに対するヒートマップの切り抜き、window sizeとの比較が可能です。
* **Multi-File Analysis:** 
    * 作成済みの複数のートマップ(csv形式)に対して、一括で解析を行うためのツールです。ヒートマップとして表示する拡散係数マップの最大値の変更、拡散係数のヒストグラムの作成、またpdf形式で比較用のファイルを出力できます。

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
### 2. 単一解析
## 2.0. GUIの説明
左画面はツールバー、右画面は左上が蛍光画像、右上が3D表示のACFとフィッティングカーブ、下がx方向(y=0)、y方向(x=0)のACFプロットとフィッティングカーブです。
## 2.1.Data Loading
Load Single TIF DATAを選択し、解析対象のMilti-Tiffファイルを選択してください。その際に、単一の蛍光画像であることに注意してください。(複数の蛍光チャンネルを持つTiffファイルには対応していません。その場合はImage Jなどを用いて、単一蛍光画像に分割してください。)
スライドショーについては利用可能ですが、Moving AverageやROIを設定したままだと、描画に非常に負荷がかかり、処理落ちします。その場合は、Stopを押して、スライドショーを停止・もしくはアプリの再起動を行なってください。

## 2.2.Scan Parameters
* `Pixel_Size`: 1ピクセルのサイズ (nm)
* `Pixel_Dwell`: Pixel Timeに相当。1ピクセルの滞留時間 (us)
* `Line_Time`: 1ラインの走査時間 (ms)

## 2.3.ROI & Processing
このウィンドウでACFの計算を行います。各種パラメータを設定後、Update Image & ACFを押すことで、処理後の蛍光画像とACFが出力されます。フィッティングを行う前に必ずこのボタンを押してください。
* `Moving Average` : ここで選択した枚数に対して、輝度の平均値をそれぞれのピクセルから差し引く処理を行います。例えば、Moving Average=5とすると、あるピクセルにおいて、同一ピクセルの前後を含めた5枚の輝度平均値を差し引きます。使わない場合はチェックを外してください。褪色や重心移動の影響を差し引くために利用します。
* `ROI Config`: ROIの形状をここで選択します。ACFの計算とフィッティングについては、このROIの内部についてのみ行われます。また、ヒートマップを作成する場合はこのROI内に対して作成が行われます。作成したROIはSave ROIにて、jpeg形式で保存可能です。(本アプリケーションの他のツールにも再利用可能です。)
    * `Full Image` : 全画面を選択します。
    * `RECT ROI` : 正方形や長方形の形状が再現可能です。中心座標と縦横のピクセル数を数値入力で定めることができます。また、蛍光画像上をクリック・スライドすることによっても選択可能です。
    * `Draw Poly` : 任意の形状を多角形で再現可能です。Draw Polyを選択した後に、蛍光画像状にプロットを実施してください。最初のプロットを再度クリックすることで多角形が完成されます。
* `Masking & ARICS Options` :
    * `Fill Mode` : Mean Fill, Zero Fill, Noneから選択可能です。Mean FillはROIとウィンドウの包含領域の輝度平均値で、ROI外の輝度値をPaddingします、Zero PaddingはROI外の輝度値を0で埋めます。NoneはROI外の輝度値をそのまま計算に用います。
    * `ARICS Options` : ARICS Normは、ROIを用いたヒートマップを作成するときに、正規化を行うオプションです。ROI内を1、ROI外を0としたバイナリマスクのACFで、それぞれのACFを割り算することで正規化を実現します。Exclude LagはROI外のピクセルにおけるACFをフィッティング計算から除外するオプションです。

## 2.4.Range & Omit
フィッティング、ヒートマップの作成に用いるACFのプロットを選択します。フィッティング、ヒートマップの作成前に、必ずパラメータの設定をし、Refresh Plotsをクリックしてください。
* `Omit Radius` : 計算したACFのうち、中心からどのラグのACFまでをフィッティングから除外するか選択します。特にラグ1までのACFはショトノイズであるケースが多いため、除外することを推奨します。
* `Fitting Range` : 計算するACFの範囲を選択します。ACFのグラフの点線を選択してスライドさせることも可能です。Auto-Detectを選択すると、ACFが単調減少している領域を自動で検出します。Auto-Detectを選択してヒートマップを作成すると、それぞれのピクセルのACFプロットに対してAuto-Detectが実行されます。

## 2.5.Single Fit
各種パラメータの入力を行なってください。Tripletについてはデフォルトでoffになっています。なお、ヒートマップの作成については、このパラメータを用いて作成が行われます。したがって、ヒートマップの作成前に必ずビームウェストを入力し、fixするようにしてください。
キャリブレーションを行う際には、拡散係数を入力しfixして、ビームウェストの決定を行なってください。

## 2.6. Output
Single Fitの結果に対してACFのグラフの出力が可能です。(PNG形式)

## 2.7. Heatmap
* `パラメータの入力` : 
    * `Win` : window sizeを選択します。各ピクセルに対して、ここで選択したピクセルを一辺とする領域の輝度揺らぎに対して拡散係数の計算を行います。
    * `Step` : 解析ステップを選択します。例えば2を入力すると2つのピクセルに対して1つのピクセルに解析が行われます。全ピクセルに対して解析を行いたい場合は1を入力してください。
* `実行` : Gen Heatmapを選択してください。作成後はすぐに、Save Heatmapを選択してください。再度Gen Heatmapを選択すると、作成後のヒートマップが失われるので注意してください。ヒートマップはpng画像と、各ピクセルの拡散係数を記録したcsvファイルが出力されます。このcsvファイルはこの後の解析で再利用可能です。
* `表示パラメータの選択` : Auto Scaleは取得された拡散係数のうち、下から何パーセントまでを表示させるかを選択します。この選択を外すとMax Dによる閾値設定になります。閾値設定後に再度出力させたい場合は、Regen Viewを選択してください。

### 3. Batch Config Window
フォルダ内の複数のtiffファイルを同一条件で連続解析します。選択したディレクトリの中に、さらに子ディレクトリがある場合、その内部のtiffファイルまで検索します。ROIの選択方法、パラメータの設定方法は全て単一解析と同じです。tiffファイルごとにROIやパラーメータを変えてセットすることができます。

### 4. ROI Analysis Tool 
作成されたヒートマップ（CSV）と元画像を用いて、事後的な解析やROIごとの数値確認を行います。
ROIの選択方法、パラメータの設定方法は全て単一解析と同じです。tiffファイルごとにROIやパラーメータを変えてセットすることができます。

### 5. ROI Analsysis Tool
## 5.1.Loading
Load Reference Imageで解析する蛍光画像、Load Diffusion CSVで作成後のヒートマップ(csv形式)を選択してください。View Modeで画面に描画する対象を選択できます。

## 5.2. 



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
