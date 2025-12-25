# RICS Analysis App (Python)

Pythonによる Raster Image Correlation Spectroscopy (RICS) 解析用GUIアプリケーションです。
顕微鏡で取得したMulti-tifデータから、拡散係数（Diffusion Coefficient）を算出・マッピングします。

## 主な機能
* **GUI操作:** Tkinterを用いたインタラクティブな操作画面。
* **前処理:** 移動平均（Moving Average）による背景除去。
* **ROI選択:** 画像上のドラッグによる範囲指定、クリックによる中心移動。
* **ACF計算:** 高速フーリエ変換（FFT）を用いた2D自己相関関数の計算。
* **フィッティング:** 3D拡散モデルを用いたカーブフィッティング。
    * パラメータ固定（Fix/Free）機能。
    * Pixel Jump (Omit Center): ショットノイズ除去のため中心点を除外。
    * Fitting Range: フィッティングに用いるラグ範囲の調整。
* **ヒートマップ作成:** Scanning RICSによる拡散係数のマッピング。
    * 局所領域ごとの解析（ROI限定可）。
    * 閾値による外れ値の除外（Clamp High D）。
    * 別ウィンドウでの結果確認。
* **画像保存:** 解析結果グラフやヒートマップのJPEG/PNG保存。

## 環境構築 (Installation)

本アプリケーションは GUIライブラリとして **Tkinter** を使用しています。
特に macOS では、システム標準の Python (`/usr/bin/python3`) を使用すると、Tcl/Tkライブラリのバージョン不整合により**画面が真っ白になる問題**が発生します。

そのため、必ず以下の手順に従って **Homebrew版 Python** と **仮想環境 (venv)** を使用してください。

### 前提条件
* Python 3.10 以上
* Homebrew (macOSの場合)

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
    プロジェクト専用のPython環境を作成します。
    ```bash
    python3 -m venv venv
    ```

4.  **仮想環境の有効化 (Activate)**
    ```bash
    source venv/bin/activate
    ```
    ※ ターミナルの先頭に `(venv)` と表示されればOKです。

5.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```
    ※ `requirements.txt` がない場合は以下を実行:
    `pip install numpy scipy matplotlib tifffile`

### Windows でのセットアップ手順

1.  Python公式サイトからPythonをインストール（インストール時に "tcl/tk and IDLE" にチェックが入っていることを確認）。
2.  PowerShellなどでプロジェクトフォルダに移動。
3.  仮想環境作成: `python -m venv venv`
4.  有効化: `.\venv\Scripts\Activate`
5.  ライブラリインストール: `pip install -r requirements.txt`

## 使い方 (Usage)

### 1. アプリケーションの起動
必ず仮想環境を有効化した状態で実行してください。

```bash
# Mac (venv有効化済みの場合)
python gui_app.py
```
### 2. 解析パラメータの設定 (`config.py`)
実験条件に合わせて、`config.py` 内の以下の値を編集してください。**ここが合っていないと拡散係数が正しい桁になりません。**

* `PIXEL_SIZE`: 1ピクセルのサイズ (um)
* `PIXEL_DWELL_TIME`: 1ピクセルの滞留時間 (s)
* `LINE_TIME`: 1ラインの走査時間 (s)
* `W0`, `WZ`: レーザーのビームウェスト径 (um)

### 3. GUI操作フロー
1.  **Load Data:** TIFファイル（Multi-stack）を読み込みます。
2.  **ROI & Preprocessing:**
    * 画像プレビュー上でドラッグしてROIを選択します。
    * クリックでROIの中心を移動できます。
    * `Mov.Avg` で背景除去の強度（フレーム数）を調整し "Update" します。
3.  **Range & Omit:**
    * `Omit Radius` を `1` 以上に設定すると、ノイズ源となる中心点（ラグ0）を除外します。
    * グラフ上の赤い点線をドラッグして、フィッティング範囲を調整します。
4.  **Single Point Fitting:**
    * "Run Fitting" で現在選択中のROIに対してフィッティングを行います。
5.  **Heatmap Analysis:**
    * `Win Size` (局所領域サイズ) と `Step` (計算間隔) を設定します。
    * "Generate Heatmap" で計算を開始します（別スレッドで実行）。
    * 必要に応じて "Clamp High D" で閾値を設定し、外れ値を除去して表示します。

## トラブルシューティング

### Q. アプリを起動しても画面が真っ白で何も表示されない (macOS)
**A.** macOS標準のPythonを使用している可能性があります。
ターミナルで `which python` を実行し、`/usr/bin/python` と表示される場合はNGです。
「環境構築」のセクションに従い、HomebrewでインストールしたPython (`/opt/homebrew/bin/python3` 等) を使い、仮想環境 (`venv`) 上で実行してください。

### Q. ヒートマップ作成が遅い
**A.** `Step` の値を大きくしてください（例: 4や8）。`Step=1` にすると全画素について計算するため、非常に時間がかかります。

### Q. フィッティングがうまくいかない
**A.**
1.  `config.py` の `PIXEL_DWELL_TIME` や `LINE_TIME` が正しいか確認してください。
2.  `Omit Radius` を 1〜2 程度に設定してショットノイズを除去してください。
3.  `Mov.Avg`（移動平均）を適切にかけて、不動成分（細胞膜などの構造）を除去してください。

## Author
Kensuke Takahama
