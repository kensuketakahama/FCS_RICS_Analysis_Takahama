# Directory

```
.
└── FCS
│   ├── Data
│   ├── src
│   ├── requirements.txt
│   ├── check_raw_data.py
│   ├── test.py
│   └── main.py
├── RICS
│   ├── Data
│   ├── src
│   ├── requirements.txt
│   ├── config.py
│   ├── gui_app.py
│   └── main.py
├── .gitignore
└── RWADME.md
```

# 環境構築 (Installation)

本アプリケーションは GUIライブラリとして **Tkinter** を使用しています。
特に macOS では、システム標準の Python (`/usr/bin/python3`) を使用すると、Tcl/Tkライブラリのバージョン不整合により**画面が真っ白になる問題**が発生します。

そのため、必ず以下の手順に従って **Homebrew版 Python** と **仮想環境 (venv)** を使用してください。

## 前提条件
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
    各ディレクトリ内で以下のコマンドを実行してください。
    ```bash
    pip install -r requirements.txt
    ```

### Windows でのセットアップ手順

1.  Python公式サイトからPythonをインストール（インストール時に "tcl/tk and IDLE" にチェックが入っていることを確認）。
2.  PowerShellなどでプロジェクトフォルダに移動。
3.  仮想環境作成: `python -m venv venv`
4.  有効化: `.\venv\Scripts\Activate`
5.  ライブラリインストール: `pip install -r requirements.txt`