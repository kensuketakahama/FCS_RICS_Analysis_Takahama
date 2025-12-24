# FCS_RICS_Analysis_Takahama

FCS・RICS 解析用スクリプト

# Requirements

mac での実行は確認しているが、windows で対応可能かは不明。  
両方の OS で動作可能なようには実装している。  
python3.8.2 以降で実行可能

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
└── RWADME.md
```

<pre>
.
├── FCS
└── 
├── xxx
└── README.md
</pre>

# RICS

RICS は現在 GUI でサポートしている。  
実行する前に各種パラメータを config 内に入力する。  
ディレクトリ内で

`python3 gui_app.py`

で実行可能。

# FCS

main.py でサポートしている。  
現在は fitting する範囲の選択ができなくなっている。これについては今後 gui として実装する予定である。  
`python3 main.py`
で実行可能。  
実行する前に main.py 内の各種パラメータ、fix するか否か、データパスを入力してから実行する。
