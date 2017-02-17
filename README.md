# GPGPU for Fuzzy c-means

## Directory Descriptions
* doc/
    * readme.md 結果ファイルの説明
    * *.xlsx 結果出力ファイル
* fcm.h
* fcm.cpp
* bandwidthTest.cu メインファイル
    
## On Progress
* 2017/02/17  
[1次元データを用いた検証結果](output_uik_2d.xlsx)を見るに、<s>プログラムに問題なさそうな気がする。</s>  
(追記)大有りだった。irisのデータを読み込む際に文字列を考慮していなかったため、xkが不正な値となっていた。  
uik→resultsにするところでうまくいっていない場合あり。
* 2016/02/16  
irisのデータを用いた場合、正しい分類データと比較すると50~100の誤分類となる。
どこがおかしいのか不明なので、一度単体テストを行う。
