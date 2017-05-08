# GPGPU for Fuzzy c-means
汎用性を高めたFCM実行プログラム

## 設定ファイルの設定方法
+ type = [pca|]
	現在は未実装で，このパラメータは無効です．
	+ pca: parallel cluster analysis implementation
+ dataset = [iris|random]
	irisのデータセットまたはrandomでデータセットを生成する．
+ random
	データセットの生成方法をrandomにした場合のみ有効なパラメータです．
	その他の場合は無視されます．
	+ N データ数
	+ P 次元数
	+ C クラスタ数
	+ min ランダムで割り振られる最小値
	+ max ランダムで割り振られる最大値


	例
	```
		{
		"parallel": true,
		"dataset": "iris",
		"random": {
			"N": 150,
			"P": 4,
			"C": 3,
			"min": 0.0,
			"max": 1.0,
		}
	}
	```
	




## Abstract
* 目的  
q値クラスタリングにおける(1)計算速度の向上 (2)解の精度向上を
CUDAで実現する。
* 研究成果  
行列演算を並列化することにより、計算速度の向上が見られた。  
詳細は...
* 今後の研究方針  
通常FCMを複数のスレッドで初期温度を変えて並列に実行できることを確認した。
展開元の状態を最良解に変更し、並列実行する。

並列幅優先探索


    

## Directory Descriptions
* doc/
    * readme.md 結果ファイルの説明
    * *.xlsx 結果出力ファイル
* fcm.h
* fcm.cpp
* bandwidthTest.cu メインファイル
    
## On Progress

* 2017/02/21  
目的関数を評価値とするのは、途中のクラスタリングが必ずしも正しいとはいえないので
最適ではない。クラスタリングが失敗したものをはずす方向で進める。
* 2017/02/20  
目的関数と誤分類数の相関調べたらうまくでたので、これを評価関数としてもよさそう。  
ここだけはハードクラスタリングにしてもよいかもしれない。
* 2017/02/17  
[1次元データを用いた検証結果](output_uik_2d.xlsx)を見るに、<s>プログラムに問題なさそうな気がする。</s>  
(追記)大有りだった。irisのデータを読み込む際に文字列を考慮していなかったため、xkが不正な値となっていた。  
uik→resultsにするところでうまくいっていない場合あり。
* 2016/02/16  
irisのデータを用いた場合、正しい分類データと比較すると50~100の誤分類となる。
どこがおかしいのか不明なので、一度単体テストを行う。
