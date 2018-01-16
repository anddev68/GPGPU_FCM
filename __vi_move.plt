N = 1024
div = 128
path = "vi72"

set xrange [0:10]
set yrange [0:10]

set key right bottom
set key font "Arial,6pt"    # 凡例のフォントの変更

# 温度を求める関数
temp(i) = 200 ** ((i + 1.0 - N / 2.0) / (N / 2.0))


###############################################################################

plot for [i=1:N:div] sprintf("%s/Thigh=%.5f.txt", path, temp(i)) using 1:2 with lp  lc i/div pt i/div+1 title sprintf("Thigh=%.6f", temp(i))
replot for [i=1:N:div] sprintf("%s/Thigh=%.5f.txt", path, temp(i)) using 3:4 with lp  lc i/div pt i/div+1 title sprintf("Thigh=%.6f", temp(i))
replot for [i=1:N:div] sprintf("%s/Thigh=%.5f.txt", path, temp(i)) using 5:6 with lp  lc i/div pt i/div+1 title sprintf("Thigh=%.6f", temp(i))
