# error-diff.plt
# 誤分類数と，目的関数FCMの最大増加分(下がって，戻る挙動を示したもので，戻った量が大きいもの)を表したもの

set datafile separator " "

# 1:4で，温度とdiff
set xlabel "Temperature"
set ylabel "Diff"


#set xrange [4:6]
#set yrange [30:50]


#plot "__c1.txt" using 1:4 with p
#replot "__c2.txt" using 1:4 with p
#replot "__c3.txt" using 1:4 with p

plot "__dump.txt" using 1:11 with p