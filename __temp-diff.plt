# error-diff.plt
# 誤分類数と，目的関数FCMの最大増加分(下がって，戻る挙動を示したもので，戻った量が大きいもの)を表したもの

set datafile separator " "
set xrange [0:150]

# 1:4で，温度とdiff
set xlabel "Temperature"
set ylabel "Diff"


plot "__dump.txt" using 1:4 with p