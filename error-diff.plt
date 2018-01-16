# error-diff.plt
# 誤分類数と，目的関数FCMの最大増加分(下がって，戻る挙動を示したもので，戻った量が大きいもの)を表したもの

set datafile separator " "

# 1:2で，温度と繰り返し回数
#set xlabel "Temperature"
#set ylabel "Iterations"
#set yrange [:18]
#plot "__dump.txt" using 1:2 with p

# 2:4で，繰り返し回数とdiffの相関を示す
#set xlabel "Iterations"
#set ylabel "Diff"
#set xrange [1:20]
#plot "__dump.txt" using 2:4 with p

# 3:4で，誤分類数と最大増減分の相関を示す
#set xlabel "Error"
#set ylabel "Diff"
#plot "__dump.txt" using 3:4 with p

# 1:4で，温度とdiff
set xlabel "Temperature"
set ylabel "Diff"
plot "__dump.txt" using 1:4 with p

# 1:5で，温度とave
#set xlabel "Temperature"
#set ylabel "Average"
#plot "__dump.txt" using 1:5 with p

# 1:5で，温度とsub
#set xlabel "Temperate"
#set ylabel "Sub"
#plot "__dump.txt" using 1:5 with p

# 2:5で，繰り返し回数とsub
#set xlabel "Iterations"
#set ylabel "Sub"
#plot "__dump.txt" using 2:5 with p