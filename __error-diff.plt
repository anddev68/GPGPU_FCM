
set datafile separator " "

# 1:2で，温度と繰り返し回数
set xlabel "Error"
set ylabel "Diff"

#set xrange [:18]

plot "__dump.txt" using 3:4 with p