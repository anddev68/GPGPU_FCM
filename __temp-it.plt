set datafile separator " "

# 1:2�ŁC���x�ƌJ��Ԃ���
set xlabel "Temperature"
set ylabel "Iterations"
#plot "__c1.txt" using 1:2 with p
#replot "__c2.txt" using 1:2 with p
plot "__dump.txt" using 1:2 with p