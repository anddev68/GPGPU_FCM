set datafile separator " "

# 1:2‚ÅC‰·“x‚ÆŒJ‚è•Ô‚µ‰ñ”
set xlabel "Temperature"
set ylabel "Iterations"
#plot "__c1.txt" using 1:2 with p
#replot "__c2.txt" using 1:2 with p
plot "__dump.txt" using 1:2 with p