reset
set nokey
set xlabel "Iterations of Clustering "
set ylabel "The number of error"
plot for [i=0:31] sprintf("err%d.txt",i) with lp title "a"