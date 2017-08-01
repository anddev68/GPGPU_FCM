set datafile separator " "
set yrange [] reverse 
plot for [i=1:8] "entropy.txt" using i with lp title sprintf("%d", i)