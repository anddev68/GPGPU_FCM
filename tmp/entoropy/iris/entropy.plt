set datafile separator " "
set yrange [] reverse 
plot for [i=1:8] "3.txt" using i with lp title sprintf("%d", i)