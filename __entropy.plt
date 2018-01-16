set datafile separator ","
#set yrange [] reverse 
set xlabel "Iterations"
set ylabel "|H|"
set xrange [1:]
plot for [i=1:8] "__entropy.txt" using i with lp title sprintf("%d", i)