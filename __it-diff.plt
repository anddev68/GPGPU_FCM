
set datafile separator " "

set xlabel "Iterations"
set ylabel "Diff"

plot "__dump.txt" using 2:4 with p