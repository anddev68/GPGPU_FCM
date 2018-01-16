
set datafile separator " "

set xlabel "Temperature"
set ylabel "HDiff"

plot "__dump.txt" using 1:6 with p