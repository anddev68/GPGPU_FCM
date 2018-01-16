
set datafile separator " "

set xlabel "Temperature"
set ylabel "HDiff"

set yrange [250:300]

plot "__dump.txt" using 1:7 with p