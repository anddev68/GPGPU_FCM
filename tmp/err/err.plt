set xlabel "Iterations of clustering (0-indexed)"
set ylabel "The number of error"
set yrange [0:]
plot for [i=0:31] sprintf("err%d.txt",i) with lp title sprintf("Thigh=%1.2f", 20.0**((i + 1.0 - 16.0) / 16.0))


