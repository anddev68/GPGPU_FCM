set datafile separator ","


set xlabel "Iterations"
set ylabel "Diff-n"

set yrange [:180]
set xrange [1:]

N = 512

plot for [i=0:511:16] sprintf('diff/%d.txt', i) title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) with lp