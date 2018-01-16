set datafile separator " "
#set yrange [] reverse 
set xlabel "Iterations"
set ylabel "Diff-n"
set xrange [1:30]
plot for [i=0:511:32] sprintf('diff/%d.txt', i)  using 0:2 with lp
