set datafile separator ","
set xlabel "Iterations"
set ylabel "J"
#set xrange [1:]
#set yrange [60:100]
plot for [i=1:256:8] "__jfcm.txt" using i with p title sprintf("%d", i)
