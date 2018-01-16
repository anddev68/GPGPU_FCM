
set datafile separator " "

# 1:2‚ÅC‰·“x‚ÆŒJ‚è•Ô‚µ‰ñ”
set xlabel "Error"
set ylabel "Diff"

#set xrange [:18]

plot "__dump.txt" using 3:4 with p