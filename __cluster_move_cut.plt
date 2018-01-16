N = 512
path = "vi11"

#set xrange [6:10]
#set yrange [6:10]
#set nokey

set title "cluster moving"
plot for [i=1:N:126] sprintf("< head -n 3 %s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 1:2 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))

replot for [i=1:N:126] sprintf("< haed -n 3 %s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 3:4 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))

replot for [i=1:N:126] sprintf("< head -n 3 %s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 5:6 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))


#replot "__xk.txt" using 1:2 with p