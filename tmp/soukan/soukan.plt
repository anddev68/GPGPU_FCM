set nokey
set xlabel "�땪�ސ�(��)"
set xrange [0:49]
set yrange [0:200]
set ylabel "�ړI�֐�Jtsallis'�̒l"
plot for [i=0:64] sprintf("soukan%d.txt",i)