# error-diff.plt
# �땪�ސ��ƁC�ړI�֐�FCM�̍ő呝����(�������āC�߂鋓�������������̂ŁC�߂����ʂ��傫������)��\��������

set datafile separator " "

# 1:4�ŁC���x��diff
set xlabel "Temperature"
set ylabel "Diff"


#set xrange [4:6]
#set yrange [30:50]


#plot "__c1.txt" using 1:4 with p
#replot "__c2.txt" using 1:4 with p
#replot "__c3.txt" using 1:4 with p

plot "__dump.txt" using 1:11 with p