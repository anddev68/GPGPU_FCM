# error-diff.plt
# �땪�ސ��ƁC�ړI�֐�FCM�̍ő呝����(�������āC�߂鋓�������������̂ŁC�߂����ʂ��傫������)��\��������

set datafile separator " "
set xrange [0:150]

# 1:4�ŁC���x��diff
set xlabel "Temperature"
set ylabel "Diff"


plot "__dump.txt" using 1:4 with p