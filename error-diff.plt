# error-diff.plt
# �땪�ސ��ƁC�ړI�֐�FCM�̍ő呝����(�������āC�߂鋓�������������̂ŁC�߂����ʂ��傫������)��\��������

set datafile separator " "

# 1:2�ŁC���x�ƌJ��Ԃ���
#set xlabel "Temperature"
#set ylabel "Iterations"
#set yrange [:18]
#plot "__dump.txt" using 1:2 with p

# 2:4�ŁC�J��Ԃ��񐔂�diff�̑��ւ�����
#set xlabel "Iterations"
#set ylabel "Diff"
#set xrange [1:20]
#plot "__dump.txt" using 2:4 with p

# 3:4�ŁC�땪�ސ��ƍő呝�����̑��ւ�����
#set xlabel "Error"
#set ylabel "Diff"
#plot "__dump.txt" using 3:4 with p

# 1:4�ŁC���x��diff
set xlabel "Temperature"
set ylabel "Diff"
plot "__dump.txt" using 1:4 with p

# 1:5�ŁC���x��ave
#set xlabel "Temperature"
#set ylabel "Average"
#plot "__dump.txt" using 1:5 with p

# 1:5�ŁC���x��sub
#set xlabel "Temperate"
#set ylabel "Sub"
#plot "__dump.txt" using 1:5 with p

# 2:5�ŁC�J��Ԃ��񐔂�sub
#set xlabel "Iterations"
#set ylabel "Sub"
#plot "__dump.txt" using 2:5 with p