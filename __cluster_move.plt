N = 512
path = "vi64"

set xrange [0:]
set yrange [0:]
set nokey

# �w�肵�����x���łȂ���΃}�C�i�X���o�͂���֐�
LOW = 0.0
HIGH = 100000
div = 16
myfunc(x, tmp) = LOW<tmp&&tmp<HIGH? x: -100

# ���x�����߂�֐�
temp(i) = 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))


## �w�肵�����x���ŁC����n��ڂ̃N���X�^�����O�񐔂Ȃ�\������
myfunc2(x, i) = i==3&&LOW<temp(i)&&temp(i)<HIGH? x: -100




###############################################################################

## �w�肵�����x��Ԃ݂̂̃N���X�^���S�̈ړ���\������

#set title sprintf("%.4f<Thigh<%.4f", LOW, HIGH)

#plot for [i=1:512:1] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($5, i)):(myfunc($6, i)) with p title sprintf("Thigh=%.4f", temp(i))

#replot for [i=1:512:1] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($1, i)):(myfunc($2, i)) with p title sprintf("Thigh=%.4f", temp(i))

#replot for [i=1:512:1] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($3, i)):(myfunc($4, i)) with p title sprintf("Thigh=%.4f", temp(i))

###############################################################################



###############################################################################

## �m�[�}���o�[�W����

set title sprintf("%.4f<Thigh<%.4f", LOW, HIGH)
plot for [i=1:512:div] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($5, temp(i))):(myfunc($6, temp(i))) with lp title sprintf("Thigh=%.4f", temp(i))

replot for [i=1:512:div] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($1, temp(i))):(myfunc($2, temp(i))) with lp title sprintf("Thigh=%.4f", temp(i))

replot for [i=1:512:div] sprintf("%s/Thigh=%.4f.txt", path, temp(i)) using (myfunc($3, temp(i))):(myfunc($4, temp(i))) with lp title sprintf("Thigh=%.4f", temp(i))

###############################################################################

#set title "cluster moving"
#plot for [i=1:N:div] sprintf("%s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 1:2 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))

#replot for [i=1:N:div] sprintf("%s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 3:4 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))

#replot for [i=1:N:div] sprintf("%s/Thigh=%.4f.txt", path, 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0))) using 5:6 with lp title sprintf("Thigh=%.4f", 25 ** ((i + 1.0 - N / 2.0) / (N / 2.0)))


#replot "__xk.txt" using 1:2 with p