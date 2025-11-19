import numpy
import matplotlib.pyplot

a = numpy.zeros([3, 2])
a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 9
a[2, 1] = 12
print(a)
matplotlib.pyplot.imshow(a, interpolation="nearest")

matplotlib.pyplot.colorbar()  # 添加颜色条便于观察数值
matplotlib.pyplot.show()  # 关键：显示图形窗口