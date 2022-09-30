import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

x = range(1, 10)
y = [1, 3, 9, 12, 15, 23, 34, 31, 23]
plt.plot(x,y)  # 传入x轴和y轴数据，通过plot绘制折线图
plt.show()

