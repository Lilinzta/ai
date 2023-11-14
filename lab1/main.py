# Source: https://tjxlab.gitbooks.io/bigdata/content/da-shu-ju-fen-xi-yu-wa-jue-shi-yan/shi-yanyi-ff1a-knn-fen-lei-shi-yan.html

# numpy 库用于科学计算，matplotlib 库用于绘图
# Scikit-learn 库中的 neighbors 模块和 datasets 模块分别包含KNN分类器和一些常用的数据集
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# 定义邻居个数，设定KNN算法中邻居个数为15
n_neighbors_22 = 15

# 加载iris数据集，使用 Scikit-learn 中的 load_iris() 函数加载鸢尾花数据集
iris_22 = datasets.load_iris()

# 选择特征，使用了鸢尾花数据集中的前两个特征
X_22 = iris_22.data[:, :2]
y_22 = iris_22.target

# 设置步长和创建颜色映射，h 是在可视化决策边界时使用的步长
# cmap_light 和 cmap_bold 是两个颜色映射，用于可视化轻量级背景和加粗版背景颜色
h_22 = .02  # step size in the mesh
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 循环进行KNN分类和可视化
for weights in ['uniform', 'distance']:
    # 使用循环，分别使用两种权重（uniform和distance）创建KNN分类器实例，并对数据进行拟合。
    clf = neighbors.KNeighborsClassifier(n_neighbors_22, weights=weights)
    clf.fit(X_22, y_22)

    # 为了绘制决策边界，定义一个网格，并在网格上计算模型的预测值
    x_min, x_max = X_22[:, 0].min() - 1, X_22[:, 0].max() + 1
    y_min, y_max = X_22[:, 1].min() - 1, X_22[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_22),
                         np.arange(y_min, y_max, h_22))

    # 使用模型对网格中的点进行预测
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将预测结果以颜色的形式绘制在网格上
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 将训练数据集上的点用散点图表示，并显示图表的标题
    plt.scatter(X_22[:, 0], X_22[:, 1], c=y_22, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors_22, weights))

# 最后，通过 plt.show() 将图形显示出来
plt.show()
