# Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

# 导入必要的库
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# 使用load_digits()加载数字数据集，并提取特征和标签。确定了样本数、特征数和唯一标签的数量。
data_22, labels_22 = load_digits(return_X_y=True)
(n_samples, n_features), n_digits_22 = data_22.shape, np.unique(labels_22).size

# 使用PCA将数据的维度减少为2个，以便进行可视化。
reduced_data_22 = PCA(n_components=2).fit_transform(data_22)

# 在减少的数据上应用K均值，并创建一个2D散点图以可视化聚类。
kmeans_22 = KMeans(init="k-means++", n_clusters=n_digits_22, n_init=4)
kmeans_22.fit(reduced_data_22)

# 定义一个网格中的点的步长
h = 0.02

x_min, x_max = reduced_data_22[:, 0].min() - 1, reduced_data_22[:, 0].max() + 1
y_min, y_max = reduced_data_22[:, 1].min() - 1, reduced_data_22[:, 1].max() + 1

# 生成一个网格，以便在图中可视化聚类的结果
xx_22, yy_22 = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

Z = kmeans_22.predict(np.c_[xx_22.ravel(), yy_22.ravel()])

Z = Z.reshape(xx_22.shape)
plt.figure(1)
plt.clf()

# 在2D空间上绘制聚类结果，用白色十字标记聚类中心。
plt.imshow(Z,
           interpolation="nearest",
           extent=(xx_22.min(),
                   xx_22.max(),
                   yy_22.min(),
                   yy_22.max()),
           cmap=plt.cm.Paired,
           aspect="auto",
           origin="lower",)

plt.plot(reduced_data_22[:, 0], reduced_data_22[:, 1], "k.", markersize=2)
centroids = kmeans_22.cluster_centers_
plt.scatter(centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# 显示绘图
plt.show()
