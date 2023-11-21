# 导入必要的库
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载加利福尼亚州房价数据集
housing_22 = fetch_california_housing()
X_22, y_22 = housing_22.data, housing_22.target

# 划分数据集为训练集和测试集
X_train_22, X_test_22, y_train_22, y_test_22 = train_test_split(
    X_22, y_22, test_size=0.2, random_state=42)

# 初始化线性回归模型
model_22 = LinearRegression()

# 训练模型
model_22.fit(X_train_22, y_train_22)

# 在测试集上进行预测
y_pred_22 = model_22.predict(X_test_22)

# 评估模型性能
mse_22 = mean_squared_error(y_test_22, y_pred_22)
print(f'Mean Squared Error on Test Set: {mse_22}')

# 绘制真实值与预测值的对比图
plt.scatter(y_test_22, y_pred_22)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()
