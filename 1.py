from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 定义一个一行的ndarray
my_array = np.array([1, 2, 3, 4, 5])

# 将一行的ndarray转换为一列
my_column = my_array.reshape(-1, 1)

# 输出结果
print(my_column)

# 生成数据集
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=1)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 定义模型
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(type(X))

print(f"均方误差: {mse:.2f}")