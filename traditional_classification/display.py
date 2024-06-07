import tensorflow as tf
import matplotlib.pyplot as plt

### 测试，用于显示图片

# 加载FashionMNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 显示第一张图片
index = 1
image = X_train[index]

# 图片预处理
image = image / 255.0

# 使用matplotlib显示图片
plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray')
plt.title("Class: {}".format(y_train[index]))
plt.axis('off')  # 不显示坐标轴
plt.show()