import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

### 像素值+SVM

# 加载FashionMNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 将图像数据展平
X_train_flattened = np.reshape(X_train, (-1, 28 * 28))
X_test_flattened = np.reshape(X_test, (-1, 28 * 28))

# 标准化特征
scaler = StandardScaler()
X_train_flattened = scaler.fit_transform(X_train_flattened)
X_test_flattened = scaler.transform(X_test_flattened)

# 训练SVM模型
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X_train_flattened, y_train)

# 测试模型
predictions = clf.predict(X_test_flattened)
print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))