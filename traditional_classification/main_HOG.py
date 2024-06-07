import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

### HOG+SVM

# 加载FashionMNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# HOG参数设置
win_size = (28, 28)
block_size = (14, 14)
block_stride = (7, 7)
cell_size = (7, 7)
nbins = 9

# 提取HOG特征
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
descriptors_train = []
for img in X_train:
    descriptors_train.append(hog.compute(img).flatten())
descriptors_train = np.array(descriptors_train)

descriptors_test = []
for img in X_test:
    descriptors_test.append(hog.compute(img).flatten())
descriptors_test = np.array(descriptors_test)

# 标准化特征
scaler = StandardScaler()
descriptors_train = scaler.fit_transform(descriptors_train)
descriptors_test = scaler.transform(descriptors_test)

# 训练SVM模型
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(descriptors_train, y_train)

# 测试模型
y_pred = clf.predict(descriptors_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 计算混淆矩阵  
cm = confusion_matrix(y_test, y_pred,normalize='true') 
  
# 输出混淆矩阵  
print("Confusion Matrix:")
print(cm)  # 打印混淆矩阵

# 设置类别标签  
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # 获取数据集中的类别名称  
class_names = [i for i in range(10)]

# 设置全局字体大小为14
plt.rcParams.update({'font.size': 14})
  
# 绘制混淆矩阵热图  
plt.figure(figsize=(10, 7)) 
sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, cmap='Blues', fmt=".2g")  
  
plt.xlabel('Predicted') 
plt.ylabel('True')
plt.show()

