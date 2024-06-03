import os
import ctypes

# 直接在 Python 脚本中设置 LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib'
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

try:
    ctypes.CDLL('/usr/local/cuda-12.2/lib64/libcublas.so.12')
    print("libcublas.so.12 loaded successfully")
    ctypes.CDLL('/usr/local/cuda-12.2/lib64/libnvrtc.so.12')
    print("libnvrtc.so.12 loaded successfully")
    ctypes.CDLL('/usr/local/cuda-12.2/lib64/libnvrtc-builtins.so.12.2')
    print("libnvrtc-builtins.so.12.2 loaded successfully")
except OSError as e:
    print("Error load")

from load_data import load_data, show_heat_map
from cuml.linear_model.logistic_regression import LogisticRegression as cuLogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# 加载数据
train, test = load_data()

# show_heat_map(data_train=train)

predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)', 'Age at enrollment', 'Debtor',
              'Tuition fees up to date', 'Application mode']

kf = KFold(n_splits=5, shuffle=True, random_state=1)

# 定义CuML的逻辑回归模型
alg = cuLogisticRegression(max_iter=1000, tol=0.001, fit_intercept=True)

# 存储预测结果
predictions = []

# 存储真实目标值
true_targets = []

# 存储每个折叠的准确度
train_accuracies = []
test_accuracies = []

# 交叉验证
n_splits = kf.get_n_splits(train)  # 获取分割次数
bar_length = 50

# 进行交叉验证
for a, (train_index, test_index) in enumerate(kf.split(train), 1):
    # 获取训练和测试集的索引
    X_train, X_test = train.iloc[train_index][predictors], train.iloc[test_index][predictors]
    y_train, y_test = train.iloc[train_index]['Target'], train.iloc[test_index]['Target']

    # 训练模型
    alg.fit(X_train, y_train)

    # 保存模型
    joblib.dump(alg, f'model_fold_{a}.pkl')

    # 进行预测
    train_pred = alg.predict(X_train).values
    test_pred = alg.predict(X_test).values

    # 存储预测结果
    predictions.append(test_pred)
    true_targets.append(y_test.to_numpy())

    # 计算并存储准确度
    train_accuracy = accuracy_score(y_train.to_numpy(), train_pred)
    test_accuracy = accuracy_score(y_test.to_numpy(), test_pred)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 计算进度条
    progress = a / n_splits
    hashes = '#' * int(progress * bar_length)
    print(f'\rprogress: [{hashes:<{bar_length}}] {int(progress * 100)}% ({a}/{n_splits})', end='')

# 将所有预测结果合并
all_predictions = np.concatenate(predictions, axis=0)
all_true_targets = np.concatenate(true_targets, axis=0)

# 计算整体准确度
accuracy = accuracy_score(all_true_targets, all_predictions)
print("\n\nOverall Accuracy:", accuracy)

# 打印分类报告
print("\nClassification Report:\n", classification_report(all_true_targets, all_predictions))

# 可视化训练过程
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_splits + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, n_splits + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.show()

