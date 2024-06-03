from load_data import load_data, show_heat_map
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

train, test = load_data()

# show_heat_map(data_train=train)

predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)', 'Age at enrollment', 'Debtor',
              'Tuition fees up to date', 'Application mode']

kf = KFold(train.shape[0], shuffle=True, random_state=1)

# 定义逻辑回归模型
alg = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.001)

# 存储预测结果
predictions = []

# 存储真实目标值
true_targets = []

# 交叉验证
n_splits = kf.get_n_splits(train)  # 获取分割次数
bar_length = 50

# 进行交叉验证
for a, (train_index, test_index) in enumerate(kf.split(train), 1):
    # 获取训练和测试集的索引
    X_train, X_test = train.iloc[train_index, :-1][predictors], train.iloc[test_index, :-1][predictors]
    y_train, y_test = train.iloc[train_index, -1], train.iloc[test_index, -1]

    # 训练模型
    alg.fit(X_train, y_train)

    # 进行预测
    predictions.append(alg.predict(X_test))
    true_targets.append(y_test)

    # 计算进度条
    progress = a / n_splits
    hashes = '#' * int(progress * bar_length)
    print(f'\rprogress: [{hashes:<{bar_length}}] {int(progress * 100)}% ({a}/{n_splits})', end='')

# 将所有预测结果合并
all_predictions = np.concatenate(predictions, axis=0)
all_true_targets = np.concatenate(true_targets, axis=0)

# 计算准确度
accuracy = accuracy_score(all_true_targets, all_predictions)
print("Accuracy:", accuracy)

# 打印分类报告
print("\nClassification Report:\n", classification_report(all_true_targets, all_predictions))
