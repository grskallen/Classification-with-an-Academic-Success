from load_data import load_data
import pandas as pd
import joblib
import numpy as np
import gc

# 加载数据
_, test = load_data()
test_ids = test['id']

# 确保测试数据包含与训练数据相同的特征列
predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)',
              'Tuition fees up to date']

# 加载测试数据集
test_data = pd.read_csv('./dataset/test.csv')[predictors]

# 预测结果存储列表
all_predictions = []

# 加载模型并进行预测
model_file = 'model_fold_29.pkl'
model = joblib.load(model_file)
test_predictions = model.predict(test_data).to_numpy()  # 确保返回 numpy 数组
all_predictions.append(test_predictions)

# 显式删除模型对象
del model
gc.collect()

# 将所有折叠的预测结果合并为一个列表
all_predictions = np.concatenate(all_predictions, axis=0)

# 创建一个新的DataFrame，包含测试集ID和预测结果
submission_df = pd.DataFrame({
    'ID': test_ids,
    'PredictedTarget': all_predictions
})

# 替换DataFrame中的值
submission_df['PredictedTarget'] = submission_df['PredictedTarget'].map({1: 'Graduate', 0: 'Dropout', 2: 'Enrolled'})

# 保存到CSV文件
submission_df.to_csv('submission.csv', index=False)
