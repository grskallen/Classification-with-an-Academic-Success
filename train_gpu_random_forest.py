import ctypes
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from cuml.ensemble import RandomForestClassifier as cumlRF
import joblib
import gc
from load_data import load_data, show_heat_map
train, test = load_data()

predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)', 'Age at enrollment', 'Debtor',
              'Tuition fees up to date', 'Application mode']

kf = KFold(n_splits=5, shuffle=True, random_state=1)

predictions = []
true_targets = []
train_accuracies = []
test_accuracies = []

n_splits = kf.get_n_splits(train)
bar_length = 50
max_depth = 10
n_estimators = 25

for a, (train_index, test_index) in enumerate(kf.split(train), 1):
    X_train, X_test = train.iloc[train_index][predictors], train.iloc[test_index][predictors]
    y_train, y_test = train.iloc[train_index]['Target'], train.iloc[test_index]['Target']

    model = cumlRF(max_depth=max_depth, n_estimators=n_estimators, random_state=0)

    model.fit(X_train, y_train)

    joblib.dump(model, f'model_fold_{a}.pkl')

    train_pred = model.predict(X_train).to_numpy()
    test_pred = model.predict(X_test).to_numpy()

    predictions.append(test_pred)
    true_targets.append(y_test.to_numpy())

    train_accuracy = accuracy_score(y_train.to_numpy(), train_pred)
    test_accuracy = accuracy_score(y_test.to_numpy(), test_pred)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    del model
    gc.collect()

    progress = a / n_splits
    hashes = '#' * int(progress * bar_length)
    print(f'\rprogress: [{hashes:<{bar_length}}] {int(progress * 100)}% ({a}/{n_splits})', end='')

all_predictions = np.concatenate(predictions, axis=0)
all_true_targets = np.concatenate(true_targets, axis=0)

submission_df = pd.DataFrame({
    'PredictedTarget': all_predictions
})
submission_df['PredictedTarget'] = submission_df['PredictedTarget'].map({1: 'Graduate', 0: 'Dropout', 2: 'Enrolled'})
submission_df.to_csv('train_result.csv', index=False)

accuracy = accuracy_score(all_true_targets, all_predictions)
print("\n\nOverall Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(all_true_targets, all_predictions))

plt.figure(figsize=(12, 6))
plt.plot(range(1, n_splits + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, n_splits + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.show()
