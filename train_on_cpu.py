from load_data import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

train, test = load_data()

predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)', 'Age at enrollment', 'Debtor',
              'Tuition fees up to date', 'Application mode']

kf = KFold(train.shape[0], shuffle=True, random_state=1)

alg = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.001)

predictions = []

true_targets = []

n_splits = kf.get_n_splits(train)
bar_length = 50

for a, (train_index, test_index) in enumerate(kf.split(train), 1):
    X_train, X_test = train.iloc[train_index, :-1][predictors], train.iloc[test_index, :-1][predictors]
    y_train, y_test = train.iloc[train_index, -1], train.iloc[test_index, -1]

    alg.fit(X_train, y_train)

    predictions.append(alg.predict(X_test))
    true_targets.append(y_test)

    progress = a / n_splits
    hashes = '#' * int(progress * bar_length)
    print(f'\rprogress: [{hashes:<{bar_length}}] {int(progress * 100)}% ({a}/{n_splits})', end='')

all_predictions = np.concatenate(predictions, axis=0)
all_true_targets = np.concatenate(true_targets, axis=0)

accuracy = accuracy_score(all_true_targets, all_predictions)
print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(all_true_targets, all_predictions))
