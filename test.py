from load_data import load_data
import pandas as pd
import joblib
import numpy as np
import gc

# load data
_, test = load_data()
test_ids = test['id']

# factors considered during training setup
predictors = ['Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)',
              'Curricular units 2nd sem (evaluations)',
              'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)',
              'Curricular units 1st sem (evaluations)',
              'Tuition fees up to date']

# load test dataset
test_data = pd.read_csv('./dataset/test.csv')[predictors]

# save the predictions to list
all_predictions = []

# load model and make prediction
model_file = 'model_fold_29.pkl'
model = joblib.load(model_file)
test_predictions = model.predict(test_data).to_numpy()  # ensure return numpy array
all_predictions.append(test_predictions)

del model
gc.collect()

all_predictions = np.concatenate(all_predictions, axis=0)

# create a dataframe
submission_df = pd.DataFrame({
    'ID': test_ids,
    'PredictedTarget': all_predictions
})

# replace the int to str
submission_df['PredictedTarget'] = submission_df['PredictedTarget'].map({1: 'Graduate', 0: 'Dropout', 2: 'Enrolled'})

# save the csv
submission_df.to_csv('submission.csv', index=False)
