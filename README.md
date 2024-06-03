# Academic Success Classification
This repository contains the code for the "Classification-with-an-Academic-Success" competition on Kaggle.
## Dataset
To download the dataset, execute the following command:
```bash
kaggle competitions download -c playground-series-s4e6
```
Alternatively, you can download the dataset from the Kaggle competition page: [Classification with an Academic Success Dataset](https://www.kaggle.com/competitions/playground-series-s4e6/data).
## Getting Started
Ensure your directory structure is as follows:
```
Academic-Success-Classification/
|
+-- dataset/
|   |
|   +-- train.csv
|   +-- test.csv
|   +-- sample_submission.csv
+-- load_data.py
+-- test.py
+-- train_on_cpu.py
+-- train_gpu_logistic_regression.py
+-- train_gpu_random_forest.py
```
Once the dataset is in place, you're ready to run the code and start exploring the data.
