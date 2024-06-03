import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data():
    data_root = './dataset'
    train = pd.read_csv(data_root + '/train.csv')
    test = pd.read_csv(data_root + '/test.csv')
    train['Target'] = train['Target'].replace({'Graduate': 1, 'Dropout': 0, 'Enrolled': 2})
    # train_df = pd.DataFrame(train)
    # test_df = pd.DataFrame(test)
    # train = train_df.values
    # test = test_df.values
    return train, test


# generate heat map of each element
def show_heat_map(data_train, show_image=True):
    corr_matrix = data_train.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Correlation Matrix')
    if show_image is True:
        plt.show()

# train, test = load_data()
# show_heat_map(data_train=train)