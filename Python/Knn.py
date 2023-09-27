import matplotlib.pylab as plt
import pandas as pd
import sklearn.metrics
import sklearn.neighbors


def plot_dataset(train_data, test_data, predict_data, title):
    fig, ax = plt.subplots()

    # train data
    subset = train_data.loc[train_data['Ownership'] == 'Owner']
    ax.scatter(subset.Income, subset.Lot_Size,
               marker='o', label='Owner', color='C1')
    subset = train_data.loc[train_data['Ownership'] == 'Nonowner']
    ax.scatter(subset.Income, subset.Lot_Size,
               marker='D', label='Nonowner', color='C0')
    for _, row in train_data.iterrows():
        ax.annotate(row.Number, (row.Income, row.Lot_Size))

    # test data
    subset = test_data.loc[test_data['Ownership'] == 'Owner']
    ax.scatter(subset.Income, subset.Lot_Size,
               marker='o', label=None, color='C1', facecolors='none')
    subset = test_data.loc[test_data['Ownership'] == 'Nonowner']
    ax.scatter(subset.Income, subset.Lot_Size,
               marker='D', label=None, color='C0', facecolors='none')
    for _, row in test_data.iterrows():
        ax.annotate(row.Number, (row.Income, row.Lot_Size))

    # predict
    ax.scatter(predict_data.Income, predict_data.Lot_Size,
               marker='*', label='Predict', color='black', s=150)

    plt.xlabel('Income')
    plt.ylabel('Lot_Size')
    plt.title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def main():
    # read the file into a pandas dataframe
    riding_mowers_df = pd.read_csv('RidingMowers.csv')
    riding_mowers_df['Number'] = riding_mowers_df.index + 1
    riding_mowers_df.head()

    # split the dataframe into random train and test subsets
    train_data, test_data = sklearn.model_selection.train_test_split(riding_mowers_df,
                                                                     test_size=0.4, random_state=26)

    # dataframe of to predict
    predict_data = pd.DataFrame([
        {'Income': 60, 'Lot_Size': 20},
        # {'Income': 90, 'Lot_Size': 22},
        # {'Income': 75, 'Lot_Size': 18},
    ])

    # plot of train, test, and predict dataframes
    plot_dataset(train_data, test_data, predict_data, 'Train and Test Data (filled=Train)')

    # Initialize normalized training, validation, and complete data frames.
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_data[['Income', 'Lot_Size']])

    # Transform the full dataset
    riding_mowers_normalized = pd.concat([pd.DataFrame(scaler.transform(
        riding_mowers_df[['Income', 'Lot_Size']]), columns=['Income', 'Lot_Size']),
        riding_mowers_df[['Ownership', 'Number']]], axis=1)
    train_normalized = riding_mowers_normalized.iloc[train_data.index]
    test_normalized = riding_mowers_normalized.iloc[test_data.index]
    predict_normalized = pd.DataFrame(scaler.transform(predict_data),
                                      columns=['Income', 'Lot_Size'])

    # plot of train, test, and predict dataframes
    plot_dataset(train_normalized, test_normalized, predict_normalized,
                 'Normalized Train and Test Data (filled=Train)')

    # Initialize a data frame with two columns: `k` and `accuracy`
    train_normalized_x = train_normalized[['Income', 'Lot_Size']]
    train_normalized_y = train_normalized['Ownership']
    test_normalized_x = test_normalized[['Income', 'Lot_Size']]
    test_normalized_y = test_normalized['Ownership']

    # iterate over a range of nearest neighbor values
    # train and check with test
    results = []
    for k in range(1, 15):
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k).fit(train_normalized_x,
                                                                        train_normalized_y)
        results.append(
            {'k': k, 'accuracy': sklearn.metrics.accuracy_score(test_normalized_y,
                                                                knn.predict(test_normalized_x))})
    results = pd.DataFrame(results)
    print(results)

    # train with all data
    riding_mowers_x = riding_mowers_normalized[['Income', 'Lot_Size']]
    riding_mowers_y = riding_mowers_normalized['Ownership']
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4).fit(riding_mowers_x,
                                                                    riding_mowers_y)
    distances, indices = knn.kneighbors(predict_normalized)

    predictions = knn.predict(predict_normalized)
    for index, prediction in enumerate(predictions):
        print(64 * '-')
        print(prediction)
        print(predict_data.iloc[index])
        print(riding_mowers_df.iloc[indices[index], :])
        print(riding_mowers_normalized.iloc[indices[index], :])


if __name__ == '__main__':
    main()
