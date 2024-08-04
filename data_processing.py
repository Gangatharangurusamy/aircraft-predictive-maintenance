import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_data():
    train_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_train.txt", sep=" ", header=None)
    test_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_test.txt", sep=" ", header=None)
    truth_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_truth.txt", sep=" ", header=None)
    
    cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']
    
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    truth_df.dropna(axis=1, inplace=True)
    
    train_df.columns = cols_names
    test_df.columns = cols_names
    
    return train_df, test_df, truth_df

def preprocess_data(train_df, test_df, truth_df):
    # RUL calculation for train data
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    # Failure within window
    w1 = 30
    train_df['failure_within_w1'] = np.where(train_df['RUL'] <= w1, 1, 0)

    # Normalize columns
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'failure_within_w1'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize, index=train_df.index)
    train_df = train_df[['id', 'cycle', 'RUL', 'failure_within_w1']].join(norm_train_df)

    # Preprocess test data
    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize, index=test_df.index)
    test_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    
    # RUL calculation for test data
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['additional_rul']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['additional_rul']
    truth_df.drop('additional_rul', axis=1, inplace=True)
    
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    test_df['failure_within_w1'] = np.where(test_df['RUL'] <= w1, 1, 0)

    return train_df, test_df