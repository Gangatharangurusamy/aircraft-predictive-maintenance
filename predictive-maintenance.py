from tensorflow import keras
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Seed setting
np.random.seed(1234)
PYTHONHASHSEED = 0

# Data loading
def load_data():
    global train_df, test_df, truth_df
    train_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_train.txt", sep=" ", header=None)
    test_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_test.txt", sep=" ", header=None)
    cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    train_df.columns = cols_names
    test_df.columns = cols_names
    truth_df = pd.read_csv(r"E:\project 1\Aircraft Predictive Maintenance\aircraft dataset\PM_truth.txt", sep=" ", header=None)
    truth_df.dropna(axis=1, inplace=True)

def preprocess_data():
    global train_df, test_df
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    w1 = 30
    train_df['failure_within_w1'] = np.where(train_df['RUL'] <= w1, 1, 0)
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'failure_within_w1'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize, index=train_df.index)
    join_df = train_df[['id', 'cycle', 'RUL', 'failure_within_w1']].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize, index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

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

# Sequence generation
sequence_length = 50

def sequence_generator(feature_df, seq_length, seq_cols):
    feature_array = feature_df[seq_cols].values
    num_elements = feature_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield feature_array[start:stop, :]

def label_generator(label_df, seq_length, label):
    label_array = label_df[label].values
    num_elements = label_array.shape[0]
    return label_array[seq_length:num_elements]

# Analyze model on train set
def analyze_model_on_train_set(input_sequence_set, model_name):
    model_history_scores = model_name.evaluate(input_sequence_set, label_set, verbose=1, batch_size=50)
    print('Train Accuracy: {}'.format(model_history_scores[1]))
    y_pred = (model_name.predict(input_sequence_set, verbose=1, batch_size=200) > 0.5).astype("int32")
    y_true = label_set
    test_set = pd.DataFrame(y_pred)
    test_set.to_csv('binary_submit_train.csv', index=None)
    print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
    model_cm = confusion_matrix(y_true, y_pred)
    print(model_cm)
    model_precision = precision_score(y_true, y_pred)
    model_recall = recall_score(y_true, y_pred)
    print('Train Precision = ', model_precision, '\n', 'Train Recall = ', model_recall)

# Analyze model on test set
def analyze_model_on_test_set(input_sequence_columns, model_path):
    import time
    last_test_seq = [test_df[test_df['id'] == id][input_sequence_columns].values[-sequence_length:]
                     for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]
    last_test_seq = np.asarray(last_test_seq).astype(np.float32)
    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    last_test_label = test_df.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values
    last_test_label = last_test_label.reshape(last_test_label.shape[0], 1).astype(np.float32)
    if os.path.isfile(model_path):
        print("using " + model_path)
        model_estimator = load_model(model_path)
    start = time.time()
    scores_test = model_estimator.evaluate(last_test_seq, last_test_label, verbose=2)
    end = time.time()
    print("Total time taken for inferencing: ", "{:.2f}".format((end - start)), " secs")
    print('Test Accuracy: {}'.format(scores_test[1]))
    y_model_estimator_pred_test = (model_estimator.predict(last_test_seq) > 0.5).astype("int32")
    y_true_test = last_test_label
    test_set = pd.DataFrame(y_model_estimator_pred_test)
    test_set.to_csv('binary_submit_test.csv', index=None)
    print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
    model_estimator_conf_m = confusion_matrix(y_true_test, y_model_estimator_pred_test)
    print(model_estimator_conf_m)
    model_estimator_precision_test = precision_score(y_true_test, y_model_estimator_pred_test)
    model_estimator_recall_test = recall_score(y_true_test, y_model_estimator_pred_test)
    f1_test = 2 * (model_estimator_precision_test * model_estimator_recall_test) / (
                model_estimator_precision_test + model_estimator_recall_test)
    print('Test Precision: ', model_estimator_precision_test, '\n', 'Test Recall: ', model_estimator_recall_test, '\n',
          'Test F1-score:', f1_test)

def build_and_train_model(label_set):
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols_25 = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols_25.extend(sensor_cols)
    seq_gen = (list(sequence_generator(train_df[train_df['id'] == id], sequence_length, sequence_cols_25))
               for id in train_df['id'].unique())
    seq_set_f25 = np.concatenate(list(seq_gen)).astype(np.float32)
    features_dim = seq_set_f25.shape[2]
    out_dim = label_set.shape[1] if len(label_set.shape) > 1 else 1  # Handle case for 1D label_set
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, features_dim),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=out_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(seq_set_f25, label_set, epochs=1, batch_size=200, validation_split=0.05, verbose=2)
    model.save('model_binary_classification_rnn.h5')

# Main execution
if __name__ == "__main__":
    load_data()
    preprocess_data()
    
    # Generate labels for training
    label_gen = (label_generator(train_df[train_df['id'] == id], sequence_length, 'failure_within_w1')
                 for id in train_df['id'].unique())
    label_set = np.concatenate(list(label_gen)).astype(np.float32)
    print("Label Set Shape:", label_set.shape)
    
    build_and_train_model(label_set)
