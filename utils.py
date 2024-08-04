import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score

def sequence_generator(feature_df, seq_length, seq_cols):
    feature_array = feature_df[seq_cols].values
    num_elements = feature_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield feature_array[start:stop, :]

def label_generator(label_df, seq_length, label):
    label_array = label_df[label].values
    num_elements = label_array.shape[0]
    return label_array[seq_length:num_elements]

def analyze_model(model, input_sequence_set, label_set, set_type='Train'):
    print(f"{set_type} input shape: {input_sequence_set.shape}")
    print(f"{set_type} label shape: {label_set.shape}")
    
    # Ensure label_set is 2D
    if len(label_set.shape) == 1:
        label_set = label_set.reshape(-1, 1)
    
    scores = model.evaluate(input_sequence_set, label_set, verbose=1, batch_size=50)
    print(f'{set_type} Accuracy: {scores[1]}')
    
    y_pred = model.predict(input_sequence_set, verbose=1, batch_size=200)
    y_pred_binary = (y_pred > 0.5).astype("int32")
    y_true = label_set
    
    cm = confusion_matrix(y_true, y_pred_binary)
    print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
    print(cm)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f'{set_type} Precision = {precision}')
    print(f'{set_type} Recall = {recall}')
    print(f'{set_type} F1-score = {f1}')