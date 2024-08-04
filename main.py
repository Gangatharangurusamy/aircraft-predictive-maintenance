import numpy as np
from tensorflow.keras.models import load_model
from data_processing import load_data, preprocess_data
from model import build_model
from utils import sequence_generator, label_generator, analyze_model

# Seed setting
np.random.seed(1234)
PYTHONHASHSEED = 0

# Constants
SEQUENCE_LENGTH = 50

def main():
    # Load and preprocess data
    train_df, test_df, truth_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, truth_df)
    
    # Prepare sequences for training
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + ['s' + str(i) for i in range(1, 22)]
    seq_gen = (list(sequence_generator(train_df[train_df['id'] == id], SEQUENCE_LENGTH, sequence_cols))
               for id in train_df['id'].unique())
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    # Prepare labels for training
    label_gen = (label_generator(train_df[train_df['id'] == id], SEQUENCE_LENGTH, 'failure_within_w1')
                 for id in train_df['id'].unique())
    label_array = np.concatenate(list(label_gen)).astype(np.float32)
    
    # Reshape label_array to 2D
    label_array = label_array.reshape(-1, 1)

    print("Training sequence shape:", seq_array.shape)
    print("Training label shape:", label_array.shape)

    # Build and train model
    model = build_model(SEQUENCE_LENGTH, seq_array.shape[2], 1)
    print(model.summary())
    model.fit(seq_array, label_array, epochs=1, batch_size=200, validation_split=0.05, verbose=2)
    
    # Save model
    model.save('model_binary_classification_rnn.keras')
    
    # Analyze model on train set
    analyze_model(model, seq_array, label_array, 'Train')
    
    # Prepare test data
    test_seq = [test_df[test_df['id'] == id][sequence_cols].values[-SEQUENCE_LENGTH:]
                for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= SEQUENCE_LENGTH]
    test_seq = np.asarray(test_seq).astype(np.float32)
    
    y_mask = [len(test_df[test_df['id'] == id]) >= SEQUENCE_LENGTH for id in test_df['id'].unique()]
    test_labels = test_df.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values
    test_labels = test_labels.reshape(-1, 1).astype(np.float32)
    
    print("Test sequence shape:", test_seq.shape)
    print("Test labels shape:", test_labels.shape)

    # Analyze model on test set
    analyze_model(model, test_seq, test_labels, 'Test')

if __name__ == "__main__":
    main()