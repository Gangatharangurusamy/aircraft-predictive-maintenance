from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import preprocessing
import io
from waitress import serve

app = Flask(__name__)

# Load the trained model
model = load_model('model_binary_classification_rnn.keras')

# Define constants
sequence_length = 50
w1 = 30

# Define the column names
cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21']

# Define the sequence columns
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(['s' + str(i) for i in range(1, 22)])

# Initialize the MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()

def preprocess_data(df):
    # Normalize the cycle
    df['cycle_norm'] = df['cycle']
    
    # Normalize other columns
    cols_normalize = df.columns.difference(['id', 'cycle'])
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                           columns=cols_normalize, index=df.index)
    df = df[['id', 'cycle']].join(norm_df)
    
    return df

def generate_sequences(df):
    # Group by id and generate sequences
    seq_gen = (df[df['id'] == id][sequence_cols].values[-sequence_length:]
               for id in df['id'].unique() if len(df[df['id'] == id]) >= sequence_length)
    seq_array = np.array(list(seq_gen)).astype(np.float32)
    return seq_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read the CSV file without assigning column names
        content = file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=" ", header=None)
        df.dropna(axis=1, inplace=True)
        # Check the number of columns
        num_columns = len(df.columns)
        
        if num_columns != len(cols_names):
            return jsonify({'error': f'Expected {len(cols_names)} columns, but got {num_columns} columns'})
        
        # Assign column names
        df.columns = cols_names
        
        # Drop any extra columns if they exist
        df = df[cols_names]
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Generate sequences
        sequences = generate_sequences(df)
        
        # Make predictions
        predictions = model.predict(sequences)
        
        # Convert predictions to binary (0 or 1)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Create a result dictionary
        result = {
            'predictions': binary_predictions.tolist(),
            'ids': df['id'].unique().tolist()
        }
        
        return jsonify(result)

if __name__ == '__main__':
    print("Starting server...")
    serve(app, host="0.0.0.0", port=8080)