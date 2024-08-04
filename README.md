# Predictive Maintenance Model and Web Application for Aircraft Systems
This project involves developing an LSTM-based predictive maintenance model for early failure detection in aircraft systems and a Flask web application for real-time data prediction.

## Table of Contents
Overview
Features
Model Performance
Installation
Usage
File Structure
Technologies Used
Contributing

## Overview
The goal of this project is to predict aircraft system failures within a 30-cycle window using an LSTM neural network. Additionally, a Flask web application is provided for users to upload datasets, preprocess them, and obtain failure predictions in real-time.

## Features
### Predictive Maintenance Model:
- Designed and implemented an LSTM-based model for early failure detection.
- Achieved high performance metrics: 86% Precision, 100% Recall, 92% F1-score, and 95.5% accuracy.

### Web Application:
- Real-time data prediction using a Flask web application.
- Data normalization and sequence generation for LSTM model predictions.
- Production deployment using Waitress for reliability.

## Model Performance
- Precision: 86%
- Recall: 100%
- F1-score: 92%
- Accuracy: 95.5%
## Installation
### Prerequisites
- Python 3.8+
- pip

### Clone the Repository
```bash
git clone https://github.com/Gangatharangurusamy/aircraft-predictive-maintenance.git
cd aircraft-predictive-maintenance
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
## File Structure
```bash
.
├── app.py                 # Flask web application
├── data_preprocessing.py  # Data loading and preprocessing
├── main.py                # Main script for training the model
├── model.py               # LSTM model definition
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # HTML template for the web application
├── utils.py               # Utility functions for sequence generation and model analysis
└── README.md              # Project README file
```

## Usage

### Training the Model
```bash
python main.py
```

### Running the Flask Application
```bash
python app,py
```

## Technologies Used
- Languages: Python
- Libraries: TensorFlow, Flask, pandas, scikit-learn
- Web Server: Waitress

## Contributing
Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.


