# Sign Shield
SignShield consists of two main image classification models: A base CNN and a CNN-LSTM used to classify attacked and clean sequential images of road signs. Both models were trained on the GTSRB dataset, and the CNN-LSTM was trained using the weights of the CNN. 

## Features
- can be trained on alternate images
- high accuracy (90%+ validation accuracy for both models)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
To run the CNN:
```bash
python3 training.py
```
To run the CNN-LSTM:
```bash
python3 cnnlstm.py
```
