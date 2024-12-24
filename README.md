# Predict Next Word using LSTM-RNN

This project implements a text generation model to predict the next word in a sentence using Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN). The application demonstrates how sequential data can be used in machine learning for text prediction.

## Project Overview

The model is trained on the text of Hamlet by William Shakespeare to learn the patterns and structure of English sentences. It can predict the next word in a given input sequence of text, showcasing the power of deep learning in natural language processing.

The workflow involves:
1. Data preprocessing: Tokenizing the input text, creating sequences of words, and preparing the data for model training.
2. Model development: Building and training an LSTM-based RNN model to predict the next word.
3. Model deployment: Integrating the trained model into an application interface for real-time predictions.

## Key Features

- Natural language processing for preprocessing and tokenizing textual data
- Deep learning using an LSTM-RNN architecture for sequential text prediction
- An interactive Python-based application for real-time text prediction

## Skills Highlighted

- Machine learning with LSTM and RNN
- Natural language processing including tokenization and text sequence preparation
- Data preprocessing and feature engineering
- Model deployment and real-time inference
- Python programming

## Tools and Technologies Used

- Programming language: Python
- Deep learning framework: TensorFlow/Keras
- Jupyter Notebook for experimentation and model training
- Libraries: NumPy, Pandas, Pickle for data handling, and Matplotlib for visualizations
- Dataset: Shakespeare’s Hamlet
- Model format: HDF5 for saving and loading the trained model
- Dependencies: Listed in requirements.txt for easy setup

## How It Works

1. Preprocessing: The text data is tokenized using a Keras tokenizer and converted into numerical sequences for model training.
2. Training the model: An LSTM-based architecture is trained on the tokenized sequences to learn context and predict the next word.
3. Prediction: The trained model accepts a sequence of words as input and predicts the most probable next word.
4. Application: The app.py script enables real-time interaction with the model for predictions.

