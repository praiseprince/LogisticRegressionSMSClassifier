from flask import Flask, request, jsonify, send_file
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import pickle

with open('parameters.pkl', 'rb') as file:
    variables = pickle.load(file)

word_list_length = variables['word_list_length']
indexed_word_list = variables['indexed_word_list']
w = variables['w']
b = variables['b']

def preprocess_text(X):
    stop = set(stopwords.words('english') + list(string.punctuation))
    if isinstance(X, str):
        X = np.array([X])

    X_preprocessed = []
    for i, sms in enumerate(X):
        sms = np.array([i.lower() for i in word_tokenize(sms) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(sms)
        
    if len(X) == 1:
        return X_preprocessed[0]
    
    return X_preprocessed

def sms_vector(word_list_length, indexed_word_list, X_treated):
    vector = np.zeros(word_list_length, dtype=int)

    for word in X_treated:
        if word in indexed_word_list:
            index = indexed_word_list[word]
            vector[index] += 1

    return vector

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward_pass(X_vector, w, b):
    z = np.dot(X_vector, w) + b
    Y_hat = sigmoid(z) 

    return Y_hat

def prediction(input_text, word_list_length, indexed_word_list, w, b, threshold = 0.25):
    processed_text = preprocess_text(input_text)
    vectorized_text = sms_vector(word_list_length, indexed_word_list, processed_text)
    Y_hat = forward_pass(vectorized_text, w, b)
    pred = (Y_hat >= threshold).astype(int)

    return pred

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/check', methods=['POST'])
def spam_checker():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON provided'}), 400
        
        input_sms = data.get('sms', '')
        
        if not input_sms:
            return jsonify({'error': 'No SMS provided'}), 400
        
        pred = prediction(input_sms, word_list_length, indexed_word_list, w, b)
        result = "The SMS is likely to be spam" if pred == 1 else "The SMS is unlikely to be spam"
        return jsonify({'result': result})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True)