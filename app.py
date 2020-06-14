import flask 
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tokenizers import BertWordPieceTokenizer

# initialize our Flask application and the Keras model
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # load the pre-trained Keras model
    model = tf.keras.models.load_model('models/distilbert_batch16_epochs3_maxlen192')

    app.run()