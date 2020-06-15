import flask 
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import numba
import tensorflow as tf
from tensorflow.keras.models import Model
from tokenizers import BertWordPieceTokenizer

def fast_encode(texts, tokenizer, chunk_size = 256, maxlen = 512):
    # Maximum sequence size for BERT is 512,
    # so we wll truncate any comment that is longer than this.
    tokenizer.enable_truncation(max_length = maxlen)
    # Finally, we need to pad our input so it will have the
    # same size of 512. It means that for any comment that is
    # shorter than 512 tokens, we wll add zeros to reach 512 tokens.
    tokenizer.enable_padding(max_length = maxlen)
    all_ids = []
    # tqdm progress bar: len(texts) // chunk_size
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i + chunk_size].tolist()
        # Tokenize current text chunk
        encs = tokenizer.encode_batch(text_chunk)
        # Extending the list is squeezing the list
        all_ids.extend([enc.ids for enc in encs])
    return np.array(all_ids)


#tokenizer = BertWordPieceTokenizer('tokenizers/distilbert/vocab.txt', lowercase = False)
#print(tokenizer)
# load the pre-trained Keras model
#model = tf.keras.models.load_model('models/distilbert_batch16_epochs3_maxlen192')
#print(model)

def toxic(text):
    word = pd.DataFrame(data = {'content': [text]})
    print(word)
    word_test = fast_encode(word.content.astype(str), tokenizer, maxlen = 192)
    print(word_test)
    word_test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(word_test)
        .batch(16)
    )
    print(word_test_dataset)
    pred = model.predict(word_test_dataset, verbose = 1)
    print(pred)
    return f'Toxic {np.round(pred[0][0] * 100, 3)}%'

# initialize our Flask application and the Keras model
app = Flask(__name__, static_url_path = '', static_folder = 'static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    
    text = request.form['text']
    print(f'request: {text}')

    if text:
        return jsonify({'text' : text})
    return jsonify({'error' : 'No data provided'})
    #print(f'request values: {request.form.values()}')
    #input = [x for x in request.form.values()][0]
    #print(input)

    #output = toxic(input)
    #output = input
    #print(output)

    #return render_template('index.html', prediction_text = f'Toxicity: {output}')

if __name__ == '__main__':
    

    app.run()