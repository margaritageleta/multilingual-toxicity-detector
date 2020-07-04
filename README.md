# Multilingual Toxicity Detector
NLP deep learning model for toxicity detection in text (English, Spanish, Turkish, Russian, French, Portuguese, Italian), trained  on a TS-877 Ryzen-based NAS with 8 cores and 16 threads, with a GeForce GTX 1060 6GB graphics card. 
This repo includes the serving of the model with Tensorflow + Flask + AJAX.

<img src="img/en-non-tox-2.png">

### The model
The input is ingested by a Distilbert Transformer (from [@huggingface](https://github.com/huggingface/transformers)) previously being tokenized by the corresponding tokenizer. Then, the embeddings enter a Funnel component, which models (non-)linear combinations starting from the embedding up to the final node, which contains a neuron with a sigmoid activation function that predicts the toxicity for the given input. 

<div style="display:grid; grid-template-columns: 1fr; grid-template-rows: 1fr 1fr;">

<img src="img/en-non-tox.png">
<img src="img/en-reg.png">

<img src="img/en-reg-2.png">
<img src="img/en-tox.png" >

<img src="img/en-tox-2.png" >
<img src="img/es-non-tox.png" >

<img src="img/turk-non-tox.png" >
<img src="img/es-tox.png" >

<img src="img/es-tox-2.png" >
<img src="img/es-reg.png" >

<img src="img/fr-reg.png" >
<img src="img/fr-tox.png" >

<img src="img/ru-tox.png" >
<img src="img/ru-non-tox.png" >

<img src="img/ru-reg.png" >
<img src="img/ru-tox-2.png" >

</div>
