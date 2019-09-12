# Contextualized Word Vectors (CoVe)
## Training from Scratch in Tensorflow


### Objective
Replicate the training of CoVe embedding using **TensorFlow**  (the official implementation is in PyTorch).

###  Running the project

##### Download the data and launch raw files preprocessing to tokens (like .xml->.tok)

This will take a while as it downloads and preprocess MT-L and MT-L. It also downloads [GLoVe](https://nlp.stanford.edu/pubs/glove.pdf) and [Char-emb kazuma](https://arxiv.org/abs/1611.01587).
```{bash}
sh ./download_and_preprocess_to_tokens.sh
```

The 3 datasets are:
* **MT-S**: [WMT'16 Multimodal Translation: Multi30k (de-en)](http://www.statmt.org/wmt16/multimodal-task.html)  -  A corpus of **30,000 sentence pairs** that briefly describe Flickr captions (generally referred as Multi30K).

* **MT-M**: [IWSLT'16 (de-en)](https://wit3.fbk.eu) - A corpus of **209,772 sentence pairs** from transcribed TED presentations that cover a wide variety of
topics.

* **MT-L**: [WMT'17 (de-en)](http://data.statmt.org/wmt17/) - A corpus of **7 million sentence pairs** that comes from web crawl data, a news and commentary
corpus, European Parliament proceedings, and European Union press releases.

Regarding the embeddings, we have here 2 possible embeddings, one at the words level (GLoVe) and one at the character level (Char-emb Kazuma):
* [GLoVe](https://nlp.stanford.edu/pubs/glove.pdf): word embedding of dimension 300.
* [Char-emb kazuma](https://arxiv.org/abs/1611.01587): character level embedding (up to 4-grams) of dimension 100.

##### Training of MT-LSTM
Training of MT-LSTM as a 2 layers bidirectional LSTM encoder of an attentional sequence-to-sequence model trained on a Machine Translation task can be found in **CoVe_training_MT_S.ipynb**, **CoVe_training_MT_M.ipynb** and **CoVe_training_MT_L.ipynb**

##### Running the code
Each notebook - [CoVe_training_MT_S.ipynb](https://github.com/EsterHlav/Contextualized-Word-Vectors-CoVe-Learned-in-Translation/blob/master/CoVe_training_MT_S.ipynb), [CoVe_training_MT_M.ipynb](https://github.com/EsterHlav/Contextualized-Word-Vectors-CoVe-Learned-in-Translation/blob/master/CoVe_training_MT_M.ipynb), [CoVe_training_MT_L.ipynb](https://github.com/EsterHlav/Contextualized-Word-Vectors-CoVe-Learned-in-Translation/blob/master/CoVe_training_MT_L.ipynb) for respectively CoVe-S, CoVe-M, CoVe-L - allows to preprocess the data, build and train an MT-LSTM model, then evaluate on validation and test sets the quality of the translation, and finally shows how to compute a CoVe embedding.

##### Warning: should run on AWS/GCP with GPU
* Running the downloading can be very long, with an average of **30min**.
* Running an epoch on a recent MacBook Pro:
  * CoVe-S takes in average **1min**
  * CoVe-M takes in average **5min**
  * CoVe-L takes in average **30min**

### Sources 
* Commands from the [CoVe Github](https://github.com/salesforce/cove) where they explain how to download the data and preprocess to tokenized files: [https://github.com/salesforce/cove](https://github.com/salesforce/cove)
* [Neural Machine Translation (NMT) tensorflow tutorial (official) repository](https://github.com/tensorflow/nmt) where they explain how to make use of [tf.contrib.seq2seq](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq) for NMT: [https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt)
* [Official PyTorch implementation of CoVe](https://github.com/salesforce/cove) from the author of the paper.
