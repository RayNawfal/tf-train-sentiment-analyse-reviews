"""processing the labels of the raw data, use vectorized 300d Glove dataset to train the model for
 binary classification for predicting positive/negative reviews"""

import os
# handling errors
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import data_utils
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM

def train(params=None):
    dirname = os.path.dirname(__file__)
    imdb_dir = os.path.join(dirname, 'aclImdb/aclImdb')
    train_dir = os.path.join(imdb_dir, 'train')

    labels = []
    texts = []

    # Reading the labelled data
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        print(f"folder found: {label_type}")
        for fname in os.listdir(dir_name):
            #looping through text files only
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                # add label type according to file source of text [neg, pos]
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)


    # Trimming text
    maxlen = 100 # cuts off review after 100 words
    training_samples = 5000
    validation_samples = 3000
    max_words = 10000 # top 10,000 words in dataset

    # Tokenization section
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    print(f"Found {len(word_index)} unique tokens")
    # Padding section of the texts list --> text to sequence
    data = data_utils.pad_sequences(sequences, maxlen=maxlen)

    # Labels to arrays
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape) # padded section of TEXTs list
    print('Shape of label tensor:', labels.shape)

    # Shuffle the text and labels
    indices = np.arange(data.shape[0]) #shuffle the data order pos/neg
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[: training_samples]
    y_train = labels[: training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    glove_dir = os.path.join(dirname, 'glove.6B.300d.txt')

    embeddings_index = {}
    f = open(glove_dir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32') # index 0 is a placeholder
        embeddings_index[word] = coefs
    f.close()

    print(f"Found {len(embeddings_index)} word vectors")

    embedding_dim = 300 # 100 depends on glove.6b.11d or 300d
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector # words not found in the embedding index will be all zeros

    # Model definition
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(LSTM(128, dropout=0.2)) # recurrent_dropout=0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu')) # hidden layer
    model.add(Dense(1, activation='sigmoid')) # final output layer with binary classification
    model.summary()

    # Load pretrained word embeddings into the Embedding layer
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # Compile Model
    model.compile(optimizer='adam',
                  loss='BinaryCrossentropy',
                  metrics=['acc']) # for binary classification # rmsprop
    history = model.fit(x_train, y_train,
                        epochs=15,
                        batch_size=32,
                        validation_data=(x_val, y_val)) # batch_size=32 measuring loss and cost function

    # Evaluate the model
    scores = model.evaluate(x_train, y_train, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Save trained model [.h5 = keras save format HDF5]
    model.save(os.path.join(dirname, 'pre_trained_glove_model_05.h5'))

    print("Saved model to disk")

    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    test_dir = os.path.join(imdb_dir, "test")

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    sequences = tokenizer.texts_to_sequences(texts)
    x_test = data_utils.pad_sequences(sequences, maxlen=maxlen)
    y_test = np.asarray(labels)

    print(f"\n---Testing the Model---\n")
    model.load_weights(os.path.join(dirname, 'pre_trained_glove_model_05.h5'))
    loss_acc = model.evaluate(x_test, y_test)
    return ''.join(str(x) for x in loss_acc)
