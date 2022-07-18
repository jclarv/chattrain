import nltk
from nltk.stem import WordNetLemmatizer
from azure.storage.blob import BlobClient
lemmatizer = WordNetLemmatizer()
import json
import pickle
import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from flask import Flask, render_template

app = Flask(__name__)
app.static_folders = 'Static'
app.dynamic_folders = 'Templates'


@app.route('/')
def dynamic_page():
    return render_template("index.html")


@app.route('/button', methods=["GET", "POST"])
def train():
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']
    data_file = open('data.json').read()
    intents = json.loads(data_file)
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word
            w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    # lemmaztize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents
    print(len(documents), "documents")
    # classes = intents
    print(len(classes), "classes", classes)
    # words = all words, vocabulary
    print(len(words), "unique lemmatized words", words)
    pickle.dump(words, open('texts.pkl', 'wb'))
    pickle.dump(classes, open('labels.pkl', 'wb'))
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print("Training data created")
    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # fitting and saving the model
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    model.save('model.h5', hist)
    print("model created")
    from azure.storage.blob import BlobClient

    upload_file_path = 'model.h5'
    sas_url = 'https://ubiq.blob.core.windows.net/ubiqchatbot/model.h5?sp=racwdli&st=2022-07-06T18:58:28Z&se=2027-08-30T02:58:28Z&spr=https&sv=2021-06-08&sr=c&sig=0L6Upu%2BDRCaRjjnsc77lgnypg4rgo0LOCPp27QZ%2Funs%3D'

    client = BlobClient.from_blob_url(sas_url)

    with open(upload_file_path, 'rb') as data:
        client.upload_blob(data,overwrite='true')

    print("**file uploaded**")

    return (train)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
