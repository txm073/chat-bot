import json
import pickle
import random
import time
import os

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


class Model:

    def _create(self, input_shape, output_shape):
        # Units refer to number of neurons in each layer
        self.model = Sequential()

        # Dense layer connected to all the input data
        input_layer = Dense(units=128, input_shape=(input_shape,), activation="relu")

        # Drop out layer ignores the output from certain units
        # In order to prevent overfitting due to generalization of the data
        # Which can decrease accuracy
        first_dropout = Dropout(0.5)

        # Second dense layer connected to all the units
        # From the previous layer that weren't dropped out
        hidden_layer = Dense(units=64, activation="relu")

        # Another drop out layer
        second_dropout = Dropout(0.5)

        # Output layer uses 'softmax' activation
        # Which scales results to return a single tag vector
        output_layer = Dense(units=output_shape, activation="softmax")

        # Add all the layers to the model
        for layer in [input_layer,
                    first_dropout,
                    hidden_layer,
                    second_dropout,
                    output_layer]:

            self.model.add(layer)

    def _train(self, train_x, train_y, epochs, batch_size, filename):
        # Add an optimizer to help the network with gradient descent and speed up the training
        # Random parameters with trial and error!
        optimizer = SGD(momentum=0.9, nesterov=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        # Fit the model around the data
        file = self.model.fit(
                np.array(train_x), # Input data
                np.array(train_y), # Output data
                epochs=epochs, # Number of times the data is viewed
                batch_size=batch_size, # Amount of data viewed at once
                verbose=1 # Relative amount of info displayed in the console
                    )

        # Save the model to be loaded into main script
        self.model.save(filename, file)


class Trainer(Model):

    def __init__(self):
        #self.path = os.path.join(os.path.dirname(sys.argv[0]), "data")
        #os.chdir(self.path)
        with open("intents.json", "r") as f:
            self.intents = json.load(f)["intents"]

        self.lem = WordNetLemmatizer()

    def prepare(self):
        self.vocab = self.get_vocab()
        self.tags = self.get_tags()
        self.get_training_data()
        self.save_data()
        self._create(len(self.train_x[0]), len(self.train_y[0]))

    def save_data(self):
        pickle.dump(self.vocab, open("vocab.pkl", "wb"))
        pickle.dump(self.tags, open("tags.pkl", "wb"))

    def get_training_data(self):
        self.training = []
        for item in self.intents:
            tag = item["tag"]
            patterns = item["patterns"]
            for pattern in patterns:
                self.training.append(
                    [self.vector(pattern, self.vocab), self.vector(tag, self.tags)]
                            )
        random.shuffle(self.training)
        self.training = np.array(self.training, dtype=object)
        self.train_x = list(self.training[:, 0])
        self.train_y = list(self.training[:, 1])

    def get_tags(self):
        return [self.lem.lemmatize(tag) for tag in [i["tag"] for i in self.intents]]

    def get_vocab(self):
        string = ""
        for pattern in [i["patterns"] for i in self.intents]:
            string += " "
            string += " ".join(pattern)

        return list(set([self.lem.lemmatize(word) for word in nltk.word_tokenize(string)]))

    def vector(self, string, dataset):
        _vector = [0] * len(dataset)
        for i, word in enumerate(nltk.word_tokenize(string)):
            word = self.lem.lemmatize(word)
            if word in dataset:
                _vector[dataset.index(word)] = 1

        return _vector

    def train(self, epochs=500, batch_size=30, filename="Model.h5"):
        self._train(self.train_x,
                    self.train_y,
                    epochs=epochs,
                    batch_size=batch_size,
                    filename=filename
                        )


if __name__ == "__main__":
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), "data"))
    except OSError:
        pass

    t = Trainer()
    t.prepare()
    t.train()



