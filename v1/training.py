import os
import json
import sys
import soundex
import random
import pickle
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import time

start = time.time()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

try:
    os.chdir(os.path.dirname(sys.argv[0]))
except OSError:
    pass

# Used to split intent patterns into list of word stems
lemm = WordNetLemmatizer()
with open("intents.json") as file:
    intents = json.load(file) 

# ----- Step 1: Prepare data to load into neural network -----

# Bag of Words Model 
# Used to vectorize input data to turn it into array of 1s and 0s
# Then will be fed into the neural network to train my model

def find_match(a, b):
    a = a.lower()
    b = b.lower()
    s = soundex.getInstance()
    return s.soundex(a) == s.soundex(b)

def vector(string, dataset):
    string_words = nltk.word_tokenize(string.lower())
    string_words = [lemm.lemmatize(word.lower()) for word in string_words]

    # Start with blank list
    output_list = [0] * len(dataset)
    for index, word in enumerate(dataset):
        if word in string_words:
            output_list[index] = 1
    if 1 not in output_list:
        print(f"WARNING: could not vectorize '{string}'")
    
    return output_list


# Only execute this code if the script is not being imported
if __name__ == "__main__":

    # List of all the words that occur in patterns
    vocab = []
    # List of tags for each pattern, a.k.a. intents["tag"]
    tags = []

    ignore = "!?,."

    # List of all training data to feed into neural network
    training = []

    # Get all tags / classes
    for intent in intents["intents"]:
        tags.append(intent["tag"])
        # Get every word that appears in the patterns (get vocab set)
        for pattern in intent["patterns"]:
            words = nltk.word_tokenize(pattern)
            words = [word for word in words if word not in vocab and word not in ignore]

    tags = [lemm.lemmatize(tag.lower()) for tag in tags]

    # Save vocabulary and tags to pickle file
    pickle.dump(vocab, open("vocab.pickle", "wb"))
    pickle.dump(tags, open("tags.pickle", "wb"))

    # Loop through tags
    for index, tag in enumerate(tags):
        current_set = []
        patterns = intents["intents"][index]["patterns"] 

        # Get corresponding pattern
        # Get vector for matching tag and pattern
        # Add the vectors to the training data
        for pattern in patterns:
            pattern_vector = vector(pattern, vocab)
            tag_vector = vector(tag, tags)
            training.append([pattern_vector, tag_vector])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    # Split 2D data array into separate 1D arrays: 
    #  - Input data (word patterns)
    train_x = list(training[:, 0])

    #  - Output data (tag / class that the network predicts)
    train_y = list(training[:, 1])


    # ----- Step 2: Build Neural Network with Keras Sequential Model -----

    # Sequential model for single input and output tensor
    # Units refer to number of neurons in each layer
    model = Sequential()

    # Dense layer connected to all the input data
    # Number of units may have to increase in future
    # If train_x (number of word patterns) increases
    input_layer = Dense(units=128, input_shape=(len(train_x[0]),), activation="relu")

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
    output_layer = Dense(units=len(train_y[0]), activation="softmax")

    # Add all the layers to the model
    for layer in [input_layer, 
                first_dropout, 
                hidden_layer, 
                second_dropout, 
                output_layer]:

        model.add(layer)

    # Add an optimizer to help the network with gradient descent and speed up the training
    # Random parameters with trial and error!
    optimizer = SGD(momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Fit the model around the dataset prepared earlier
    file = model.fit(
            np.array(train_x), # Input data
            np.array(train_y), # Output data
            epochs=512, # Number of times the data is viewed
            batch_size=16, # Amount of data viewed at once
            verbose=1 # Relative amount of info displayed in the console
                )

    # Save the model to be loaded into main script
    model.save("Model.h5", file)

    end = time.time()
    print(f"Program finished in {str(end - start)[:3]}s with no errors")
