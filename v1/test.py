import os
import json
import sys
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
    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())
except OSError:
    pass

# Used to split intent patterns into list of word stems
lem = WordNetLemmatizer()
with open("intents.json") as file:
    intents = json.load(file) 

vocab = []
classes = []
docs = []
ignore = "?!,."
training = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        vocab.extend(word_list)
        docs.append((word_list, intent["tag"]))
        classes.append(intent["tag"])
    
vocab = [lem.lemmatize(word) for word in vocab if word not in ignore]
vocab = sorted(set(vocab))
classes = sorted(set(classes))

for doc in docs:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]
    for word in vocab:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

print(training)
end = time.time()
print(f"Program finished in {str(end - start)[:3]}s with no errors")