import json
import pickle
import random
import os
import re

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
from training import Trainer

try:
    os.chdir(os.path.dirname(__file__))
except OSError:
    pass

os.chdir(os.path.join(os.getcwd(), "data"))

intents = json.load(open("intents.json", "r"))["intents"]
vocab = pickle.load(open("vocab.pkl", "rb"))
tags = pickle.load(open("tags.pkl", "rb"))
t = Trainer()
model = load_model("Model.h5")
_error_threshold = 0.4
ignore = ".,'/<>!?"
nothing = ["I'm sorry, I don't understand what you mean...",
           "Sorry, I am confused...",
           "Sorry, I don't know what to say..."]

def response(intent):
    if intent is not None:
        index = tags.index(intent)
        responses = intents[index]["response"]
        return random.choice(responses)
    else:
        return random.choice(nothing)

def predict(string):
    for char in ignore:
        string = string.replace(char, "")
    results = {}
    string_vector = np.array(t.vector(string, vocab))
    prediction = model.predict(np.array([string_vector]))[0].tolist()
    for prob, tag in zip(prediction, tags):
        results[prob] = tag

    ordered = sorted(list(results.keys()), reverse=True)
    highest = ordered[0]
    if highest > _error_threshold:
        return results[highest]
    return None

def commands():
    while True:
        command = input("\nYour input -> ")
        tag = predict(command)
        if tag == "goodbye" or command == "/stop":
            break
        reply = response(tag)
        print("[BOT]", reply)

if __name__ == "__main__":
    commands()
