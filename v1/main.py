import json
import pickle
import os
import sys
import numpy as np
import random

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from training import vector, find_match

try:
    os.chdir(os.path.dirname(sys.argv[0]))
    print(os.getcwd())
except OSError:
    pass

lemm = WordNetLemmatizer()
with open("intents.json") as file:
    intents = json.load(file)

# Load vocabulary and tags from pickle file
vocab = pickle.load(open("vocab.pickle","rb"))
tags = pickle.load(open("tags.pickle","rb"))
responses = []
addons = []

for item in intents["intents"]:
    responses.append(item["response"])
    addons.append(item["addons"])

# Load the trained model
model = load_model("model.h5")

# Use the model to predict the tag / class of a string
def predict_class(string):
    results = {}
    error_threshold = 0.7
    # Convert string into a vector using the training.vector() function
    # Feed the vector into the model
    string_vector = np.array(vector(string, vocab))
    # Model returns NumPy array of the probabilities of each tag in the tags list
    # Converts the array to Python list
    # Sorts it based from lowest to highest 
    prediction = model.predict(np.array([string_vector]))[0].tolist()
    for result, tag in zip(prediction, tags):
        results[result] = tag

    sorted_probs = sorted(list(results.keys()), reverse=True)
    largest = sorted_probs[0]
    tag = results[largest]
    if largest > error_threshold:
        return tag
    return None    

def get_response(tag):
    for t, r, a in zip(tags, responses, addons):
        if t == tag:
            return f"[BOT] {random.choice(r)}"
    return None

predict_class("###PLACEHOLDER###")

print("\n[BOT] I am online\n")
while True:
    try:
        msg = input("Enter a message: ")
    
        tag = predict_class(msg)
        if tag == "goodbye":
            raise KeyboardInterrupt
        response = get_response(tag)
        if response:
            print(response)
        else:
            print("[BOT] I am sorry, I don't understand...")

    except KeyboardInterrupt:
        print(get_response("goodbye"))
        break
