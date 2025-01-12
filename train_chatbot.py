import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open(r'E:\Edunet Internship\my_chatbot\intents.json').read()
intents = json.loads(data_file)

# Tokenize patterns, build words and classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words, lower and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Print out some information
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save the words and classes to pickle files
pickle.dump(words, open(r'E:\Edunet Internship\my_chatbot\words.pkl', 'wb'))
pickle.dump(classes, open(r'E:\Edunet Internship\my_chatbot\classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)  # Empty output array with zeroes

for doc in documents:
    # Initialize the bag of words array
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Create the output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert training data to np.array
random.shuffle(training)

# Ensure that training data is homogeneous
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Training data created")

# Build the model   
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Updated optimizer (no 'lr' argument anymore)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save(r'E:\Edunet Internship\my_chatbot\chatbot_model.h5')

