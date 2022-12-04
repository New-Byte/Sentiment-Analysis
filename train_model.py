import random
import pickle
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd

lemmatizer = WordNetLemmatizer()
print('Opening data.csv...')
# utf-8
df = pd.read_csv('./data/Book1.csv', encoding='latin-1')

intents = df['intents'].tolist() #['c1', 'c2', ....]
patterns = df['patterns'].tolist() #[4,0,2]

print('Data.csv is read...')
words = []
classes = ["Undefined"]
documents = []
punctuations = ['?','!',',','.',';',':', '@', '#','$', '&', '*', '(', ')']
tag = "Undefined"
i = 0
print('Processing the data to build a model....')
while True:
    if i==len(intents):
        break
    elif str(patterns[i]) == '0':
        tag = 'Negative'
    elif str(patterns[i]) == '2':
        tag = 'Neutral'
    elif str(patterns[i]) == '4':
        tag = 'Positive'
    word_list = nltk.word_tokenize(intents[i])
    words.extend(word_list)
    documents.append((word_list, tag))
    if tag not in classes:
        classes.append(tag)
    i += 1
#Preparing, Prepared, Prepare
print('Preparing the data for processing....')
words = [lemmatizer.lemmatize(word) for word in words if word not in punctuations]
words = sorted(set(words))
classes = sorted(set(classes))

print('Dumping the data....')
pickle.dump(words, open("./data/words.pkl", 'wb'))
pickle.dump(classes, open("./data/classes.pkl", 'wb'))

training = []
output = [0] * len(classes)

# bag = [1, 0, 1, 1] 4
# output = [1, 0, 0]
print('Processing the comments...')
for document in documents:
	bag = []
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)

	output_row = list(output)
	output_row[classes.index(document[1])] = 1
	training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
#[[[1,0,1,1],[1,0,0]], []]

# y = w1x1 + w2x2 + .... + b

train_x = list(training[:,0])
train_y = list(training[:,1])

print('Building the model...')
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

print('Training the model....')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=115, batch_size=700, verbose=1)
print('Saving the model....')
model.save("./data/comment_model.h5", hist)

print('Model has graduated and is now ready for the job :)\n')