import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tf_methods import *
import sys, numpy as np

batch_size = 256
num_classes = 2
epochs = 20
feat_dim = 80
num_context = 20

#Read data
print("reading data")
train_feats, train_labels = read_data(sys.argv[1])
test_feats, test_labels = read_data(sys.argv[2])
# Znorm
print ("Z-norm")
train_mean = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mean)/(train_std)
test_feats = (test_feats - train_mean)/(train_std)
print("Adding context")
train_feats = splice_feats(train_feats, num_context)
test_feats = splice_feats(test_feats, num_context)

# create the model
model = Sequential()
#model.add(LSTM(128, input_dim=80, input_length=41, return_sequences=True))
#model.add(Dropout(0.25))
#model.add(LSTM(64))
#model.add(Dropout(0.25))
model.add(Dense(128, input_shape=(feat_dim*(2*num_context+1),), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
print(model.summary())
model.fit(train_feats, train_labels, epochs=10, batch_size=batch_size,validation_data=(test_feats, test_labels), verbose=1)

