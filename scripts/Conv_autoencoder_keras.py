import sys
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.models import Model
from keras import backend as K
from tf_methods import *
import keras

feat_dim = 80
num_context = 10
num_convlayers = 3
batch_size = 128
conv_feature_maps1 = 128
conv_feature_maps2 = 64
conv_feature_maps3 = 32

def round_to_nearest_maxpool(feats, num_convlayers, feat_dim, num_context):
    N = feats.shape[0]
    multiple = 2**num_convlayers
    nearest_next_multiple = np.ceil((2.0*num_context+1)/multiple)*multiple
    num_zeros =  int((nearest_next_multiple - (2*num_context+1))*feat_dim) 
    zeroes_vec = np.zeros(num_zeros)
    rounded_feats = np.array([np.concatenate((i,zeroes_vec),axis=0) for i in feats])
    return rounded_feats

multiple = 2**num_convlayers
new_dim = int(np.ceil((2.0*num_context+1)/multiple)*multiple)
input_img = Input(shape=(feat_dim, new_dim, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(conv_feature_maps1, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(conv_feature_maps2, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(conv_feature_maps3, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(conv_feature_maps3, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(conv_feature_maps2, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(conv_feature_maps1, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')


import numpy as np

train_feats, train_labels = read_data(sys.argv[1])
#valid_feats, valid_labels = readData(sys.argv[2])
test_feats, test_labels = read_data(sys.argv[2])
print(np.shape(train_feats))

# Znorm
print ("Z-norm")
train_mean = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mean)/(train_std)
#valid_feats = (valid_feats - train_mean)/(train_std)
test_feats = (test_feats - train_mean)/(train_std)

#Add context
print ("Adding context")
train_feats = splice_feats(train_feats, num_context)
test_feats = splice_feats(test_feats, num_context)
print(train_feats.shape)
print("Round to nearest maxpool")
train_feats = round_to_nearest_maxpool(train_feats, num_convlayers, feat_dim, num_context)
test_feats = round_to_nearest_maxpool(test_feats, num_convlayers, feat_dim, num_context)
print(train_feats.shape)
print("Reshaping")
train_feats = train_feats.reshape(train_feats.shape[0], feat_dim, new_dim, 1)
test_feats = test_feats.reshape(test_feats.shape[0], feat_dim, new_dim,1)

from keras.callbacks import TensorBoard
autoencoder.summary()
autoencoder.fit(train_feats, train_feats,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(test_feats, test_feats),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

for i in range(2*num_convlayers):
    autoencoder.layers.pop()
pretrained_CNN = autoencoder
outputs = pretrained_CNN.layers[-1].output
outputs = Flatten()(outputs)
flat_layer_dim = int((feat_dim/multiple)*(new_dim/multiple)*32)
outputs = Dense(units=flat_layer_dim, activation='relu')(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(units=128, activations='relu')(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(units=2, activation='softmax')(outputs)
new_model = Model(autoencoder.input, outputs)
new_model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
new_model.summary()

new_model.fit(train_feats, train_labels,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(test_feats, test_labels)
          )

score = new_model.evaluate(test_feats, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
