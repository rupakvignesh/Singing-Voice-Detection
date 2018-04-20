
from __future__ import print_function
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from tf_methods import *
import sys, numpy as np
from random import randint
import pdb
from tqdm import tqdm

batch_size = 512
num_classes = 2
epochs = 100
feat_dim = 80
num_context = 20

#Audio  parameters
src_sr = 16000
tgt_sr = 16000
win_size = 640
hop_size = 320
fft_size = 1024
n_mels = 80

valid_gt_path = '/home/rvignesh/singing_voice_separation/MIR1K/Test_GT/' 
train_gt_path = '/home/rvignesh/singing_voice_separation/MIR1K/Train_GT/'

#Read data
print("reading data")
train_back, train_voc, train_mixed = get_wav_from_path(sys.argv[1], src_sr, tgt_sr)
train_feats = get_mel_spec_from_wav(train_mixed, tgt_sr, fft_size, hop_size, n_mels)
train_feats_voc = get_mel_spec_from_wav(train_voc, tgt_sr, fft_size, hop_size, n_mels)
train_feats_back = get_mel_spec_from_wav(train_back, tgt_sr, fft_size, hop_size, n_mels)

valid_back, valid_voc, valid_mixed = get_wav_from_path(sys.argv[2], src_sr, tgt_sr)
valid_feats = get_mel_spec_from_wav(valid_mixed, tgt_sr, fft_size, hop_size, n_mels)
valid_feats_voc = get_mel_spec_from_wav(valid_voc, tgt_sr, fft_size, hop_size, n_mels)
valid_feats_back = get_mel_spec_from_wav(valid_back, tgt_sr, fft_size, hop_size, n_mels)

#Remove first and last frames for valid
remove_frames = lambda x: [feat[:,1:-1] for feat in x]
[train_feats_back, train_feats_voc, valid_feats_back, valid_feats_voc] = list(map(remove_frames, [train_feats_back, train_feats_voc, valid_feats_back, valid_feats_voc]))
[train_feats, valid_feats] = list(map(remove_frames, [train_feats, valid_feats]))
valid_gt = get_gt_from_path(valid_gt_path)
train_gt = get_gt_from_path(train_gt_path)
#Concatenate All Train feats
np_train_feats = np.concatenate(tuple(train_feats), axis=1).T
np_train_feats_voc = np.concatenate(tuple(train_feats_voc), axis=1).T
np_train_feats_back = np.concatenate(tuple(train_feats_back), axis=1).T
np_train_gt = np.concatenate(tuple(train_gt), axis=1).T
train_feats = np.concatenate(tuple(train_feats), axis=1).T
#Concatenate All Valid feats
np_valid_feats = np.concatenate(tuple(valid_feats), axis=1).T
np_valid_feats_voc = np.concatenate(tuple(valid_feats_voc), axis=1).T
np_valid_feats_back = np.concatenate(tuple(valid_feats_back), axis=1).T
np_valid_gt = np.concatenate(tuple(valid_gt), axis=1).T
num_valid_instances, _ = np.shape(np_valid_feats_voc)
valid_feats = np.concatenate(tuple(valid_feats), axis=1).T
# Znorm
print ("Z-norm")
train_mean = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mean)/(train_std)
valid_feats = (valid_feats - train_mean)/(train_std)
#test_feats = (test_feats - train_mean)/(train_std)

with open('train_mean.txt','w') as F:
    for i in range(len(train_mean)):
        F.write(str(train_mean[i])+' ')
        F.write('\n')
F.close()
with open('train_std.txt','w') as F:
    for i in range(len(train_std)):
        F.write(str(train_std[i])+' ')
        F.write('\n')
F.close()
pdb.set_trace()
#Overfit the model
train_feats_voc = np.concatenate((np_train_feats_voc, np_valid_feats_voc))
train_feats_back = np.concatenate((np_train_feats_back, np_valid_feats_back))
np_train_gt = np.concatenate((np_train_gt, np_valid_gt))
train_feats = np.concatenate((train_feats, valid_feats))
print("Adding context")
#np_train_feats_voc = splice_feats(np_train_feats_voc, num_context)
#np_train_feats_back = splice_feats(np_train_feats_back, num_context)
np_valid_feats_voc = splice_feats(np_valid_feats_voc, num_context)
np_valid_feats_back = splice_feats(np_valid_feats_back, num_context)
valid_feats = splice_feats(valid_feats, num_context)
#test_feats = splice_feats(test_feats, num_context)
print(np.shape(np_train_feats_voc))


#Convert to one hot
np_train_gt = np_train_gt.reshape(np.shape(np_train_gt)[0],)
np_valid_gt = np_valid_gt.reshape(np.shape(np_valid_gt)[0],)
np_train_gt = to_categorical(np_train_gt)
np_valid_gt = to_categorical(np_valid_gt)


print("Reshaping")
#train_feats_voc = np_train_feats_voc.reshape(np_train_feats_voc.shape[0], feat_dim, 2*num_context+1, 1)
#train_feats_back = np_train_feats_back.reshape(np_train_feats_back.shape[0], feat_dim, 2*num_context+1, 1)
valid_feats_voc = np_valid_feats_voc.reshape(np_valid_feats_voc.shape[0], feat_dim, 2*num_context+1, 1)
valid_feats_back = np_valid_feats_back.reshape(np_valid_feats_back.shape[0], feat_dim, 2*num_context+1, 1)
valid_feats = valid_feats.reshape(valid_feats.shape[0], feat_dim, 2*num_context+1, 1)
#test_feats = test_feats.reshape(test_feats.shape[0], feat_dim, 2*num_context+1,1)
#print(np.shape(train_feats))
# convert class vectors to binary class matrices


def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 while True:
   index = randint(0, len(features)-batch_size)
   batch_feats = features[index:index+batch_size]
   batch_labels = labels[index:index+batch_size]
   noise = np.random.normal(0, np.random.uniform(0,0.2), np.shape(batch_feats))
   batch_feats += noise
   batch_feats = splice_feats(batch_feats, num_context)
   batch_feats = batch_feats.reshape(batch_feats.shape[0], feat_dim, 2*num_context+1, 1)
   yield batch_feats, batch_labels



#Define sequential model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(feat_dim,2*num_context+1,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3), activation='relu'))
#model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())

num_train_instances, _ = np.shape(train_feats_voc)

for e in range(epochs):
    print("Epoch "+str(e))
    #shuffle = np.random.permutation(num_train_instances)
    #val_shuffle = np.random.permutation(num_valid_instances)
    for batch_ind in tqdm(range(0, num_train_instances, batch_size)):
        #reshape
        batch_x = splice_feats(train_feats[batch_ind:batch_ind+batch_size], num_context)
        batch_x = batch_x.reshape(np.shape(batch_x)[0], feat_dim, 2*num_context+1, 1)
        model.train_on_batch(batch_x, np_train_gt[batch_ind:batch_ind+batch_size])
    if((e+1)%2==0):
        score = model.evaluate(valid_feats, np_valid_gt, batch_size=128)
        print(score)

#model.fit(train_feats_voc+train_feats_back, np_train_gt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_feats_voc+valid_feats_back, np_valid_gt))
#model.fit_generator(generator(train_feats, train_labels, batch_size), steps_per_epoch=int(len(train_feats)/batch_size), epochs=10, verbose=1, validation_data=(valid_feats, valid_labels)) 
# serialize model to JSON
pdb.set_trace()
model_json = model.to_json()
with open("CNN_sequential_model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("CNN_weights.h5")
print("Saved model to disk")

