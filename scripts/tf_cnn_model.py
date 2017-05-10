from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

model_dir = ''
num_context = 5

if __name__ == "__main__":
  tf.app.run()

def add_context(features,num_context):
    features_with_context = []
    extended_features = []
    n,m = np.shape(features)
    zero_arr = np.zeros(m)
    temp = []
    for i in range(num_context):
        temp.append(zero_arr)
    extended_features.extend(temp)
    extended_features.extend(features)
    extended_features.extend(temp)

    for i in range(num_context,n+num_context):
        temp = []
        for j in range(i-2,i+2 +1):
            temp.extend(extended_features[j])
        features_with_context.append(temp)

    return np.array(features_with_context)

def cnn_model_fn(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features, [-1,2*num_context+1,np.shape(features)[1],1])

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[3,3],padding="same",activation=tf.nn.relu)

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3],padding,"same",activation=tf.nn.relu)

    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2,[_,_])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    t = tf.layers.dropout(puts=dense,rate=0.4,training=mode ==learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    loss = None
    train_op = None

    # Calcuate Loss
    if mode!=learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)

    # Configure the training optimizer
    if mode == learn.ModeLeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss,global_step=tf.contrib.framework.get_global_step(), learning_rate=0.001, optimizer="SGD")

    # Generate Predictions
    predictions = {"classes" : tf.argmax(input=logits,axis=1),"probabilities" : tf.nn.softmax(logits,name="softmax_tensor")}

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode,predictions=predictions,loss=loss,train_op=train_op)

def readData(filename):
    with open(filename) as F:
        ip_data = [lines.rstrip() for lines in F]
    ip_data = np.array([map(float,ip_data[i].split(",")) for i in range(len(ip_data))])
    M,N = np.shape(ip_data)
    feats = np.array([ip_data[i][0:N-1] for i in range(M)])
    labels = np.array([ip_data[i][N-1] for i in range(M)])
    labels = labels[:,None]
    sess = tf.Session()
    labels = sess.run(tf.concat(1, [1 - labels, labels]))
    return (feats,labels)

def main(unused_argv):
    # Load training and testing
    train_feats, train_labels = readData(sys.argv[1])
    valid_feats, valid_labels = readData(sys.argv[2])
    test_feats, test_labels = readData(sys.argv[3])

    # Znorm
    train_mean = np.mean(train_feats,axis=0)
    train_std = np.std(train_feats,axis=0)
    train_feats = (train_feats - train_mean)/(train_std)
    valid_feats = (valid_feats - train_mean)/(train_std)
    test_feats = (test_feats - train_mean)/(train_std)

    #Estimator
    classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    #Logging
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

    #Train model
    classifier.fit(x=train_feats,y=train_labels,validation_set=valid_feats,batch_size=10000,steps=20000, monitors=[logging_hook])

    # Configure accuracy metric for evaluation
    metrics = { "accuracy":learn.metric_spec.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}

    # Evaluate model and print results
    eval_results = classifier.evaluate(x=valid_feats, y = valid_labels, metrics=metrics)
    print(eval_results)
