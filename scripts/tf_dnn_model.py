import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from sklearn import preprocessing

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

def getClassificationAccuracy(networkOutputs, trueLabels):
    numberCorrect=0.0
    for labelInd in range(0, len(trueLabels)):
        if trueLabels[labelInd][np.argmax(networkOutputs[labelInd], 0)]==1:
            numberCorrect=numberCorrect+1
    print('Classification Accuracy: '+str(100*(numberCorrect/len(trueLabels)))+'%')


# Global variables
num_features = 47
num_classes = 2
num_hidden1_nodes = num_features
num_hidden2_nodes = num_features
num_outputs = num_classes
num_epoch = 1000

print "Building computational graph"
inputs = tf.placeholder(tf.float32,[None,num_features])
ground_truth_labels = tf.placeholder(tf.float32,[None, num_classes])

# first layer
weights1 = tf.Variable(tf.random_normal([num_features,num_hidden1_nodes]))
biases1 = tf.Variable(tf.zeros([num_hidden1_nodes]))
hidden1_output = tf.nn.sigmoid(tf.matmul(inputs,weights1) + biases1)

#second layer
weights2 = tf.Variable(tf.random_normal([num_hidden1_nodes,num_hidden2_nodes]))
biases2 = tf.Variable(tf.zeros([num_hidden2_nodes]))
hidden2_output = tf.nn.softmax(tf.matmul(hidden1_output,weights2)+biases2)

# output layer
weights3 = tf.Variable(tf.random_normal([num_hidden2_nodes,num_outputs]))
biases3 = tf.Variable(tf.zeros([num_outputs]))
output = tf.nn.softmax(tf.matmul(hidden2_output,weights3)+biases3)

# compute loss
loss = tf.reduce_mean(tf.square(tf.subtract(output,ground_truth_labels)))

#Optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.AdamOptimizer()
global_step = tf.Variable(0, name='global_step', trainable=False)
train = optimizer.minimize(loss, global_step=global_step)
#train = optimizer.minimize(loss)

print "Reading data"
# Read inputs
train_feats, train_labels = readData(sys.argv[1])
valid_feats, valid_labels = readData(sys.argv[2])
test_feats, test_labels = readData(sys.argv[3])

#Normalize features
train_feats = preprocessing.scale(train_feats)
valid_feats = preprocessing.scale(valid_feats)
test_feats = preprocessing.scale(test_feats)


print "Training Neural Network"
num_train_instance, _= np.shape(train_feats)
batch_size = 500
# Train Neural network
prev_loss = 0.0
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0
        sess_output = np.zeros([num_train_instance, num_classes])
        for batch_ind in range(0,num_train_instance,batch_size):
            _, batch_loss, batch_output = sess.run([train,loss,output], feed_dict={inputs:train_feats[batch_ind:batch_ind+batch_size], \
            ground_truth_labels: train_labels[batch_ind:batch_ind+batch_size]})

            sess_loss += batch_loss
            sess_output[batch_ind:batch_ind+batch_size] = batch_output
        print("Epoch "+str(i)+ " loss", sess_loss/(num_train_instance/batch_size))
        getClassificationAccuracy(sess_output, train_labels)
        if abs((prev_loss/(num_train_instance/batch_size))-(sess_loss/(num_train_instance/batch_size)))<0.000001:
            break
        prev_loss = sess_loss
        print " "

    sess_loss, sess_output = sess.run([loss,output], feed_dict={inputs: test_feats, ground_truth_labels: test_labels})
    print("Loss = ", sess_loss)
    getClassificationAccuracy(sess_output, test_labels)
