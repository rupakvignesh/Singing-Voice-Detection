import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from sklearn import preprocessing
import scipy.ndimage.morphology as mrph
import scipy.signal as scisig

# Global variables
num_features = 48
orig_feat_size = num_features
num_classes = 2
num_hidden_layers = 2
num_hidden1_nodes = 128
num_hidden2_nodes = 64
num_hidden3_nodes = 32
num_hidden3_nodes = num_features
num_outputs = num_classes
num_context = 5
num_features = (2*num_context+1)*num_features
batch_size = 512
num_epoch = 200
alpha = 0.01
weight_decay_param = 0.1
Beta = 0.001
optimizer = tf.train.RMSPropOptimizer(alpha)
max_to_keep = 10
dataset = '48'
notes = 'CNN, dropout, 2 stage hpsep, add gauss noise '

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def readData(filename):
    with open(filename) as F:
        ip_data = [lines.rstrip() for lines in F]
    ip_data = np.array([map(float,ip_data[i].split(",")) for i in range(len(ip_data))])
    M,N = np.shape(ip_data)
    feats = np.array([ip_data[i][0:N-1] for i in range(M)])
    labels = np.array([ip_data[i][N-1] for i in range(M)])
    #labels = labels[:,None]
    sess = tf.Session()
    num_classes = len(np.unique(labels))
    labels = sess.run(tf.one_hot(labels, num_classes))
    return (feats,labels)

def add_context(features,num_context):
    if num_context==0:
        return features
    else:
        extended_features = np.array([])
        n,m = np.shape(features)
        features_with_context = np.zeros((n,m*(2*num_context+1)))
        temp = np.zeros((num_context*m)).reshape(num_context,m)
        extended_features = np.concatenate((temp,features,temp))

        for i in range(n):
            features_with_context[i] = extended_features[i:i+(2*num_context+1)].reshape(1,(2*num_context+1)*m)

        return features_with_context

def smooth_network_outputs(networkOutputs, window_size):
    col1 = networkOutputs[:,0]
    col2 = networkOutputs[:,1]
    smoothed_col1 = scisig.medfilt(col1, 51)
    smoothed_col2 = scisig.medfilt(col2, 51)
    networkOutputs[:,0] = smoothed_col1
    networkOutputs[:,1] = smoothed_col2
    return networkOutputs

def getClassificationAccuracy(networkOutputs, trueLabels, set_string):
    numberCorrect=0.0
    #networkOutputs = np.argmax(networkOutputs,1)
    #net_dil = mrph.binary_dilation(networkOutputs,np.ones(5))
    #net_dil_er = mrph.binary_erosion(net_dil,np.ones(5))
    for labelInd in range(0, len(trueLabels)):
        if trueLabels[labelInd][np.argmax(networkOutputs[labelInd], 0)]==1:
        #if trueLabels[labelInd][net_dil_er[labelInd]]==1:
            numberCorrect=numberCorrect+1
    print(set_string+' Classification Accuracy: '+str(100*(numberCorrect/len(trueLabels)))+'%')

def pad_zeros(x):
    x_instances = np.shape(x)[0]
    x = np.concatenate((x, np.zeros((x_instances, num_zero_pad))), axis=1)
    return x

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def conv1d(x,W, stride=2):
    return tf.nn.conv1d(x, W, stride=stride, padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,1,32])),
                        'W_conv2':tf.Variable(tf.random_normal([3,3,32,64])),
                        'W_fc1':tf.Variable(tf.random_normal([(orig_feat_size/4)*3*16, 128])),
                        'W_fc2':tf.Variable(tf.random_normal([128, 64])),
                        'out':tf.Variable(tf.random_normal([64, num_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                        'b_conv2':tf.Variable(tf.random_normal([64])),
                        'b_fc1':tf.Variable(tf.random_normal([128])),
                        'b_fc2':tf.Variable(tf.random_normal([64])),
                        'out':tf.Variable(tf.random_normal([num_classes]))}

    x = tf.reshape(x, shape=[-1, orig_feat_size, 2*num_context +1,1])
    #Layer1
    conv1 = tf.nn.conv2d(x,weights['W_conv1'],strides=[1,1,1,1],padding='SAME')
    conv1 = maxpool2d(conv1)
    #Layer2
    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1,1,1,1],padding='SAME')
    conv2 = maxpool2d(conv2)
    #Fully-connected layer 1
    fc = tf.reshape(conv2,[-1,(orig_feat_size/4)*3*16])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc1'])+biases['b_fc1'])
    #Dropout
    fc = tf.nn.dropout(fc, 0.75)
    #Fully-connected layer 2
    fc2 = tf.nn.relu(tf.matmul(fc, weights['W_fc2']+ biases['b_fc2']))
    #Dropout
    fc2 = tf.nn.dropout(fc2, 0.75)

    output = tf.matmul(fc2, weights['out'])+biases['out']
    print(conv1.get_shape())
    print(conv2.get_shape())
    print(weights)
    print(biases)

    return {'output':output, 'conv1': conv1, 'conv2':conv2}


experiments_folder = '/Users/RupakVignesh/Desktop/spring17/7100/experiments/expt24'
summaries_dir = experiments_folder+'/summaries'

#Write Parameters to file
with open(experiments_folder+'/Params.txt','w') as F:
    F.write("Feat dim = "+ str(num_features)+'\n')
    F.write("num_classes = " +str(num_classes)+'\n')
    F.write("num_hidden_layers = "+str(num_hidden_layers)+'\n')
    F.write("num_hidden1_nodes = "+str(num_hidden1_nodes)+'\n')
    F.write("num_hidden2_nodes = "+str(num_hidden2_nodes)+'\n')
    F.write("num_epoch = "+str(num_epoch)+'\n')
    F.write("num_context = "+ str(num_context)+'\n')
    F.write("Alpha = "+str(alpha)+'\n')
    F.write("weight_decay_param ="+str(weight_decay_param)+'\n')
    F.write("batch_size"+str(batch_size)+'\n')
    F.write("Optimizer = "+str(optimizer)+'\n')
    F.write("Dataset = "+dataset+'\n')
    F.write("Notes: "+notes+'\n')

F.close()


print "Building computational graph"
inputs = tf.placeholder(tf.float32,[None,num_features])
std = tf.placeholder(tf.float32)
inp = gaussian_noise_layer(inputs,std)
ground_truth_labels = tf.placeholder(tf.float32,[None, num_classes])


output = convolutional_neural_network(inp)['output']

# compute loss
#loss = tf.reduce_mean(tf.square(tf.subtract(output,ground_truth_labels)))
#regularizers = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) +tf.nn.l2_loss(weights4)
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ground_truth_labels) )
#sparse_penalty = Beta*tf.reduce_mean(tf.abs(hidden1_output)+tf.abs(hidden2_output)+tf.abs(output))
#loss = loss+sparse_penalty
tf.summary.scalar("loss", loss)

#Optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
train = optimizer.minimize(loss, global_step=global_step)

print ("Reading data")
# Read inputs
train_feats, train_labels = readData(sys.argv[1])
valid_feats, valid_labels = readData(sys.argv[2])
test_feats, test_labels = readData(sys.argv[3])
print(np.shape(train_feats))

# Znorm
print ("Z-norm")
train_mean = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mean)/(train_std)
valid_feats = (valid_feats - train_mean)/(train_std)
test_feats = (test_feats - train_mean)/(train_std)

#Add context
print ("Adding context")
train_feats = add_context(train_feats, num_context)
valid_feats = add_context(valid_feats, num_context)
test_feats = add_context(test_feats, num_context)
print(np.shape(train_feats))

print "Training Neural Network"
num_train_instance, _= np.shape(train_feats)
shuffle=np.random.permutation(len(train_feats))
saver = tf.train.Saver(max_to_keep=max_to_keep)

# Train Neural network
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0
        sess_output = np.zeros([num_train_instance, num_classes])

        for batch_ind in range(0,num_train_instance,batch_size):
            _, batch_loss, batch_output, train_summary = sess.run([train,loss,output, merged], feed_dict={inputs:train_feats[shuffle[batch_ind:batch_ind+batch_size]], \
            ground_truth_labels: train_labels[shuffle[batch_ind:batch_ind+batch_size]], std:0.2})

            sess_loss += batch_loss
            sess_output[batch_ind:batch_ind+batch_size] = batch_output

        train_writer.add_summary(train_summary, i)
        valid_summary = sess.run(merged, feed_dict={inputs:valid_feats, ground_truth_labels: valid_labels, std:0.0})
        valid_writer.add_summary(valid_summary,i)

        if ((i+1)%10==0):
            saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+ " loss", sess_loss/(num_train_instance/batch_size))
        getClassificationAccuracy((sess_output), train_labels[shuffle], 'Train ')

        sess_loss, sess_output = sess.run([loss,output], feed_dict={inputs: valid_feats, ground_truth_labels: valid_labels, std:0.0})
        print("Valid Loss = ", sess_loss)
        getClassificationAccuracy(sess_output, valid_labels, 'Valid ')

        sess_loss, sess_output = sess.run([loss,output], feed_dict={inputs: test_feats, ground_truth_labels: test_labels, std:0.0})
        print("Test Loss = ", sess_loss)
        getClassificationAccuracy(sess_output, test_labels, 'Test ')
        print " "
