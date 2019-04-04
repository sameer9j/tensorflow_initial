import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 1280
n_epochs = 50
n_train = 60000
n_test = 10000

# Read in data
mnist_folder = 'data/mnist'
#utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# create weights and bias
w = tf.get_variable(name="w",shape=(784,100),initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01,dtype=tf.float32))
b = tf.get_variable(name="b",shape=(100), initializer=tf.zeros_initializer(dtype=tf.float32))

w_h1 = tf.get_variable(name="w_h1",shape=(100,10),initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01,dtype=tf.float32))
b_h1 = tf.get_variable(name="b_h1",shape=(10), initializer=tf.zeros_initializer(dtype=tf.float32))

# Build model
l1 = tf.nn.sigmoid(tf.matmul(img,w)+b)
logits =  tf.matmul(l1,w_h1)+b_h1

# Define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)

# Define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#Try momentum optimizer!
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(loss)

# Calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()

#Output captured below:
"""
Average loss epoch 0: 0.5882294829501662
Average loss epoch 1: 0.2044224739074707
Average loss epoch 2: 0.13688923383868018
Average loss epoch 3: 0.10141981687656669
Average loss epoch 4: 0.08288880047756572
Average loss epoch 5: 0.06626715155881505
Average loss epoch 6: 0.05509507222924122
Average loss epoch 7: 0.04611854627728462
Average loss epoch 8: 0.038516750212672146
Average loss epoch 9: 0.032246889589830886
Average loss epoch 10: 0.02627711240635362
Average loss epoch 11: 0.02236289764906085
Average loss epoch 12: 0.019223189436245795
Average loss epoch 13: 0.015808779053216756
Average loss epoch 14: 0.01408769736109778
Average loss epoch 15: 0.011495583593238925
Average loss epoch 16: 0.00904284514010299
Average loss epoch 17: 0.008532668907888407
Average loss epoch 18: 0.00599905758619655
Average loss epoch 19: 0.004736797965326628
Average loss epoch 20: 0.0041560376777725165
Average loss epoch 21: 0.0032515759081688037
Average loss epoch 22: 0.002866349553927606
Average loss epoch 23: 0.0022685080739604526
Average loss epoch 24: 0.0017949718018170707
Average loss epoch 25: 0.001504836896391109
Average loss epoch 26: 0.0013832453757437857
Average loss epoch 27: 0.0011934468408959897
Average loss epoch 28: 0.0011119561697637967
Average loss epoch 29: 0.0010164195934288898
Average loss epoch 30: 0.000945925412071479
Average loss epoch 31: 0.000855780424561005
Average loss epoch 32: 0.0007906852947916229
Average loss epoch 33: 0.0007260460365931829
Average loss epoch 34: 0.0006672576513834471
Average loss epoch 35: 0.0006423634449886375
Average loss epoch 36: 0.0005962645335848508
Average loss epoch 37: 0.0005643196569127572
Average loss epoch 38: 0.0005271492111713214
Average loss epoch 39: 0.0005015783745600561
Average loss epoch 40: 0.0004646606224881442
Average loss epoch 41: 0.0004397360161271726
Average loss epoch 42: 0.0004250109716870844
Average loss epoch 43: 0.0003962663562730128
Average loss epoch 44: 0.0003741304559340744
Average loss epoch 45: 0.00035675189005709144
Average loss epoch 46: 0.00034188188282166456
Average loss epoch 47: 0.00032837891092978765
Average loss epoch 48: 0.0003105290596194695
Average loss epoch 49: 0.0002998034171976669
Total time: 34.8652617931366 seconds
Accuracy 0.9782
"""