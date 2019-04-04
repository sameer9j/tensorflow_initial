import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

# Define paramaters for the model
learning_rate = 0.001
batch_size = 1
n_epochs = 10000
n_train = 4
n_test = 4

# Read in data
train = (([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),([[0.],[1.],[1.],[0.]]))
test = (([[0.,0.],[1.,0.],[1.,1.],[1.,1.]]),([[1.],[0.],[1.],[0.]]))

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

# Create weights and bias
w = tf.get_variable(name="w",shape=(2,4),initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01,dtype=tf.float32))
b = tf.get_variable(name="b",shape=(4), initializer=tf.zeros_initializer(dtype=tf.float32))

w_h1 = tf.get_variable(name="w_h1",shape=(4,1),initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01,dtype=tf.float32))
b_h1 = tf.get_variable(name="b_h1",shape=(1), initializer=tf.zeros_initializer(dtype=tf.float32))

# Build model

l1 = tf.nn.sigmoid(tf.matmul(img,w)+b)
logits =  tf.matmul(l1,w_h1)+b_h1

# Define loss function

entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# Calculate accuracy with test set
preds = tf.nn.sigmoid(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/q1a', tf.get_default_graph())
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
    prediction = sess.run(preds, feed_dict={img: test[0]})
    prediction = np.asarray(prediction)
    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()

"""
Average loss epoch 9990: 0.0131610999815166
Average loss epoch 9991: 0.013156050117686391
Average loss epoch 9992: 0.013151001650840044
Average loss epoch 9993: 0.013145956210792065
Average loss epoch 9994: 0.01314092124812305
Average loss epoch 9995: 0.01313588977791369
Average loss epoch 9996: 0.013130851555615664
Average loss epoch 9997: 0.013125827070325613
Average loss epoch 9998: 0.013120795832946897
Average loss epoch 9999: 0.013115773210301995
Total time: 20.692629098892212 seconds
Accuracy 1.0
"""