import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

# Define paramaters for the model
learning_rate = 0.01
batch_size = 500
n_epochs = 45
iteration = 0

# Input
height = 28
width = 28
channels = 1
n_inputs = height * width
best_loss_val = np.infty


#Read in data
print(os.listdir("./digit-recognizer"))

train = pd.read_csv('./digit-recognizer/train.csv')
test = pd.read_csv('./digit-recognizer/test.csv')

train.shape
test.shape

y_train = train["label"].values
X_train = train.drop(labels = ["label"],axis = 1).values
X_test = test.values

X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)

img = tf.placeholder(tf.float32, shape=[None, n_inputs], name="img")
label = tf.placeholder(tf.int32, shape=[None], name="label")
training = tf.placeholder_with_default(False, shape=[], name='training')

# Build model

n_hidden1 = 784
n_hidden2 = 1000
#n_hidden3 = 200
n_hidden3 = 400
n_hidden4 = 50
n_outputs = 10

hidden1 = tf.layers.dense(img, n_hidden1, name="hidden1", activation=tf.nn.relu)
dp1= tf.nn.dropout(hidden1,keep_prob = 0.9)
hidden2 = tf.layers.dense(dp1, n_hidden2, name="hidden2", activation=tf.nn.relu)
dp2= tf.nn.dropout(hidden2,keep_prob = 0.9)
hidden3 = tf.layers.dense(dp2, n_hidden3, name="hidden3", activation=tf.nn.relu)
hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.relu)
logits = tf.layers.dense(hidden4, n_outputs, name="outputs")

# Define loss function

# Creating a cross entropy function
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)

# Define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct = tf.nn.in_top_k(logits, label, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

check_interval = batch_size*10
max_checks_without_progress = 20
checks_since_last_progress = 0
best_model_params = None 

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

loss_summary = tf.summary.scalar('loss', loss)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

merged_summary_op = tf.summary.merge_all()

writer1 = tf.summary.FileWriter('./Pgraphs/train', graph = tf.get_default_graph())
writer2 = tf.summary.FileWriter('./Pgraphs/val', graph = tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            sess.run(optimizer, feed_dict={img: X_batch, label: y_batch, training: True})

        train_summary = sess.run(merged_summary_op, feed_dict = {img: X_batch, label: y_batch})
        val_summary = sess.run(merged_summary_op, feed_dict= {img: X_valid, label: y_valid})
        
        writer1.add_summary(train_summary, epoch)
        writer1.flush()
        writer2.add_summary(val_summary, epoch)
        writer2.flush()
        
        acc_batch = accuracy.eval(feed_dict={img: X_batch, label: y_batch})
        acc_val = accuracy.eval(feed_dict={img: X_valid, label: y_valid})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(epoch, acc_batch * 100, acc_val * 100, best_loss_val))

    prediction = sess.run(preds, feed_dict={img: X_test})
    prediction = np.asarray(prediction)
        
    predictions = np.argmax(prediction,axis = 1)
    predictions = pd.Series(predictions,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)
    submission.to_csv("mnist_submission_v1.csv",index=False)