### Import Lab

import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import tensorflow as tf


### Func 1
def Hidden_Layer_Neurons_Calculate(feature_num, label_num):
# Calculate layer number and neuron number for each layer by feature_num and label_num

    # Calculate upper bound of hidden neuron number
    hidden_neo_num_upper = 1
    while hidden_neo_num_upper < feature_num:
        hidden_neo_num_upper *= 2

    # Set at least one layer, the neuron number is hidden_neo_num_upper
    layer_num = 1
    hidden_neo_num = hidden_neo_num_upper
    hidden_neo_num_list = [hidden_neo_num]

    # Calculate layer number and neuron number for each layer
    while (math.floor(hidden_neo_num / 3)) > label_num:
        hidden_neo_num = math.floor(hidden_neo_num / 3)
        layer_num += 1
        hidden_neo_num_list.append(hidden_neo_num)

    # Put feature_num, hidden_layer_num, label_num together into a list
    num_list = hidden_neo_num_list
    num_list.insert(0, feature_num)
    num_list.append(label_num)

    print('The layer number is: %d, and structure is:'% layer_num, num_list)

    return layer_num, num_list


### Func 2
def Weight_Init(layer_num, num_list):
# Definite weight dictionary for different layer number and neuron number

    weight_dict = {}
    stddev = 0.01
    # Put num_list into dictionary
    for i in range(layer_num):
        hidden_layer = {
            'hidden_layer' + str(i + 1): tf.Variable(tf.random_normal([num_list[i], num_list[i + 1]], stddev=stddev))}
        weight_dict.update(hidden_layer)

    out = {'out': tf.Variable(tf.random_normal([num_list[-2], num_list[-1]], stddev=stddev))}
    weight_dict.update(out)

    return weight_dict


### Func 3
def Bias_Init(layer_num, num_list):
# Definite bias dictionary for different layer number and neuron number

    bias_dict = {}
    stddev = 0.01
    # Put num_list into dictionary
    for i in range(layer_num):
        bias_layer = {'bias_layer' + str(i + 1): tf.Variable(tf.random_normal([num_list[i + 1]], stddev=stddev))}
        bias_dict.update(bias_layer)

    out = {'out': tf.Variable(tf.random_normal([num_list[-1]], stddev=stddev))}
    bias_dict.update(out)

    return bias_dict


### Func 4
def NN_Layer_Coef_Set(feature_num, label_num):
# Set layer coefficients

    # Get layer number and neuron number for each layer
    layer_num, num_list = Hidden_Layer_Neurons_Calculate(feature_num, label_num)

    X = tf.placeholder("float", [None, feature_num])
    Y = tf.placeholder("float", [None, label_num])

    # Placeholder for dropout
    keep_prob = tf.placeholder(tf.float32)

    img_reshape = tf.contrib.layers.flatten(X)

    weight_dict = Weight_Init(layer_num, num_list)
    bias_dict = Bias_Init(layer_num, num_list)

    return weight_dict, bias_dict, X, Y, keep_prob, img_reshape, layer_num


### Func 5
def NN_Layer_Set(feature_num, label_num):
# Set layer relationship of the network

    # Get layer coefficients of each layer
    weight_dict, bias_dict, X, Y, keep_prob, img_reshape, layer_num = NN_Layer_Coef_Set(feature_num, label_num)

    # Set layer relationship
    last_layer = img_reshape
    for i in range(layer_num):
        hidden_str = 'hidden_layer' + str(i + 1)
        bias_str = 'bias_layer' + str(i + 1)

        this_layer = tf.add(tf.matmul(last_layer, weight_dict[hidden_str]), bias_dict[bias_str])
        this_layer = tf.nn.relu(this_layer)

        this_layer_dropped = tf.nn.dropout(this_layer, keep_prob)
        last_layer = this_layer_dropped

    out_layer = tf.matmul(last_layer, weight_dict['out']) + bias_dict['out']

    return out_layer, keep_prob, X, Y


### Func 6
def NN_Session(feature_num, label_num, model_name, X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, training_epochs):
# Run tf.Session() and definite related operation
    print('Neural Network!')

    # Set cost compare step
    cost_compare_step = 50

    # Set learning rate decay coefficients and decay rate range
    learning_rate_base = 0.1
    decay_step = math.floor(training_epochs / 100)
    decay_rate_list = [x / 10 for x in range(3, 8)]

    decay_rate = tf.placeholder(tf.float32)
    global_step = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, decay_step, decay_rate, staircase=True)

    # Dropout keep rate range
    keep_rate_list = [x / 10 for x in range(3, 8)]

    # Get output of NN
    logits, keep_prob, X, Y = NN_Layer_Set(feature_num, label_num)

    # Calculate loss function
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    # Set optimizer
    opt = tf.train.AdamOptimizer(learning_rate)

    # Train operate
    train_op = opt.minimize(loss_op)

    # Initialize operate
    init_op = tf.global_variables_initializer()

    # Set save coefficients
    current_path = os.getcwd()
    model_save_path = os.path.join(current_path, 'datasets', 'cartoon_set', model_name)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        highest_score = 0
        # Establish score matrix to store the accuracy score of each parameter combination
        score_matrix = np.zeros([len(decay_rate_list), len(keep_rate_list)])

        print('Start scanning!')
        # Double loop: scanning two hyper-parameter
        for i in range(len(decay_rate_list)):
            for j in range(len(keep_rate_list)):
                print('Parameters Combination ', i * len(keep_rate_list) + j + 1)

                # Initialize at the beginning of each parameter combination
                sess.run(init_op)


                for epoch in range(training_epochs):
                    # Update learning rate
                    learning_rate_now = sess.run(learning_rate, feed_dict={global_step: epoch, decay_rate: decay_rate_list[i]})
                    # Training and get cost
                    _, cost = sess.run([train_op, loss_op], feed_dict={X: X_tr, Y: Y_tr, keep_prob: keep_rate_list[j], decay_rate: decay_rate_list[i]})
                    # print('Epoch: %05d' %(epoch+1), ', cost is: {:.9f}'.format(cost), ', learning rate is: {:.9f}'.format(learning_rate_now))

                    # If cost reaches the threshold
                    if (epoch + 1) % cost_compare_step == 0:
                        if cost < 0.1:
                            break;
                        # print('Accuracy: {:.3f}'.format(accuracy.eval({X: X_tr, Y: Y_tr, keep_prob: 1})))
                        # print('Now learning rate is: %f' %learning_rate_now)

                print('Training end at epoch: %05d' % (epoch + 1), ', Now the cost is: %f' % cost)

                # Calculate the accuracy of this model
                Y_tr_pred = tf.nn.softmax(logits)
                pred_result = tf.equal(tf.argmax(Y_tr_pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(pred_result, 'float'))
                score = sess.run(accuracy, {X: X_tr, Y: Y_tr, keep_prob: 1})
                print('Score is: ', score)

                # Store the accuracy score in matrix
                score_matrix[i, j] = score

                # Update the best parameter combination and their score
                if score > highest_score:
                    print("Best parameter changed!")
                    save_path = saver.save(sess, model_save_path)

                    best_decay_rate = decay_rate_list[i]
                    best_keep_rate = keep_rate_list[j]
                    highest_score = score

        print('Scaning finished!')
        print('The best decay rate is ', best_decay_rate)
        print('The best keep rate is ', best_keep_rate)
        print('The highest score is ', highest_score)
        print(score_matrix)

        # Plate the distribution of score for all the parameter combinations

        plt.figure(figsize=(8, 6))
        plt.title('Accuracy Score Distribution under Different Hyper-parameter')
        plt.xlabel('keep rate')
        plt.ylabel('accuracy score')
        for i in range(len(decay_rate_list)):
            plt.plot(keep_rate_list, score_matrix[i, :], label='decay rate = ' + str(decay_rate_list[i]))
        plt.legend()
        plt.show()

        # Read the best model
        saver.restore(sess, save_path)

        # Get the accuracy for the test data
        Y_pred = tf.nn.softmax(logits)
        pred_result = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_result, 'float'))
        print('Test Accuracy:', accuracy.eval({X: X_tt, Y: Y_tt, keep_prob: 1}))


        # For the Final test with new datasets
        print('----------------------------------------')
        print('For the final test with new dataset:')
        saver.restore(sess, save_path)
        Y_pred = tf.nn.softmax(logits)
        pred_result = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_result, 'float'))
        print('Test Accuracy:', accuracy.eval({X: X_F_tt, Y: Y_F_tt, keep_prob: 1}))