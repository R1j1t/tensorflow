from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

##args = parser.parse_args(argv[1:])
##args = parser.parse_args(argv[1:])

my_feature_colmn = []
(train_x, train_y), (test_x, test_y) = iris_data.load_data()

for key in train_x.keys():
    my_feature_colmn.append(tf.feature_column.numeric_column(key=key))

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for unit in params["hidden_units"]:
        net = tf.layers.dense(net,units = unit, activation=tf.nn.relu)
    
    logits = tf.layers.dense(net,params["n_classes"],activation=None)

    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions={
            'class_ids' :  predicted_classes[:, tf.newaxis],
            'probabilities' : tf.nn.softmax(logits),
            'logits' : logits
            }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    metrics = {'accuracy':accuracy}
    tf.summary.scalar('accuracy',accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_colmn,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })

# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, 100),
    steps=1000)        














