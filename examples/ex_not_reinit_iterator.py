import tensorflow as tf

# Define training and validation datasets with the same structure.

training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50).repeat()

iterator_train = training_dataset.make_one_shot_iterator()
iterator_validation = validation_dataset.make_one_shot_iterator()

next_element_train = iterator_train.get_next()
next_element_val = iterator_validation.get_next()

##training_init_op = iterator.make_initializer(training_dataset)
##validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
with tf.Session() as sess:
    
    writer = tf.summary.FileWriter('/Users/rajat/Documents/graphs/', sess.graph)

    for _ in range(20):
  # Initialize an iterator over the training dataset.
        for _ in range(100):
            sess.run(next_element_train)

      # Initialize an iterator over the validation dataset.
        for _ in range(50):
            sess.run(next_element_val)
