# tensorflow

## Installation problem
If even after installing tensorflow, you get the error as `No module named 'tensorflow'`,
then it is because you haven't run the following command:

For python 3.x `pip3 install --upgrade tensorflow`
Please note that at present(6-Jan-2018) tensorflow does not support 3.6. So it is 
advised to install 3.5.x to run your progrmas.

For python 2.7 `pip install --upgrade tensorflow`

The above command will install/upgrade the wheels.

## Variable

In tensorflow, the model parameters are defined as varaibles (as mentioned below). Note
that these can be used and even modified by the computation. We have to pass the intial 
value of the `tf.Variable` an example is shown below.

`W = tf.Variable(tf.zeros([784,10]))`

`b = tf.Variable(tf.zeros([10]))`

### Important note
Before Variables can be used within a session, they must be initialized using that session.
This step takes the initial values (in this case tensors full of zeros) that have already 
been specified, and assigns them to each Variable. This can be done for all Variables at once:

`sess.run(tf.global_variables_initializer())` for tensorflow 1.x

`sess.run(tf.initialize_all_variables()) ` for tensorflow 0.x

## Placeholders

In tensorflow, a `placeholder` is a value which we will input when we excute the code
in tensorflow. The difference between `tf.Variable` and `tf.placeholder` is that you
will have to provide the value for `tf.Variable` it is mostly assigned to the variable
such as weights or baises. But `tf.placeholder` is used for the training data ([ref](https://stackoverflow.com/a/36703529/7630458))

A sample code (from: tensorflow.org): 

`x = tf.placeholder(tf.float32, shape=[None, 784])` 

`y_ = tf.placeholder(tf.float32, shape=[None, 10])`

Here, shape defines the shape of the tensor. Where 1st argument (of `shape`) explaing
the number of rows, and 2nd the no. of columns. `None` is like a dynamic input of the 
training examples. The `shape` argument to placeholder is optional, but it allows 
TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes ([ref](https://www.tensorflow.org/get_started/mnist/pros#build_a_softmax_regression_model))

