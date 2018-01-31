# tensorflow

## Installation problem
If even after installing tensorflow, you get the error as `No module named 'tensorflow'`,
then it is because you haven't run the following command:

For python 3.x `pip3 install --upgrade tensorflow`
Please note that at present(6-Jan-2018) tensorflow does not support 3.6. So it is advised to install 3.5.x to run your progrmas.

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

## Importing Features

This ia an important part because of the type of data types in the features. TF provides a seperate module called `feature_column` and the entire documentation can be accessed [here](https://www.tensorflow.org/get_started/feature_columns).

## Model selection for our Neural network

To access or create a model you use an `Estimator` class. ([ref](https://developers.google.com/machine-learning/glossary/#Estimators)) 

### Pre-defined models 

Example model DNNClassifier: 
```
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```

>feature_columns as the name suggest is the matrix containing all the features. 

>As you can see a parameter called `hidden_units` which takes a list. Each element of this list is the # of neurons in a layer and total # of layers being the length of this list.

>n_class is the number of possible values our network can predict. So for example, if we have classifier for flowers (as in the case of tensorflow tutorial [ref](https://www.tensorflow.org/get_started/get_started_for_beginners))
the there are 3 possible outcomes of the network.

>An optional parameter of this estimator is called optimizer: basic details can be found [here](https://developers.google.com/machine-learning/glossary/#optimizer)
(I will update this section as I move forward with the tutorials)


### Custom estimators [ref](https://www.tensorflow.org/get_started/custom_estimators)

## Training the model

By defining the model we have the basic structure ready. Now our task is to train the neural network. This can be thought of in terms of sklearn as follows

>Model creeation (sklearn):

 ```
 reg_var = linear_model.LinearRegression(fit_intercept=True,          
                                        normalize=True,
                                        copy_X=True,
                                         n_jobs=1)
                                        
 ```
>Model training (sklearn):
`reg_var.fit(X_train,Y_train)`

>Model training (tensorflow)
In tensorflow much like in sklearn we will use the following syntax:

```
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```

