# tensorflow

Back to [home page](tf.notes)

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
```py
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```

- feature_columns as the name suggest is the matrix containing all the features. 

- As you can see a parameter called `hidden_units` which takes a list. Each element of this list is the # of neurons in a layer and total # of layers being the length of this list.

- n_class is the number of possible values our network can predict. So for example, if we have classifier for flowers (as in the case of tensorflow tutorial [ref](https://www.tensorflow.org/get_started/get_started_for_beginners))
the there are 3 possible outcomes of the network.

- An optional parameter of this estimator is called optimizer: basic details can be found [here](https://developers.google.com/machine-learning/glossary/#optimizer)
(I will update this section as I move forward with the tutorials)


### Custom estimators [ref](https://www.tensorflow.org/get_started/custom_estimators)

## Training the model

By defining the model we have the basic structure ready. Now our task is to train the neural network. This can be thought of in terms of sklearn as follows

#### Model creeation (sklearn):

 ```python
 reg_var = linear_model.LinearRegression(fit_intercept=True,          
                                        normalize=True,
                                        copy_X=True,
                                         n_jobs=1)
                                        
 ```
#### Model training (sklearn)
`reg_var.fit(X_train,Y_train)`

#### Model creation (tensorflow)

```py
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```

#### Model training (tensorflow)

In tensorflow much like in sklearn we will use the following syntax:

```python
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```
`steps` hyperparameter is equivalent to number of iterations. `default = 1000`
Note: As mentioned on the tensflow Tutorials more number of iterations doesnt guartantee a better model. 

`input_fn` 'parameter' identifies the function that will provide the training data (including the batch size). Our input training function has entries, 1st features, 2nd labels, 3rd batch size. Important to note here is the datatype in which tensorflow takes input.

`train_feature` is a <b>python dictonary</b> where `keys` are names of the feature and value is an array and each value coresponding to key is an array, containing the values for each example in the training set. You dont have to worry about htis and conversion in explained in the below (hint: it uses tf.data.Dataset)

`train_label` is an array containing the value of the label for each example.

`args.batch_size` is the value of batch size i.e. number of examples used in one iteration. The smaller the number of batch size the faster is the training but with reduced accuracy.

`train_input fn` in the example above defined as follows:

```Python
    def train_input_fn(features, labels, batch_size):
        ## The following call will convert input features and labels into `tf.data.Dataset` 
        dataset = tf.data.Dataset.from_tensor_slices((dict (features), labels))

        return dataset.make_one_shot_iterator().get_next()

        #this return statment passes a batch of examples back to the train method.
```

If you want to shuffle the data (which is recemonded) then you can use `tf.data.Dataset.shuffle(buffer_size=1000)` or in the above case `dataset.shufflebuffer_size=1000()`. If `buffer_size`> # of examples will ensure good shuffling.

## Evaluating the model

In Tensorflow each `Estimator` provides an `evaluate` method. This can be called as follows

```Python

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

The difference between `classifier.train` and `classifier.evaluate` is the data inserted. In `classifier.train` we provide the training data `X_train` and `Y_train` and in `classifier.evaluate` we provide `X_test`, `Y_test`.


## Prediction using learned model

As in the case of evaluation (i.e. error our model has) each model (which we have stored in classifier) has prediction method as well. We will use the function created in while evaluating here as well.

```py
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(X_predict, batch_size=args.batch_size))
```

Our X_predict can be imported from a variety of sources and will be converted to suitable form (dictanaries) by `tf.data.Dataset`.

>The `predict` method returns a python iterable, yielding a dictionary of prediction results for each example. This dictionary contains several keys. The `probabilities` key holds a list of floating-point values, each representing the probability that the input example is a particular label. 
`'probabilities': array([  1.19127117e-08,   3.97069454e-02,   9.60292995e-01])`, here 3rd element (index 2) is post probable
>The class_ids key holds a one-element array that identifies the most probable species. `'class_ids': array([2])`

