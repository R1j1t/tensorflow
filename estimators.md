# tf.sstimators

tf websites explains estimators as follows:
>An **Estimator** is any class derived from tf.estimator.Estimator. TensorFlow provides a collection of pre-made Estimators (for example, LinearRegressor) to implement common ML algorithms.

A more through explanation can be found [here](https://www.tensorflow.org/programmers_guide/estimators)

## Parts of Estimators
The below section is relevant to pre-defined estimators but in general true.

1. Creating input function(s)
2. Defining feature columns
3. Intilazing an estimator (predefined from tf), and **specifing** (Data is provided in the next step) the feature columns and various hyperparameters.
4. Train, evaluate and predict on the estimator

All the above parts are explained in great detail on tf website and I would request you to go through them.

### Creating input function(s)

input function is needed for training, evaluation and predictions. An input function will return a **2 element tuple** of features and labels.

- Features (as mentioned in getting started started notes) is a dictonary where key is the feature name and value is an array containing all the examples.
- label - is an array containing the label corresponding to a input (obviously in supervised learning).

To do the transtion of input data to feature dictonary it is recemonded to use `tf.dataset`. More info regarding dataset api and input function is avilable [here](https://www.tensorflow.org/get_started/premade_estimators#create_input_functions)

### Defining feature columns

Feature columns stores the feature values and also stores the meta-data like the feature datatype (features should alsways be numbers and there are ways to handle categorical features) and whether a feature is fixed length. A very detailed article can be found [here](https://www.tensorflow.org/get_started/feature_columns)

### Intilazing an estimator

Intilizing refers to specifing the models and other parameters. eg of some of the models are as follows:

- [tf.estimator.DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)—for deep models that perform multi-class classification.
- [tf.estimator.DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)—for wide-n-deep models.
- [tf.estimator.LinearClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier)— for classifiers based on linear models.

There are many more models avilable, unfortunately i could not find a link where all the predefined models are avilable.

Syntax of model selection goes like this: `tf.estimator.model` where model can be DNNClassifier/DNNLinearCombinedClassifier or others.

```py

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

### Train, evaluate and predict on the estimator

As mentioned in getting started each model has train method, evaluate method and predict method.

#### Training is done as follows:

```py
# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    #steps argument tells the method to stop training after a number of training steps.
    steps=args.train_steps)
```

#### Evaluate the trained model

```py
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

### Making predictions

```py
# Generate predictions from the model

predictions = classifier.predict(
    input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                            batch_size=args.batch_size))
```