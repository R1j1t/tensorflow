# Miscellaneous info on tensorflow

## 1 tensorflow dependencies
To do large complex numerical computation like muatrix multiplication, `tf` 
depends on libraries like numpy, which execute the same in other programing language ([ref](https://www.tensorflow.org/get_started/mnist/pros#start_tensorflow_interactivesession)). But there
overheads, and so to make it more efficient, `tf` migrates an entire operation to a different 
language to increase speed and reduce overheads.

## 2 index of the highest entry in a tensor

`tf.argmax` is used to find the index of the highest entry in a tensor along some axis (note here
that, tensor can have more than 2 dimensions, unlike matrix).

example code: `tf.argmax(y,1)`

## 3 Hyperparatmeter defination
Hyperparatmeter are those parameters that you change durring run to get a good model. eg include learning rate, number of iterations etc

## Estimators defination
Estimator as the tf website says is a high level representation of a **complete** model.The estimator API provides method to 
- train the model
- evaluate the model
- generate predictions

![Tensorflow API](https://www.tensorflow.org/images/tensorflow_programming_environment.png)

