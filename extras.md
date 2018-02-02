# Miscellaneous info on tensorflow

# 1
To do large complex numerical computation like muatrix multiplication, `tf` 
depends on libraries like numpy, which execute the same in other programing language ([ref](https://www.tensorflow.org/get_started/mnist/pros#start_tensorflow_interactivesession)). But there
overheads, and so to make it more efficient, `tf` migrates an entire operation to a different 
language to increase speed and reduce overheads. 

# 2

`tf.argmax` is used to find the index of the highest entry in a tensor along some axis (note here
that, tensor can have more than 2 dimensions, unlike matrix).

example code: `tf.argmax(y,1)`

# 3
Hyperparatmeter are those parameters that you change durring run to get a good model. eg include learning rate, number of iterations etc
