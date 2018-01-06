# Miscellaneous info on tensorflow

To do large complex numerical computation like muatrix multiplication, `tf` 
depends on libraries like numpy, which execute the same in other programing language ([ref](https://www.tensorflow.org/get_started/mnist/pros#start_tensorflow_interactivesession)). But there
overheads, and so to make it more efficient, `tf` migrates an entire operation to a different 
language to increase speed and reduce overheads. 

