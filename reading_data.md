# Reading data

Back to [home page](/tf.notes/)

This page will deep dive into tensorflow implementations pertaining to inputing the data (mainly tf.data.Dataset and tf.data.Iterator). This page can be thought of as an extension to [tf.data](/tf.notes/tf.data) page.

- `tf.data.Dataset` is API is used for the following tasks.
  -  Creating the source construct
  - Applying transformation construct on a dataset
- `tf.data.Iterator` provides the main way to implementation the extraction of data from a dataset. Some iterators may need to be intialized before use (like `make_initializable_iterator`) or iterator which dont need initialization (like `make_one_shot_iterator()`).

## Basic Mechanics [ref]()

So let me start with a very fundamental idea behind dataset, iterator and extraction of data.

- Datasets hold the data, but to be able to use that data in tensorflow, you should import data (and perform some actions if necessary). This all comes under Datasets.
- Iterators are used to as tool for reading the data **once the dataset operations are performed**.
- extracting data is the part used by the estimators to get the data for learning **from the dataset  via iterator**. This include shuffling, batching of the dataset.

## Dataset structure

Before starting with the dataset structure we will lay the foundation of various terminologies used in this module.

### Terminology
--> A dataset contains elements of same structure.
--> Each element may be a combination of tensors (`tf.Tensor`) which are called components.
--> Each component has a data type  tf.Dtype and a shape tf.TensorShape representing static shape of **each element**.

### Dataset

```py
#from_tensor_slices() takes a tensor as an input and returns a Dataset whose elements are slices of the given tensors. [ref: https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset#from_tensor_slices]
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

An important tool is to apply transformations to the input dataset. Transformations can be like normalization. To do the same we use dataset transformation functions like `Dataset.map()`, `Dataset.flat_map()`, `Dataset.apply()` and `Dataset.filter()`. They apply the transformation to each element. I would strongly encourage you to read both these answers on stackoverflow [ans-1](https://stackoverflow.com/a/47096355/7630458) and [ans-2](https://stackoverflow.com/a/47099301/7630458) to have a better understanding.

Psedo code below:

```py
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
```

## Creating Iterator

Once you have built the **dataset to represent your input data**, the next task is to make an iterator to access elements from that dataset. iterators are of following type:
- one shot:
  - no need to initialize
  - most easily usable with an Estimator (atleast till v1.6)
  - `iterator = dataset.make_one_shot_iterator()`
  - The only issue you might run into with this iterator is the that it doesn't support parameterization(explained in initializable iterator).

  Example code
  ```py
  dataset = tf.data.Dataset.range(100)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  for i in range(100):
    value = sess.run(next_element)
  ```
- initializable:
  - require to be explicitly initialized by `iterator.initializer` before using it
  - But with this iterator we can use parameterize the defination of the dataset with `tf.placeholder()`

  example code
  ```py
  max_value = tf.placeholder(tf.int64, shape=[])
  dataset = tf.data.Dataset.range(max_value)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  # Initialize an iterator over a dataset with 10 elements.
  sess.run(iterator.initializer, feed_dict={max_value: 10})
  for i in range(10):
    value = sess.run(next_element)
  # ==> value = 9

  # Initialize the same iterator over a dataset with 100 elements.
  sess.run(iterator.initializer, feed_dict={max_value: 100})
  for i in range(100):
    value = sess.run(next_element)
  # ==> value = 99
  ```
- reinitializable: This is useful when training and testing is performed during same execution.
  - Suppose you have 2 dataset; one for training and other for test now instead of writing 2 iterators which induce 2 sub graphs in tensorboard we can use a reinitializable iterator
  - A more general explanation would be, using different `Dataset` objects that have the same structure (i.e. the same types and compatible shapes for each component)
  - example code
  ```py
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
      # Initialize an iterator over the training dataset.
      sess.run(training_init_op)
      for _ in range(100):
        sess.run(next_element)

      # Initialize an iterator over the validation dataset.
      sess.run(validation_init_op)
      for _ in range(50):
        sess.run(next_element)
  ```
  - explanation through examples is available [here](https://github.com/R1j1t/tf.notes/tree/master/examples)
- feedable: This iterator gives you the freedom to select an iterator for a different `Dataset`. The tensorflow website had and I quote
  >  It offers the same functionality as a reinitializable iterator, but it does not require you to initialize the iterator from the start of a dataset when you switch between iterators. For example, using the same training and validation example from above, you can use tf.data.Iterator.from_string_handle to define a feedable iterator that allows you to switch between the two datasets:

  Example code
  ```py
  # Define training and validation datasets with the same structure.
  training_dataset = tf.data.Dataset.range(100).map(
      lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
  validation_dataset = tf.data.Dataset.range(50)

  # A feedable iterator is defined by a handle placeholder and its structure. We
  # could use the `output_types` and `output_shapes` properties of either
  # `training_dataset` or `validation_dataset` here, because they have
  # identical structure.
  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
      handle, training_dataset.output_types, training_dataset.output_shapes)
  next_element = iterator.get_next()

  # You can use feedable iterators with a variety of different kinds of iterator
  # (such as one-shot and initializable iterators).
  training_iterator = training_dataset.make_one_shot_iterator()
  validation_iterator = validation_dataset.make_initializable_iterator()

  # The `Iterator.string_handle()` method returns a tensor that can be evaluated
  # and used to feed the `handle` placeholder.
  training_handle = sess.run(training_iterator.string_handle())
  validation_handle = sess.run(validation_iterator.string_handle())

  # Loop forever, alternating between training and validation.
  while True:
    # Run 200 steps using the training dataset. Note that the training dataset is
    # infinite, and we resume from where we left off in the previous `while` loop
    # iteration.
    for _ in range(200):
      sess.run(next_element, feed_dict={handle: training_handle})

    # Run one pass over the validation dataset.
    sess.run(validation_iterator.initializer)
    for _ in range(50):
      sess.run(next_element, feed_dict={handle: validation_handle})  
  ```

## Consuming values from an iterator

The last important step is to consume the values, which is done with the help of `Iterator.get_next()`. This method returns one or more `tf.Tensor` objects. Important point to note here is that,
> calling Iterator.get_next() **does not immediately advance** the iterator. Instead you must use the returned tf.Tensor objects in a TensorFlow expression, and pass the result of that expression to tf.Session.run() to get the next elements and advance the iterator.

Once the iterator reaches the end of the dataset, `Iterator.get_next()` returns out of range error.

Example code
```py
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset") #(or break)  # ==> "End of dataset"
```

With this, the detailed module on inputing data has covered a great amount resources, but this is not complete. Some of the function calls were not covered in great detail and we will be doing it next. We will reading the input data from numpy arrays, TFRecord and text in [this](data_from_file) module.

## Processing data with `Dataset.map()`

The `dataset.map(f)` transformation produces **a new dataset** by applying a function `f` to each element of the input dataset. First example looks into image loading and applying necessary transformation before feeding to neural network.

#### Decoding image data and resizing it

It is necessary to convert the input images to same shape and they maybe batched into same shape for GIF. The code is very simple and looks as below:

Steps
1. `tf.read_file()` Reads and outputs the entire contents of the input filename and converts them to a tensor of type `string`.[ref](https://www.tensorflow.org/api_docs/python/tf/read_file)
2. `tf.image.decode_image()` Detects whether an image is a BMP(`decode_bmp`), GIF(`decode_gif`), JPEG(`decode_jpeg`), or PNG(`decode_png`), and performs the appropriate operation to convert the input bytes `string` into a Tensor of type `uint8`. Important to note here is the return value of this function.
>Tensor with type uint8 with shape [height, width, num_channels] for BMP, JPEG, and PNG images and shape [num_frames, height, width, 3] for GIF images.

Here num_channels represent the number of color channels of the decoded image.[ref](https://www.tensorflow.org/api_docs/python/tf/image/decode_image)
3. `tf.image.resize_images()` resizes the input `image` into specified `size` using the specified `method`. If the aspect ratio of image is not equal to the size then it will lead to distortion to prevent that use `[tf.image.resize_image_with_crop_or_pad](https://www.tensorflow.org/api_docs/python/tf/image/resize_image_with_crop_or_pad)`. PLEASE CONTINUE READING THIS [FROM TENSORFLOW SITE](https://www.tensorflow.org/api_docs/python/tf/image/resize_images).

```py
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```  
### Applying arbitrary logic with **tf.py_func()**

It is sometimes useful to call upon external Python libraries when parsing your input data. To do so, invoke, the `tf.py_func()` operation in a `Dataset.map()` transformation. [ref](https://www.tensorflow.org/api_docs/python/tf/py_func)

```py
tf.py_func(
    func, # Funtion to operate
    inp,  # Input tensor objects
    Tout, # list or tuple of tensorflow data types indicating what func returns
    stateful=True,
    name=None
)
```

Example can be found [here](https://www.tensorflow.org/programmers_guide/datasets#applying_arbitrary_python_logic_with_tfpy_func) for this.

## Batching dataset elements

### Simple batching

`Dataset.batch()` is the most basic form of batching where n consecutive elements of a dataset are stacked in a single element.  example code below.

```py
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

### Batching tensors with padding

If the input data has varying sizes then padding is used. `Dataset.padded_batch()` (detailed explanation [here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch)) transformation enables you to batch tensors of different shape by specifying one or more dimensions in which they may be padded. The output may be of variable or constant length.

```py
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

## Randomly shuffling input data by **Dataset.shuffle()**

Tensorflow website explains it breifly as "it Maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer". Example below.

```py
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```
