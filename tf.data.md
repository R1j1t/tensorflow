# dataset

Back to [home page](/tf.notes/)

tf.data module is an introduction to tf.data module and a much more detailed notes can be found under the title [Reading Data](reading_data) . This page will assist in loading the data, manipulate it and pipe it to your model. First we will look at the tensor slicing.

## Reading in-memory data
For this we will use numpy arrays and gradually move to general input methods.

### Slicing
`tf.data.Dataset.from_tensor_slices((array,array...))`

This function (`from_tensor_slices()`) return `tf.data.Dataset` representing **slices** of the array. For example an array of shape (10000,4,3) and I pass this array to the function. So the returned value is of shape (4,3).(We can group the data back by using batch argument explained later here).

An example from tensorflow website is shown below:

```py
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train

mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print(mnist_ds)
```
returned value is :
`<TensorSliceDataset shapes: (28,28), types: tf.uint8>`
It is interesting to note that the number of examples is not printed because the dataset does not know how many items it contain.

Lets try this function with dictonaries as input!

```python
 data_dict={'key1': [34,110],
     'key2': [43,23]}

data_ds = tf.data.Dataset.from_tensor_slices([dict(a)])

print(data_ds)
```

returns the following:
`<TensorSliceDataset shapes: {key2: (), key1: ()}, types: {key2: tf.int32, key1: tf.int32}>`

**Note:** If the value stored in a key is not array then the fn call will return a error.

Try for multiple arrays as argument.

### Manipulation

After slicing step, the next step will be to batch the data and preferably shuffle the data as well. To do this we will use the following command:

`data_ds = data_ds.shuffle(int).repeat().batch(batch_size)`

Each of the following functions perform the following task.

> The `shuffle` method uses a fixed-size buffer to shuffle the items as they pass through. Setting a `buffer_size` greater than the number of examples in the Dataset ensures that the data is completely shuffled.
> The `repeat` method has the Dataset restart when it reaches the end. To limit the number of epochss, set the `count` argument.
> The `batch` method collects a number of examples and stacks them, to create batches. This **adds a dimension to their shape**. The new dimension is added as the first dimension. If the batching of data is not uniform the you may get `data_ds.batch(2)` as `<BatchDataset shapes: ({key2: (?), key1: (?)},() ,()), types: tf.uint8>`

### Returning the value

For returning the the value to other functions you will need to use the following method.

`return dataset.make_one_shot_iterator().get_next()`

The result is a structure of [TensorFlow tensors](https://www.tensorflow.org/programmers_guide/tensors), matching the layout of the items in the Dataset

Sample code for the above explanation is below.

```py

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

```
## Reading data from CSV

This is the most common use case of dataset class.

To read a CSV file we will use the `TextLineDataset` object to read the file line by line. But here the header will contain the details of the CSV file (like created plaform and other things which is not needed for our project) and not training examples so we will skip the 1st colomn by using `skip()` method.

### Builing the dataset

Syntax is as follows: `ds = tf.data.TextLineDataset(path_to_file).skip(1)`

Now our dataset is ready, and its time to parse the data into features and labels.

### Building a CSV line parser

```py
# Metadata describing the features coloumn
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']

FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary of features
    features = dict(zip(COLUMNS,fields))

    # Separate the label from the features
    label = features.pop('label')

    return features, label
```

### Parse the lines
The map method takes a map_func argument that describes how each item in the Dataset should be transformed.
![map method](https://www.tensorflow.org/images/datasets/map.png)
So to parse the lines as they are streamed out of the csv file, we pass our `_parse_line` function to the map method.

Final File for the above explanation is below.

```py

CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

```
