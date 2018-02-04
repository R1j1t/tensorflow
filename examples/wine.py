import numpy as np
import tensorflow as tf
from sklearn import datasets

wine = datasets.load_wine()

ftr_names = wine.feature_names
features = wine.data
labels = wine.target

wine_names = ['class_0', 'class_1', 'class_2']

x_train = features[:-20,:]
x_test = features[-20:,:]

y_train = labels[:-20]
y_test = labels[-20:]

print ('X_train.shape = %t',(x_train.shape))
print ('X_test.shape = %t',(x_test.shape))
print ('Y_train.shape = %t',(y_train.shape))
print ('Y_test.shape = %t',(y_test.shape))

ftr_dict_train = dict(zip(ftr_names,x_train.T))
ftr_dict_test = dict(zip(ftr_names,x_test.T))

def train_input_fn(features,labels,batch_size):
    ### Slicing ###
    data_ds = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    ### manipulation ###
    assert batch_size is not None,"Input an int for batch size"
    data_ds = data_ds.shuffle(100).repeat().batch(batch_size)
    ## returning ##
    return data_ds.make_one_shot_iterator().get_next()

def evaluate_input_fn(features,labels,batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features,labels)

    data_ds = tf.data.Dataset.from_tensor_slices(inputs)

    data_ds = data_ds.batch(batch_size)

    return data_ds.make_one_shot_iterator().get_next()

my_feature_columns = []
for key in ftr_dict_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns,
                                        hidden_units = [10,10],
                                        n_classes = 3)

classifier.train(input_fn = lambda:train_input_fn(ftr_dict_train,y_train,200),
                steps = 1000)

eval_result = classifier.evaluate(input_fn = lambda:evaluate_input_fn(ftr_dict_test,y_test,100))

print ('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

predictions = classifier.predict(input_fn = lambda:evaluate_input_fn(ftr_dict_test,labels=None,
                                                                     batch_size = 1000))


for pred_dict, expec in zip(predictions, y_test):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(wine_names[class_id],
                              100 * probability, expec))
