import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('epochs',50,'The number of epochs')
flags.DEFINE_string('batch_size',128,'The batch size')

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
	X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
	nb_classes=np.unique(y_train).shape[0]
    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
	shape=X_train.shape[1:]
    # TODO: train your model here
	input = Input(shape=shape)
	x = Flatten()(input)
	x = Dense(150, activation = 'relu') (x)
	x = Dropout (0.5)(x)
	x = Dense(nb_classes, activation = 'softmax') (x)
	model = Model(input, x)
	model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
	
	model.fit(X_train, y_train, FLAGS.batch_size, FLAGS.epochs, validation_data=(X_val, y_val), shuffle = True)
	#model.fit(X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
