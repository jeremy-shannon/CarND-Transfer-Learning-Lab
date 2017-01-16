import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf
# TODO: import Keras layers you need here
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('batch_size', '', "Batch size for model training")
flags.DEFINE_string('epochs', '', "Number of epochs for model training")


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
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    batch_size = int(FLAGS.batch_size)
    epochs = int(FLAGS.epochs)
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    print('Classes:', num_classes, 'Input shape:', input_shape)
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # TODO: train your model here
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, shuffle=True, 
                        validation_data=(X_val, y_val), verbose=2)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
