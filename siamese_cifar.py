'''Adapted from keras' mnist_siamese_graph.py based on the following paper:

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")

import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop, SGD
from keras.utils.visualize_util import plot
from keras import backend as K

import os

import PIL
from PIL import Image

def unpickle(filename):
    import cPickle
    fo = open(filename, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def bw_to_img(my_array):
    im = Image.fromarray(my_array.reshape((32,32)).astype('uint8')*255)
    return im

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print("eucl_dist_output_shape: {0}".format((shape1[0], 1)))
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 10
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
X_train, y_train, X_test, y_test = np.load("cifar-train-data.npy"), np.load("cifar-train-labels.npy"), np.load("cifar-test-data.npy"), np.load("cifar-test-labels.npy")
X_train = X_train.reshape(50000, 1024)
X_test = X_test.reshape(10000, 1024)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255
input_dim = 1024
nb_epoch = 20

print(X_train[0].shape)

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

print(np.array([tr_pairs[:, 0], tr_pairs[:, 1]]).shape)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
hist = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          nb_epoch=nb_epoch)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

base_network_json = base_network.to_json()
with open("base_network.json", "w") as json_file:
    json_file.write(base_network_json)


base_network.save('base_network.h5')
base_network.save_weights('base_network_weights.h5')

print(hist.history)

plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def extract_features():
    try:
        os.remove('./cifar-train.txt')
    except OSError:
        pass
    try:
        os.remove('./cifar-test.txt')
    except OSError:
        pass

    input_dim = 1024

    dataset = unpickle('./cifar-10-batches-py/batches.meta')  
    index_to_label = dataset['label_names']

    with open('./cifar-train.txt', 'a') as f:
        for i in range(len(X_train)):
            x = X_train[i]
            y = y_train[i]
            label = index_to_label[int(y)]
            im = bw_to_img(x * 255)
            feature = base_network.predict(x.reshape(1, input_dim)).T.reshape(128, )
            d = dict()
            name = "{0}_{1}_{2}.png".format(i, label, int(y))
            d['i'] = i
            d['name'] = name
            d['label'] = label
            d['y'] = y
            d['feature'] = list(feature)
            im.save("./cifar-train-images/{0}".format(name))
            f.write(str(d) + "\n")

    with open('./cifar-test.txt', 'a') as f:
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]
            label = index_to_label[int(y)]
            im = bw_to_img(x * 255)
            feature = base_network.predict(x.reshape(1, input_dim)).T.reshape(128, )
            d = dict()
            name = "{0}_{1}_{2}.png".format(i, label, int(y))
            d['i'] = i
            d['name'] = name
            d['label'] = label
            d['y'] = y
            d['feature'] = list(feature)
            im.save("./cifar-test-images/{0}".format(name))
            f.write(str(d) + "\n")

extract_features()
