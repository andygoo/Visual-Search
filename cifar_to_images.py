import numpy as np
import os
import PIL
from PIL import Image
from keras.models import load_model

def unpickle(filename):
    import cPickle
    fo = open(filename, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def bw_to_img(my_array):
	im = Image.fromarray(my_array.reshape((32,32)).astype('uint8')*255)
	return im

X_train = np.load("cifar-train-data.npy")
y_train = np.load("cifar-train-labels.npy")
X_test = np.load("cifar-test-data.npy")
y_test = np.load("cifar-test-labels.npy")

dataset = unpickle('./cifar-10-batches-py/batches.meta')  
index_to_label = dataset['label_names']

base_network = load_model('base_network.h5')
base_network.compile()

try:
    os.remove('./cifar-info.txt')
except OSError:
    pass

input_dim = 1024

with open('./cifar-info.txt', 'a') as f:
	for i in range(len(X_train)):
		x = X_train[i]
		y = y_train[i]
		label = index_to_label[int(y)]
		im = bw_to_img(x)
		feature = base_network.predict(x.reshape(1, input_dim)).T.reshape(128, )
		d = dict()
		name = "{0}_{1}_{2}.png".format(i, label, int(y))
		d['i'] = i
		d['name'] = name
		d['label'] = label
		d['y'] = y
		d['feature'] = list(feature)
		im.save("./cifar-images/{0}".format(name))
		f.write(str(d) + "\n")
