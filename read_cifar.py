import numpy as np
import collections
import PIL
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def unpickle(filename):
    import cPickle
    fo = open(filename, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def save_image_from_arr(arr, dim):
	img = Image.fromarray(arr, 'RGB')
	img = img.resize(dim)
	img.save('BW_{0}.png'.format(dim))

def bw_to_img(my_array):
	im = Image.fromarray(my_array.reshape((32,32)).astype('uint8')*255)
	im.save("bw_to_img.png")

full_train_data = np.zeros((50000L, 32L, 32L))
full_train_labels = np.zeros((50000L, ))

for batch_number in range(1, 6):	
	dataset = unpickle('./cifar-10-batches-py/data_batch_{0}'.format(batch_number))  
	data = dataset['data']
	labels = dataset['labels']
	data = np.array([im.reshape((32, 32, 3), order='F') for im in data])
	for i in range(len(data)):
		im = data[i]
		imT = im.T
		for i in range(len(imT)):
			imT[i, :, :] = imT[i, :, :].T
		new_im = imT.T
		data[i] = new_im
	data = np.array([rgb2gray(im) for im in data])
	full_train_data[(batch_number-1)*10000:batch_number*10000] = data
	full_train_labels[(batch_number-1)*10000:batch_number*10000] = labels

np.save("cifar-train-data.npy", full_train_data)
np.save("cifar-train-labels.npy", full_train_labels)

dataset = unpickle('./cifar-10-batches-py/test_batch')
data = dataset['data']
labels = dataset['labels']
print(data.shape, len(labels))
data = np.array([im.reshape((32, 32, 3), order='F') for im in data])
for i in range(len(data)):
	im = data[i]
	imT = im.T
	for i in range(len(imT)):
		imT[i, :, :] = imT[i, :, :].T
	new_im = imT.T
	data[i] = new_im
data = np.array([rgb2gray(im) for im in data])
full_test_data = data
full_test_labels = np.array(labels)

np.save("cifar-test-data.npy", full_test_data)
np.save("cifar-test-labels.npy", full_test_labels)