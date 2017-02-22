from __future__ import absolute_import
from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

import operator

import random
import numpy as np
import PIL
from PIL import Image
import sys
import Queue

def get_image(image_path):
	return Image.open(image_path)

def convert_image_to_arr(image):
	return np.array(list(image.getdata()))

def get_network(network_json_path):
	return load_model('base_network.h5')

def get_feature(image_path):
	if "train" in image_path:
		return fetch_feature_from_file("cifar-train.txt", image_path)
	else:
		return fetch_feature_from_file("cifar-test.txt", image_path)

def fetch_feature_from_file(feature_file, image_path):
	image_name = image_path.split("/")[-1]
	with open(feature_file, 'r') as f:
		for line in f:
			d = eval(line)
			if d['name'] == image_name:
				return d['feature']

def findknn(feature, k):
	distances = []
	with open('cifar-train.txt', 'r') as f:
		for line in f:
			d = eval(line)
			name = d['name']
			f = d['feature']
			distance = np.linalg.norm(np.array(feature) - np.array(f))
			distances.append(('./cifar-train-images/' + name, distance))
	distances.sort(key=operator.itemgetter(1))
	return [element[0] for element in distances[:k]]

if len(sys.argv) != 2:
	print("Incorrect usage! Correct usage: python visual-search.py <path to CIFAR-10 image>")
	sys.exit()

image_path = sys.argv[1]

f = get_feature(image_path)
print(findknn(f, 10))

