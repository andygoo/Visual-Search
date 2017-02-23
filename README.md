# Visual-Search

Visual-Search finds images that are visually similar to a given image. It first trains a Siamese Neural Network as described in Hadsell et al. 2006 (see references) to learn an image embedding. Then, the k-nearest neighbors algorithm is used to find the closest images according to this embedding. The neural network is trained on training data from the CIFAR-10 dataset.

# Siamese Neural Network Architecture

![alt text](https://github.com/dhruvilbadani/visual-search/model.png "Siamese Neural Network")

### Requirements

Visual-Search requires NumPy, SciPy, Keras, Theano and the PIL Imaging library to run. You will need all of the files on this repo. Some of the data is too big to be out on GitHub. You can download ```cifar-train-data.npy``` and ```cifar-test-data.npy``` [here](https://drive.google.com/drive/folders/0B-25mAWK5f0CTll4RDFlVXpfWjQ?usp=sharing). They should be kept in the same directory as this repo.
```sh
pip install numpy
pip install scipy
pip install keras
pip install Pillow
```
The CIFAR-10 dataset is a set of 60,000 32 x 32 black and white images. Out of these, 50,000 are training images while 10,000 are test images. We train on these 50,000 CIFAR-10 train images. While the search will accept any 32 x 32 black and white image, it is advised to run the search on CIFAR-10 images since the training set was completely CIFAR-10 data. Links to the CIFAR-10 training and test datasets have been provided above.
### Running

```
python visual-search.py <path to CIFAR-10 image>
```
For example:
```
python visual-search.py ./cifar-test-images/0_cat_3.png
```

### References

[1] [Raia Hadsell, Sumit Chopra, Yann LeCun: Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)