import tensorflow as tf
import numpy as np

import cv2
import os
import random


# dogs (1) or cats (0)
NUM_CLASSES = 2

# the raw images from kaggle are resized to 227x227 RGB files
IMAGE_X = 227
IMAGE_Y = 227
# image net mean
IMGNET_MEAN = [123.68, 116.779, 103.939]



# analyze a jpg file path and get its label
def __analyze_jpg_label(img_file_path):
	'''
	input : img_file_path	string scalar tensor
	output : a scalar tensor with dtype=tf.int32
	'''
	# split the path to get file name using '/'(Unix-style path)
	segments = tf.string_split(source=[img_file_path], delimiter='/')
	# get file name prefix
	title = segments.values[-1]		# get file name excluding the path part
	title = tf.substr(title, pos=0, len=3)	# get file name prefix (first 3 char)
	# convert to image label (cat or dog)
	logits = tf.equal(x=title, y='dog')
	return tf.to_int32(logits)

# read a jpg file and convert it to alexnet input format
def __read_jpg_file(img_file_path):
	'''
	'''
	# read raw bytes
	raw_bytes = tf.read_file(img_file_path)
	# this function is not supported in latest version ?
	raw_img = tf.image.decode_jpeg(raw_bytes)
	# resize the image to alexnet input size
	resized_img = tf.image.resize_images(raw_img, size=[IMAGE_X, IMAGE_Y])
	# over
	return tf.to_float(resized_img)

# read single jpg file and return the label and image data
def __parse_single_example(img_file_path):
	''' input:	img_file_path --- single jpg file path (scalar string)
		output:	a tuple (label, image), with dtype = int and float32
				output image with shape 
	'''
	# get the image label
	label = __analyze_jpg_label(img_file_path)
	# read image data
	image = __read_jpg_file(img_file_path)
	# subtract image mean
	image = image - IMGNET_MEAN
	# over
	return label, image

# build input pipeline using datasets
def BuildInputPipeline(file_path,
				file_names,
				batch_size,
				num_parallel_calls=1,
				num_epoch=1):
	''' input:	file_path --- python string
			file_names --- python 1D string list
			batch_size --- size of batch
			num_parallel_calls
			num_epoch --- number of epochs
	   output:	dataset --- a dataset consisting of batches of (label,image) data from input files'''
	# get file path list
	file_path_names = [os.path.join(file_path, _name_) for _name_ in file_names]
	# shuffle the list
	random.shuffle(file_path_names)
	# transform 1D python string list to 1D tf string tensor
	tf_file_path_names = tf.constant(file_path_names, dtype=tf.string)
	# build a file name dataset
	file_dataset = tf.data.Dataset.from_tensor_slices(tf_file_path_names)
	# build a dataset that read all files
	# the dataset consists of (label, image) pairs read from all files indicated in input list
	dataset = file_dataset.map(map_func=__parse_single_example, num_parallel_calls=num_parallel_calls)
	# set the epoch
	dataset = dataset.repeat(count=num_epoch)
	# shuffle the dataset
	dataset = dataset.shuffle(buffer_size=10*batch_size)
	# set the batch size
	dataset = dataset.batch(batch_size=batch_size)
	# use prefetch to allow asynchronous input
	# i think prefetch one batch is enouth
	dataset = dataset.prefetch(buffer_size=1)
	# over
	return dataset
