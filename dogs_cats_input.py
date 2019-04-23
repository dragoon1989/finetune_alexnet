import tensorflow as tf

# cifar10 classes = 10
NUM_CLASSES = 10
# raw cifar10 image size = 32x32 with RGB channels
RAW_IMAGE_SIZE = 32
IMAGE_C = 3
# this size is too small to be fed to AlexNet, so the raw image will be zoomed up in pipeline
IMAGE_X = 227
IMAGE_Y = 227
# ImageNet mean for each channel
IMGNET_MEAN = [104., 117., 124.]

# zoom up single batch of raw images to desired size
# this must be applied after dataset batch op
def __zoom_up_image_batch(images):
	'''
	input:
		images : single batch of raw images (BATCH_SIZE x RAW_IMAGE_SIZE x RAW_IMAGE_SIZE x IMAGE_C, dtype=tf.float32
	output:
		resized image (BATCH_SIZE x IMAG_X x IMAG_Y x IMAGE_C, dtype=tf.float32)
	'''
	return tf.image.resize_bilinear(images=images, size=[IMAGE_X, IMAGE_Y])

# parse single example from cifar10 dataset
def __parse_single_example(example):
	''' input : example --- single example (1 + 1024x3 bytes)
	   output: label --- int32 scalar tensor
	   	       image --- float32 3D tensor (format = HWC)'''
	raw_bytes = tf.decode_raw(bytes=example, out_type=tf.uint8)
	# convert label to tf.int32
	label = tf.to_int32(raw_bytes[0])
	# reshape to scalar tensor
	label = tf.reshape(tensor=label, shape=[])
	# convert image to tf.float32
	image = tf.to_float(raw_bytes[1:(1+RAW_IMAGE_SIZE*RAW_IMAGE_SIZE*3)])
	# reshape image to CHW format
	image = tf.reshape(tensor=image, shape=[3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE])
	# permute the image to HWC format (use tf.transpose)
	#image = tf.transpose(a=image, perm=[2, 0, 1])
	image = tf.transpose(a=image, perm=[1, 2, 0])
	# subtrack the image mean
	image -= np.array(IMGNET_MEAN, dtype=np.float32)
	# over
	return label,image

# build a dataset that consists of all examples from a given data file
def __read_single_file(file_name):
	''' input:	file_name --- single data file name (scalar string tensor)
	   output:	dataset --- a dataset that consists of (label, image) elements'''
	# build a fixed length dataset
	dataset = tf.data.FixedLengthRecordDataset(filenames=file_name, record_bytes=1+3*RAW_IMAGE_SIZE*RAW_IMAGE_SIZE)
	# parse all examples and form a new dataset
	dataset = dataset.map(map_func=__parse_single_example)
	# over
	return dataset

# build input pipeline using datasets
def BuildInputPipeline(file_name_list,
				batch_size,
				num_parallel_calls=1,
				num_epoch=1):
	''' input:	file_name_list --- 1D string tensor
			batch_size --- size of batch
			num_parallel_calls
			num_epoch --- number of epochs
	   output:	dataset --- a dataset consisting of batches of (label,image) data from input files'''
	# build a file name dataset
	file_names_dataset = tf.data.Dataset.from_tensor_slices(file_name_list)
	# build a dataset that read all files named by file_names_dataset
	# the dataset consists of (label, image) pairs read from all files indicated in input list
	dataset = file_names_dataset.interleave(map_func=__read_single_file,
								     cycle_length=4,
								     block_length=16,
								     num_parallel_calls=num_parallel_calls)
	# set the epoch
	dataset = dataset.repeat(count=num_epoch)
	# shuffle the dataset
	dataset = dataset.shuffle(buffer_size=10*batch_size)
	# set the batch size
	dataset = dataset.batch(batch_size=batch_size)
	# resize images
	_resize_op = lambda labels, images : (labels, __zoom_up_image_batch(images))
	dataset = dataset.map(map_func=_resize_op, num_parallel_calls=num_parallel_calls)
	# use prefetch to allow asynchronous input
	# i think prefetch one batch is enouth
	dataset = dataset.prefetch(buffer_size=1)
	# over
	return dataset
