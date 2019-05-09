import sys
import getopt
import os

import tensorflow as tf
import numpy as np

from dogs_cats_input import IMAGE_X
from dogs_cats_input import IMAGE_Y
from dogs_cats_input import NUM_CLASSES

from dogs_cats_input import BuildInputPipeline
from dogs_cats_input import BuildAugmentedInputPipeline
#from alexnet_model import inference
from alexnet_model import loss_func
from alexnet_model import AlexNet

# constants
train_set_size = 25000
test_set_size = 30
train_data_path = ['./data/train/']
test_data_path = ['./data/test/']
weights_path = './alexnet_weights/bvlc_alexnet.npy'
model_path = './tmp/'
summary_path = './tensorboard/'
summary_name = 'summary-default'    # tensorboard default summary dir

# hyperparameters
train_batch_size = 50
test_batch_size = 50
num_epochs = 50
lr0 = 1e-5
dropout_rate = 0.5
l1_scale = 1e-3
l2_scale = 1e-3

# add a switch to control model checkpoint saver (because we do not need checkpoint when studying)
_save_ckpt = False

############################# build the model #############################
# get all file names
train_image_names = os.listdir(train_data_path)
test_image_names = os.listdir(test_data_path)
# build the input pipeline
with tf.name_scope('input_pipeline'):
	# pipeline to read from training set
	aug_data_size, train_dataset = BuildAugmentedInputPipeline(file_path=train_data_path,
										file_names=train_image_names,
										batch_size=train_batch_size,
										raw_data_size=train_set_size,
										num_parallel_calls=4,
										num_epoch=1)
	train_iterator = train_dataset.make_initializable_iterator()
	# pipeline to read from test set
	test_dataset = BuildInputPipeline(file_path=test_data_path,
										file_names=test_image_names,
										batch_size=test_batch_size,
										num_parallel_calls=4,
										num_epoch=1)
	test_iterator = test_dataset.make_initializable_iterator()
	# handle of pipelines
	train_handle = train_iterator.string_handle()
	test_handle = test_iterator.string_handle()
	# build public data entrance
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
	labels, images = iterator.get_next()
	# build placeholder for model input and output
	# batch of data will be fed to these placeholders
	input_images = tf.placeholder(tf.float32, shape=(None, IMAGE_X, IMAGE_Y, 3))
	input_labels = tf.placeholder(tf.int32, shape=None)
	# dropout rate will be fed to this placeholder
	keep_prob = tf.placeholder(tf.float32)

# set global step counter
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

# build the AlexNet model
# the following layers will be trained from scratch
trained_layers = ['fc6', 'fc7', 'fc8']
model = AlexNet(input_images, keep_prob, NUM_CLASSES, trained_layers)

# inference logtis before softmax
logits_before_softmax = model.fc8

# compute loss function
with tf.name_scope('train_loss'):
	# compute loss function
	batch_loss, total_loss = loss_func(input_labels, logits_before_softmax)
	# summary the train loss
	tf.summary.scalar(name='train_loss', tensor=batch_loss)
	# add L1 and L2 regularization
	#model.add_l1_regularization(l1_scale)
	model.add_l2_regularization(l2_scale)

# optimize model parameters
with tf.name_scope('optimization'):
	# define a placeholder to control learning rate
	lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
	# optimize the model
	train_op = tf.train.AdamOptimizer(learning_rate=lr,
									   beta1=0.9,
									   beta2=0.999,
									   epsilon=1e-08).minimize(batch_loss, global_step=global_step)

# evaluate model accuracy on batch of data
with tf.name_scope('train_batch_accuracy'):
	# compute batch predictions (dtype cast to tf.int32)
	batch_predictions = tf.to_int32(tf.math.argmax(tf.nn.softmax(logits_before_softmax), axis=1))
	# compute batch accuracy
	correct_prediction = tf.math.equal(batch_predictions, input_labels)
	batch_accuracy = tf.math.reduce_mean(tf.to_float(correct_prediction))
	# summary the batch accuracy
	tf.summary.scalar(name='train_accuracy', tensor=batch_accuracy)


# function to train the model in one epoch
def train(cur_lr, sess, summary_writer, summary_op):
	'''
	input:
		cur_lr : learning rate for current epoch (scalar)
		sess : tf session to run the training process
		summary_writer : summary writer
		summary_op : summary to write in training process
	'''
	# get iterator handles
	train_handle_val = sess.run(train_handle)
	# initialize iterator
	sess.run(train_iterator.initializer)
	# training loop
	current_batch = 0
	while True:
		try:
			# read batch of data from training set
			batch_labels, batch_images = sess.run([labels, images], feed_dict={handle:train_handle_val})
			# feed this batch to AlexNet
			_, batch_loss_val, batch_acc_val, global_step_val, train_summary_buff = \
				sess.run([train_op, batch_loss, batch_accuracy, global_step, summary_op],
						feed_dict={input_labels : batch_labels,
								   input_images : batch_images,
								   lr : cur_lr,
								   keep_prob : 1-dropout_rate})
			current_batch += 1
			# print indication info
			if current_batch % 50 == 0:
				msg = '\tbatch number = %d, loss = %.2f, train accuracy = %.2f%%' % \
						(current_batch, batch_loss_val, batch_acc_val*100)
				print(msg)
				# write train summary
				summary_writer.add_summary(summary=train_summary_buff, global_step=global_step_val)
		except tf.errors.OutOfRangeError:
			break
	# over

# function to validate the model after one epoch
def validate(sess, summary_writer):
	'''
	input :
		sess : tf session to run the validation
		summary_writer : summary writer
	'''
	# get iterator handle
	test_handle_val = sess.run(test_handle)
	# initialize iterator
	sess.run(test_iterator.initializer)
	# validation loop
	correctness = 0
	loss_val = 0
	while True:
		try:
			# read batch of data from testing set
			batch_labels, batch_images = sess.run([labels, images], feed_dict={handle:test_handle_val})
			# test on single batch
			batch_predictions_val, total_loss_val, global_step_val = \
						sess.run([batch_predictions, total_loss, global_step],
								 feed_dict={input_labels : batch_labels,
											input_images : batch_images,
											keep_prob : 1.0})
			correctness += np.asscalar(np.sum(a=(batch_predictions_val==batch_labels), dtype=np.float32))
			loss_val += np.asscalar(total_loss_val)
		except tf.errors.OutOfRangeError:
			break
	# compute accuracy and loss after a whole epoch
	current_acc = correctness/test_set_size
	loss_val /= test_set_size
	# print and summary
	msg = 'test accuracy = %.2f%%' % (current_acc*100)
	test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
	test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_val)])
	# write summary
	summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_val)
	summary_writer.add_summary(summary=test_loss_summary, global_step=global_step_val)
	# print message
	print(msg)
	# over
	return current_acc

# simple function to adjust learning rate between epochs
def update_learning_rate(cur_epoch):
	'''
	input:
		epoch : current No. of epoch
	output:
		cur_lr : learning rate for current epoch
	'''
	cur_lr = lr0
	if cur_epoch > 10:
		cur_lr = lr0/10
	if cur_epoch >20:
		cur_lr = lr0/100
	if cur_epoch >30:
		cur_lr = lr0/1000
	if cur_epoch >40:
		cur_lr = lr0/2000
	# over
	return cur_lr

# main entrance
if __name__ == "__main__":
	# set tensorboard summary path
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value
	
	# train and test the model
	cur_lr = lr0
	best_acc = 0
	with tf.Session() as sess:
		# initialize variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		
		# initialize IO
		# build tf saver
		saver = tf.train.Saver()
		# build the tensorboard summary
		summary_writer = tf.summary.FileWriter(summary_path+summary_name)
		train_summary_op = tf.summary.merge_all()
		
		# load pretrained AlexNet weights
		# pretrained weights will not be trained
		model.load_initial_weights(sess, weights_path)
		
		# train in epochs
		for cur_epoch in range(1, num_epochs+1):
			# print epoch title
			print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
			# train
			train(cur_lr, sess, summary_writer, train_summary_op)
			# validate
			cur_acc = validate(sess, summary_writer)
			# update learning rate if necessary
			cur_lr = update_learning_rate(cur_epoch)
				
			if cur_acc > best_acc:
				# save check point
				saver.save(sess=sess,save_path=model_path+best_model_ckpt)
				# print message
				print('model improved, save the ckpt.')
				# update best loss
				best_acc = cur_acc
			else:
				# print message
				print('model not improved.')
	# finished
	print('++++++++++++++++++++++++++++++++++++++++')
	print('best accuracy = %.2f%%.'%(best_acc*100))