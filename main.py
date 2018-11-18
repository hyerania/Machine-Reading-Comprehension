# coding: utf-8
import argparse
import logging
import os
import sys
import tensorflow as tf


from preprocessing import load_json, preprocess
from embedding import get_glove
from mrcModel import *

logging.basicConfig(level=logging.INFO)

def modelSetup(sess, model, trainDir):
	checkpoint = tf.train.get_checkpoint_state(trainDir)
	path = ""
	if checkpoint:
		path = checkpoint.model_checkpoint_path + ".index"

	# Model previously exists
	if checkpoint and (tf.gfile.Exists(checkpoint.model_checkpoint_path) or tf.gfile.Exists(path)):
		model.saver.restore(sess, checkpoint.model_checkpoint_path)
	else: # No saved checkpoints
		sess.run(tf.global_variables_initializer())



if __name__ == "__main__":
	## Static variables
	data_dir = "./dataset/"
	train_dir = "./train/"

	# Hyperparameters
	learning_rate = 0.001
	batch_size = 60

	# Read data
	# dev_data = load_json(os.path.join(data_dir,"dev-v1.1.json"))
	# train_data = load_json(os.path.join(data_dir,"train-v1.1.json"))
	# print("Loading devset:")
	# preprocess(dev_data, "dev", data_dir)
	# print("Loading trainset:")
	# preprocess(train_data, "train", data_dir)

	## Getting train and dev data
	train_context = os.path.join(data_dir, "train.context")
	train_questions = os.path.join(data_dir, "train.question")
	train_ans_span = os.path.join(data_dir, "train.span")
	dev_context = os.path.join(data_dir, "dev.context")
	dev_questions = os.path.join(data_dir, "dev.question")
	dev_ans_span = os.path.join(data_dir, "dev.span")

	## Create Glove Vector
	id2word, word2id, embed_matrix = get_glove(os.path.join(data_dir,"glove.6B.100d.txt"), 100)

	# Initialize model
	mrcModel = mrcModel(id2word, word2id, embed_matrix)

	# Configuration
	config = tf.ConfigProto()
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode")
	args = parser.parse_args()

	# Train Mode
	if args.mode == 'train':
		print("Training Network")
		logFile = logging.FileHandler(os.path.join(train_dir, "logFile.txt"))
		logging.getLogger().addHandler(logFile)

		with tf.Session(config = config) as sess:
			modelSetup(sess, mrcModel, train_dir)
			print ("Model finished setup")
			# mrcModel.train()

	# Test Mode
	elif args.mode == 'test':
		print("Testing Network")