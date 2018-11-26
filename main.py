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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
	dev_data = load_json(os.path.join(data_dir,"dev-v1.1.json"))
	train_data = load_json(os.path.join(data_dir,"train-v1.1.json"))
	print("Loading devset:")
	preprocess(dev_data, "dev", data_dir)
	print("Loading trainset:")
	preprocess(train_data, "train", data_dir)

	## Getting train and dev data
	train_context = os.path.join(data_dir, "train.context")
	train_questions = os.path.join(data_dir, "train.question")
	train_ans_span = os.path.join(data_dir, "train.span")
	dev_context = os.path.join(data_dir, "dev.context")
	dev_questions = os.path.join(data_dir, "dev.question")
	dev_ans_span = os.path.join(data_dir, "dev.span")

	## Create Glove Vector
	id2word, word2id, embed_matrix = get_glove(os.path.join(data_dir,"glove.6B.100d.txt"), 100)


	# Configuration
	config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode")
	parser.add_argument("--spanMode")
	parser.add_argument("--CharCNN")
	parser.add_argument("--Highway")
	parser.add_argument("--Bidaf")
	args = parser.parse_args()
    
	spanMode = False
	CharCNN = False
	Highway = False
	Bidaf = False
	if args.spanMode == 'True':
		spanMode = True
	if args.CharCNN == 'True':
		CharCNN = True
	if args.Highway == 'True':
		Highway = True
	if args.Bidaf == 'True':
		Bidaf = True
	print('Mode Running:')
	print('SpanMode: ', spanMode)
	print('CharCNN: ', CharCNN)
	print('Highway: ', Highway)
	print('Bidaf: ', Bidaf)
	# Initialize model
	mrcModel = mrcModel(id2word, word2id, embed_matrix, CharCNN = CharCNN, Highway = Highway, Bidaf = Bidaf)

	# Train Mode
	if args.mode == 'train':
		print("Training Network")
		logFile = logging.FileHandler(os.path.join(train_dir, "logFile.txt"))
		logging.getLogger().addHandler(logFile)

		with tf.Session(config = config) as sess:
			modelSetup(sess, mrcModel, train_dir)
			mrcModel.train(sess, train_context, train_questions, train_ans_span, dev_questions, dev_context, dev_ans_span, spanMode= spanMode, CharCNN = CharCNN)
			
		print ("Training Network Finished")
	# Test Mode
	elif args.mode == 'test':
		print("Testing Network")
		with tf.Session(config = config) as sess:
			modelSetup(sess, mrcModel, train_dir)