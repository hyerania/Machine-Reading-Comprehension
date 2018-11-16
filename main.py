# coding: utf-8
import os

from preprocessing import load_json, preprocess
from embedding import get_glove
from mrcModel import *


if __name__ == "__main__":
	## Static variables
	data_dir = "./dataset/"

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


