# coding: utf-8
import json
import os
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from layers import Highway, RNNEncoder, BidafAttention, SimpleSoftmaxLayer
from helperFunctions import masked_softmax

### Model
class mrcModel(object):
    def __init__(self, id2word, word2id, embed_matrix):
        ### Hyperparameters
        self.hidden_bidaf_size = 150
        self.hidden_encoder_size = 150
        self.hidden_full_size = 200
        self.context_len = 300
        self.question_len = 30
        # embed_size = 100

        self.id2word = id2word
        self.word2id = word2id
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embed_layer(embed_matrix)
            self.create_layers()
            self.add_loss()
        
        print("Finished initialization of model")

    def add_placeholders(self):
        # Add placeholders for the inputs
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.context_len])
        self.question_ids = tf.placeholder(tf.int32, shape=[None, self.question_len])
        self.question_mask = tf.placeholder(tf.int32, shape=[None, self.question_len])
        self.answer_span = tf.placeholder(tf.int32, shape=[None, 2]) # The start and end index

        # Add a placeholder to feed in the probability (for dropout)
        self.prob_dropout = tf.placeholder_with_default(1.0, shape=())

    def add_embed_layer(self, embed_matrix):
#         with vs.variable_scope("embedding"):
        with tf.variable_scope("embedding"):
            embedding_matrix = tf.constant(embed_matrix, dtype=tf.float32, name="embed_matrix")
            self.context_embed = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids)
            self.question_embed = embedding_ops.embedding_lookup(embedding_matrix, self.question_ids)
    
    def create_layers(self):
        ### Add highway layer
        embed_size = self.context_embed.get_shape().as_list()[-1]
        high_way = Highway(embed_size, -1.0)
        for i in range(2):
            self.context_embed = high_way.add_layer(self.context_embed, scopename = "HighwayLayer")
            self.question_embed = high_way.add_layer(self.question_embed, scopename = "HighwayLayer")
        
        ### Add RNN Encoder Layer
        rnn_encoder = RNNEncoder(self.hidden_encoder_size, self.prob_dropout)
        context_hidden_layer = rnn_encoder.add_layer(self.context_embed, self.context_mask, scopename="EncoderLayer")
        question_hidden_layer = rnn_encoder.add_layer(self.question_embed, self.question_mask, scopename="EncoderLayer")
        
        
        ### Add Attention Layer using BiDAF
        attention_layer = BidafAttention(2*self.hidden_encoder_size, self.prob_dropout)
        combination_cq = attention_layer.add_layer(context_hidden_layer, self.context_mask, question_hidden_layer, self.question_mask, scopename = "BiDAFLayer")
        hidden_BiDAF = RNNEncoder(self.hidden_bidaf_size, self.prob_dropout)
        # The final BiDAF layer is the output_hidden_BiDAF
        output_hidden_BiDAF = hidden_BiDAF.add_layer(combination_cq, self.context_mask, scopename="BiDAFEncoder")
        

        ### Add Output Layer: Predicting start and end of answer
        final_combination_cq = tf.contrib.layers.fully_connected(output_hidden_BiDAF, num_outputs=self.hidden_full_size)
        
        # Compute start distribution
        # with vs.variable_scope("Start")
        start_layer = SimpleSoftmaxLayer()
        self.start_val, self.start_probs = start_layer.add_layer(final_combination_cq, self.context_mask, scopename="StartSoftmax")

        # Compute end distribution
        # with vs.variable_scope("End")
        end_layer = SimpleSoftmaxLayer()
        self.end_val, self.end_probs = end_layer.add_layer(final_combination_cq, self.context_mask, scopename="EndSoftmax")

        
#         masked_softmax => Function not a clss (Pass masked softmax here)
        # HIghway layer
#             Highway()
#             Highway.add_layer()
#         from Layers import *
        
        
    def add_loss(self):
#         with vs.variable_scope("loss"):
        with tf.variable_scope("loss"):
            # Loss for start prediction
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_val, labels=self.answer_span[:, 0])
            self.loss_start = tf.reduce_mean(loss_start) # Average across batch

            # Loss for end prediction
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_val, labels=self.answer_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end) #Average across batch

            # Total loss
            self.loss = self.loss_start + self.loss_end
