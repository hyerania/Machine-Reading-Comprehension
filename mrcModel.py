# coding: utf-8
import json
import os
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs


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
        context_size = self.context_embed.get_shape().as_list()[-1]
        for i in range(2):
            self.context_embed = self.create_highway_layer(self.context_embed, context_size, scope_name = "HighwayLayer", carry_bias = -1.0)
            self.question_embed = self.create_highway_layer(self.question_embed, context_size, scope_name = "HighwayLayer", carry_bias = -1.0)
        
        
        ### Add RNN Encoder Layer
        rnn_encoder = RNNEncoder(self.hidden_encoder_size, self.prob_dropout)
        context_hidden_layer = rnn_encoder.add_layer(self.context_embed, self.context_mask, scopename="EncoderLayer")
        question_hidden_layer = rnn_encoder.add_layer(self.question_embed, self.question_mask, scopenape="EncoderLayer")
        
        
        ### Add Attention Layer using BiDAF
        attention_layer = BidafAttention(self.prob_dropout, 2*self.hidden_encoder_size)
        output_BiDAF = attention_layer.add_layer(question_hidden_layer, self.question_mask, context_hidden_layer, self.context_mask)
        self.attention = tf.reduce_max(outputBiDAF, axis=2)
        #!! See if you can remove the first parameter since we don't use
        _, self.bidaf_probability = masked_softmax(self.attention, self.context_mask, 1)
        combination_cq = tf.concat([context_hidden_layer, output_BiDAF], axis=2)
        
        hidden_BiDAF = RNNEncoder(self.hidden_bidaf_size, self.prob_dropout)
        # The final BiDAF layer is the output_hidden_BiDAF
        output_hidden_BiDAF = hidden_BiDAF.add_layer(combination_cq, self.context_mask, scopename="BiDAFLayer")
        

        ### Add Output Layer: Predicting start and end of answer
        final_combination_cq = tf.contrib.layers.fully_connected(output_hidden_BiDAF, num_outputs=self.hidden_full_size)
        
        # Compute start distribution
        # with vs.variable_scope("Start")
        with tf.variable_scope("Start"):
            start_layer = Softmax()
            self.start_val, self.start_probs = start_layer.add_layer(final_combination_cq, self.context_mask)

        # Compute end distribution
        # with vs.variable_scope("End")
        with tf.variable_scope("End"):
            end_layer = Softmax()
            self.end_val, self.end_probs = end_layer.add_layer(final_combination_cq, self.context_mask)

        
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

    ### HELPER FUNCTIONS for the initialization of the model
#     def masked_softmax():

    # def create_highway_layer(self, x, size, scope_name, carry_bias=-1.0):
    #         W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
    #         b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    #         W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
    #         b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

    #         T = tf.sigmoid(self.highway_multi(x, W_T) + b_T, name="transform_gate")
    #         H = tf.nn.relu(self.highway_multi(x, W) + b, name="activation")
    #         C = tf.subtract(1.0, T, name="carry_gate")

    #         y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
    #         return y
        
    # def highway_multi(self, matrix, weight):
    #     matrixShape = matrix.get_shape().as_list()
    #     weightShape = weight.get_shape().as_list()
    #     matrixTempShape = tf.reshape(matrix, [-1, matrixShape[-1]])
    #     result = tf.matmul(matrixTempShape, weight)
        
    #     return tf.reshape(result, [-1, matrixShape[1], weightShape[-1]])
