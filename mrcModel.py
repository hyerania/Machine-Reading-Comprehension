# coding: utf-8
import json
import os
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from layers import Highway, RNNEncoder, BidafAttention, SimpleSoftmaxLayer
from helperFunctions import masked_softmax
import logging
from batcher import get_batch_generator

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
        # with vs.variable_scope("embedding"):
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
            
    def run_train_iter(self, session, batch):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)
        Inputs:
          session: TensorFlow session
          batch: a Batch object
        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        #Placeholders used a keys for creating the input_feed dictionary
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        #####MISSING attribute------------------------------------------------------------------
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        ########MISSING attribute---------------------------------------------------------------
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        return loss, global_step, param_norm, gradient_norm   
    
    
            
    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.
        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        """
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))
        
        # We will keep track of exponentially-smoothed loss
        exp_loss = None
        """

        """
        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        """
        epoch = 0

        #logging.info("Beginning training loop...")
        #WHYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY????
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            #epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len):

                # Run training iteration
                #iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch)
                #iter_toc = time.time()
                #iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()
        
