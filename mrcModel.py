# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from layers import Highway, RNNEncoder, BidafAttention, SimpleSoftmaxLayer, BasicAttentionLayer, CharEmbedding
from helperFunctions import create_char_dicts, padded_char_ids
import logging
from batcher import get_batch_generator
from official_evaluation import f1_score, exact_match_score

### Model
class mrcModel(object):
    def __init__(self, id2word, word2id, embed_matrix, CharCNN, Highway, Bidaf):
        
        #Model Params
        self.CharCNN = CharCNN
        self.Highway = Highway
        self.Bidaf = Bidaf
        
        print('Model Params Initialised')
        
        ### Hyperparameters:
        #Sizes of Nodes, batches etc
        self.hidden_bidaf_size = 150 #RNN after Bidaf hidden units
        self.hidden_encoder_size = 150 #RNN encoder hidden units
        self.hidden_full_size = 200 #Fully connected layer size after RNN encoding of bidaf
        self.context_len = 300 #Max number of words in context
        self.question_len = 30 #Max number of words in question
        self.batch_size = 60 #Batch size
        self.num_epochs = 20
        
        print('HyperParameters Initialised')
        
        #Learning parameters
        self.max_gradient_norm = 5.0 #Param for gradient Clipping
        self.learning_rate = 0.0008 #Learning rate
        self.dropout = 0.75 #Drop out for RNN encoder layer, keep prob
        print('epochs: ', self.num_epochs)
        print('learning rate: ', self.learning_rate)
        print('keep_prob: ', self.dropout)
        
        print('Learning Initialised')
        
        #Saving model parameters
        self.train_dir = './train' #Directiory to save the model
        self.print_every = 5 #To print log
        self.save_every = 500 #To save the model
        self.eval_every = 500 #To evaluate the dev set
        print('Model Save parameters Initialised')
        # embed_size = 100

        self.id2word = id2word #Dictionary for mapping id to word
        self.word2id = word2id #Dictionary for mapping word to id
        
        
        #Parameters for Char embeddings
        if self.CharCNN:
            print('Char CNN parameters Initialised')
            _, _, self.char_vocab = create_char_dicts() #Char vocab 
            self.char_embedding_size = 8 #Size of char embedding
            self.word_len = 16 #maximum word length
            self.char_out_size = 100 #output size after cnn
            self.window_width = 5 #kernel size for 1D convolution
        
        
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders() #Add the inputs(which dont require gradients)
            self.add_embed_layer(embed_matrix) #Layer to get the embeddings
            if self.CharCNN:
                self.add_char_embed_layer()
            self.create_layers() #Add the required layers
            self.add_loss() #Loss layer

            
        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables() #gets all the learning parameters 
        gradients = tf.gradients(self.loss, params) #Get gradient of loss with respect to the learning parameters
        self.gradient_norm = tf.global_norm(gradients) #Calculates the norm of all gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm) #Clip the gradients which are very high
        self.param_norm = tf.global_norm(params) #Calculates the norm of all params

        # Define optimizer and updates
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)#Update the weights
        
        # Define savers (for checkpointing)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        
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
        
        ## For Char CNN
        if self.CharCNN:
            self.char_ids_context = tf.placeholder(tf.int32, shape=[None, self.context_len, self.word_len])
            self.char_ids_question = tf.placeholder(tf.int32, shape=[None, self.question_len, self.word_len])
        print('Placeholders Defined')
        
    def add_embed_layer(self, embed_matrix):
        with tf.variable_scope("embedding"):
            embedding_matrix = tf.constant(embed_matrix, dtype=tf.float32, name="embed_matrix") #[400002, 100]
            self.context_embed = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) #[batch_size, context_len, 100]
            self.question_embed = embedding_ops.embedding_lookup(embedding_matrix, self.question_ids) #[batch_size, question_len, 100]
        print("Embed Layer Defined")   
            
    def add_char_embed_layer(self):
        char_embedding = CharEmbedding(self.char_vocab, self.char_embedding_size, self.word_len, self.char_out_size, self.window_width, self.dropout)
        context_emb_out = char_embedding.add_layer(self.char_ids_context, scopename = 'char_embed') #[batch, context_len, 100]
        question_emb_out = char_embedding.add_layer(self.char_ids_question, scopename = 'char_embed') #[batch, ques_len. 100]
        self.context_embed = tf.concat((self.context_embed, context_emb_out), axis = 2) #[batch, context_len, 200]
        self.question_embed = tf.concat((self.question_embed, question_emb_out), axis=2) #[batch, ques_len, 200]
        print("Char Embed Layer Defined")
    
    
    def create_layers(self):
        ### Add highway layer
        if self.Highway:
            embed_size = self.context_embed.get_shape().as_list()[-1] #[100] / [200 if char encoding]
            high_way = Highway(embed_size, -1.0)
            for i in range(2):
                self.context_embed = high_way.add_layer(self.context_embed, scopename = "HighwayLayer") #[batch_size, context_len, 100]
                self.question_embed = high_way.add_layer(self.question_embed, scopename = "HighwayLayer") #[batch_size, ques_len, 100]
            print("Highway Layer Defined")
               
        ### Add RNN Encoder Layer
        rnn_encoder = RNNEncoder(self.hidden_encoder_size, self.prob_dropout) 
        context_hidden_layer = rnn_encoder.add_layer(self.context_embed, self.context_mask, scopename="EncoderLayer") #[batch_size, context_len, 150]
        question_hidden_layer = rnn_encoder.add_layer(self.question_embed, self.question_mask, scopename="EncoderLayer") #[batch_size, context_len, 150]
        print("RNN encoder Layer Defined")
        
        ### Add Attention Layer using BiDAF
        if self.Bidaf:
            attention_layer = BidafAttention(2*self.hidden_encoder_size, self.prob_dropout) 
            combination_cq = attention_layer.add_layer(context_hidden_layer, self.context_mask, question_hidden_layer, self.question_mask, scopename = "BiDAFLayer") #[batch_size, context_len, 1200]
            hidden_BiDAF = RNNEncoder(self.hidden_bidaf_size, self.prob_dropout)
            # The final BiDAF layer is the output_hidden_BiDAF
            output_hidden_attention = hidden_BiDAF.add_layer(combination_cq, self.context_mask, scopename="BiDAFEncoder")#[batch, context_len, 150]
            print("Bidaf Layer Defined")
        else:
             # Perform baseline dot product attention
             last_dim = context_hidden_layer.get_shape().as_list()[-1]
             attention_layer = BasicAttentionLayer(self.prob_dropout, last_dim, last_dim)
             _, attn_output = attention_layer.add_layer(question_hidden_layer, self.question_mask, context_hidden_layer)  #[batch_size, context_len, hidden_size*2]
             output_hidden_attention = tf.concat([context_hidden_layer, attn_output], axis=2)  # [batch_size, context_len, hidden_size*4]
             print("Basic Attention Layer Defined")


        ### Add Output Layer: Predicting start and end of answer
        final_combination_cq = tf.contrib.layers.fully_connected(output_hidden_attention, num_outputs=self.hidden_full_size) #[batch, context_len, 200]
        
        # Compute start distribution
        start_layer = SimpleSoftmaxLayer()
        self.start_val, self.start_probs = start_layer.add_layer(final_combination_cq, self.context_mask, scopename="StartSoftmax")

        # Compute end distribution
        end_layer = SimpleSoftmaxLayer()
        self.end_val, self.end_probs = end_layer.add_layer(final_combination_cq, self.context_mask, scopename="EndSoftmax")
        print("Output Layer Defined")
        
    def add_loss(self):
        print("Loss Defined")
        with tf.variable_scope("loss"):
            # Loss for start prediction
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_val, labels=self.answer_span[:, 0])
            self.loss_start = tf.reduce_mean(loss_start) # Average across batch

            # Loss for end prediction
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_val, labels=self.answer_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end) #Average across batch

            # Total loss
            self.loss = self.loss_start + self.loss_end
            
    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path, CharCNN):
        """
        Get loss for entire dev set.
        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files
        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        loss_per_batch, batch_lengths = [], []

        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.batch_size, context_len=self.context_len, question_len=self.question_len, discard_examples=True):

            # Get loss for this batch
            loss = self.run_iter(session, batch, mode = 'dev_loss', CharCNN = CharCNN)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        
        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss        
            
            
            
    def run_iter(self, session, batch, mode, CharCNN):
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
        input_feed[self.question_ids] = batch.qn_ids
        input_feed[self.question_mask] = batch.qn_mask
        if CharCNN:
            input_feed[self.char_ids_context] = padded_char_ids(batch, batch.context_ids, self.id2word, self.word_len)
            input_feed[self.char_ids_question] = padded_char_ids(batch, batch.qn_ids,  self.id2word, self.word_len)
        if mode == "train":
            input_feed[self.answer_span] = batch.ans_span
            input_feed[self.prob_dropout] = self.dropout # apply dropout

            # output_feed contains the things we want to fetch.
            output_feed = [self.updates, self.loss, self.global_step, self.param_norm, self.gradient_norm]
            # Run the model
            [_, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)
            return loss, global_step, param_norm, gradient_norm
        elif mode == "dev_loss":
            input_feed[self.answer_span] = batch.ans_span
            output_feed = [self.loss]
            [loss] = session.run(output_feed, input_feed)
            return loss
        elif mode == "emScore" or mode == "f1Score":
            output_feed = [self.start_probs, self.end_probs]
            [probdist_start, probdist_end] = session.run(output_feed, input_feed)
            return probdist_start, probdist_end
            
            
    
    
            
    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path, spanMode, CharCNN):
        """
        Main training loop.
        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """
        
        # We will keep track of exponentially-smoothed loss
        exp_loss = None
        checkpoint_path = os.path.join(self.train_dir, "qa.ckpt")
        epoch = 0

        while epoch < self.num_epochs:
            epoch += 1

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.batch_size, context_len=self.context_len, question_len=self.question_len, discard_examples = True):

                # Run training iteration
                loss, global_step, param_norm, grad_norm = self.run_iter(session, batch, mode = 'train', CharCNN = CharCNN)

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.print_every == 0:
                    logging.info('epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm))

                # Sometimes save model
                if global_step % self.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.eval_every == 0:

                    # Get loss for entire dev set
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path, CharCNN = CharCNN)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))

                    # Get F1/EM on train set
                    logging.info("Calculating Train F1/EM...")
                    train_f1 = self.calc_f1(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000, spanMode=spanMode, CharCNN = CharCNN)
                    train_em = self.calc_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000, spanMode=spanMode, CharCNN = CharCNN)
                    
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
 
                    # Get F1/EM on dev set
                    logging.info("Calculating Dev F1/EM...")
                    dev_f1 = self.calc_f1(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0, spanMode=spanMode, CharCNN = CharCNN)
                    dev_em = self.calc_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0, spanMode=spanMode, CharCNN = CharCNN)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    logging.info("End of epoch %i" % (epoch))

    
    ### HELPER FUNCTIONS
    def calc_f1(self, session, context_path, question_path, answer_path, data_name, num_samples, spanMode, CharCNN):
        '''
        Calculate the F1 Score and returen the average for all or only a certain number of samples
        Inputs:
            session: current Tensorflow session
            context_path, qustion_path, answer_path: Path of actual data files
            data_name: For log file, define if using train or dev set
            num_samples: If 0, use the entire dataset, else use only the specificed number as a subset of the data
            spanMode: True boolean uses smart span selection of positions, otherwise use basic selection
            charCNN: True boolean uses char embedding, False uses GLoVe vectors
        Returns:
        F1 average score
        '''
        f1_total = 0
        example_num = 0
        for batch in get_batch_generator(self.word2id, context_path, question_path, answer_path, self.batch_size, context_len=self.context_len, question_len = self.question_len, discard_examples = False):
            start_index_pred, end_index_pred = self.get_index(session, batch, "f1Score", spanMode = spanMode, CharCNN = CharCNN)
            start_index_pred = start_index_pred.tolist()
            end_index_pred = end_index_pred.tolist()

            for id, (start_answer_pred, end_answer_pred, answer_tokens) in enumerate(zip(start_index_pred, end_index_pred, batch.ans_tokens)):
                example_num += 1

                #Find the predicted answer
                answer_tokens_pred = batch.context_tokens[id][start_answer_pred: end_answer_pred + 1]
                answer_pred = " ".join(answer_tokens_pred)

                #Find the ground truth answer
                answer_truth = " ".join(answer_tokens)

                #Calculate F1 Score using official evaluation methods
                current_f1 = f1_score(answer_pred, answer_truth)
                f1_total += current_f1

                # Tests if using all the dataset or only a sample
                if(example_num >= num_samples and num_samples != 0):
                    break
            if(example_num >= num_samples and num_samples != 0):
                break

        f1_total = f1_total/example_num
        logging.info("F1 %s: %i examples got a score of %.5f" % (data_name, example_num, f1_total))
        return f1_total
    
    def calc_em(self, session, context_path, question_path, answer_path, data_name, num_samples, spanMode, CharCNN):
        '''
        Calculate the EM Score and returen the average for all or only a certain number of samples
        Inputs:
            session: current Tensorflow session
            context_path, qustion_path, answer_path: Path of actual data files
            data_name: For log file, define if using train or dev set
            num_samples: If 0, use the entire dataset, else use only the specificed number as a subset of the data
            spanMode: True boolean uses smart span selection of positions, otherwise use basic selection
            charCNN: True boolean uses char embedding, False uses GLoVe vectors
        Returns:
        EM average score
        '''
        em_total = 0
        example_num = 0
        for batch in get_batch_generator(self.word2id, context_path, question_path, answer_path, self.batch_size, context_len = self.context_len, question_len = self.question_len, discard_examples = False):
            start_index_pred, end_index_pred = self.get_index(session, batch, "emScore", spanMode = spanMode, CharCNN = CharCNN)
            start_index_pred = start_index_pred.tolist()
            end_index_pred = end_index_pred.tolist()

            for id, (start_answer_pred, end_answer_pred, answer_tokens) in enumerate(zip(start_index_pred, end_index_pred, batch.ans_tokens)):
                example_num += 1

                #Find the predicted answer
                answer_tokens_pred = batch.context_tokens[id][start_answer_pred: end_answer_pred + 1]
                answer_pred = " ".join(answer_tokens_pred)

                #Find the ground truth answer
                answer_truth = " ".join(answer_tokens)

                #Calculate Exact Match Score using official evaluation methods
                current_em = exact_match_score(answer_pred, answer_truth)
                em_total += current_em

                # Tests if using all the dataset or only a sample
                if(example_num >= num_samples and num_samples != 0):
                    break
            if(example_num >= num_samples and num_samples != 0):
                break

        em_total = em_total/example_num
        logging.info("Exact Match %s: %i examples got a score: %.5f" % (data_name, example_num, em_total))
        return em_total

    def get_index(self, session, batch, mode, spanMode, CharCNN):
        '''
        Uses forward pass only
        Inputs:
            session: current Tensorflow session
            batch: Batch object
            mode: Describing f1Score or emScore for the run_iter function
            spanMode: True boolean uses smart span selection of positions, otherwise use basic selection
            charCNN: True boolean uses char embedding, False uses GLoVe vectors
        Returns the most likely start and end indexes for the answer for each example
        '''
        start_probs, end_probs = self.run_iter(session, batch, mode, CharCNN = CharCNN)

        if(spanMode == True):
            batch_size = batch.batch_size
            start_index = np.empty(shape=(batch_size), dtype=int)
            end_index = np.empty(shape=(batch_size), dtype=int)
            max_prob = np.empty(shape=(batch_size), dtype=float)
            #Based on context length analysis, 95th percentile are approx. 245 words, with a span of 15 words)
            for i in range(batch_size):
                test_start = 0
                test_end = 0
                test_max = 0
                for j in range(self.context_len - 16):
                    end_subset = end_probs[i, j:j+16]
                    end_max = np.amax(end_subset)
                    end_pos = np.argmax(end_subset)
                    s_prob = start_probs[i, j]
                    new_max = end_max * s_prob

                    if(new_max > test_max):
                        test_start = j
                        test_end = test_start + end_pos
                        test_max = new_max
                
                start_index[i] = test_start
                end_index[i] = test_end
                max_prob[i] = round(test_max, 4)


        else:
            start_index = np.argmax(start_probs, axis=1)
            end_index = np.argmax(end_probs, axis=1)

        return start_index, end_index
        
