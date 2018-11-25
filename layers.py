import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from helperFunctions import masked_softmax, matrix_multiplication


class RNNEncoder():
    """
    Module for Bidirectional Encoder
    Uses GRU
    """
    
    def __init__(self, size, dropout_param):
        """
        Inputs:
          size: int. Hidden size of the RNN
          dropout_param: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.size = size
        self.dropout_param = dropout_param
        
    def add_layer(self, inputs, masks, scopename):
        """
        Inputs:
          inputs: seq matrix, shape: (batch_size, seq_len, embedding_dim)
          masks: shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.
        Returns:
          out: Tensor of  shape (batch_size, seq_len, hidden_size*2).
        """
        with vs.variable_scope(scopename, reuse=tf.AUTO_REUSE):
            
            rnn_cell_fw = rnn_cell.GRUCell(self.size)
            # rnn_cell_fw = rnn_cell.LSTMCell(self.size)
            rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.dropout_param)
            rnn_cell_bw = rnn_cell.GRUCell(self.size)
            # rnn_cell_bw = rnn_cell.LSTMCell(self.size)
            rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.dropout_param)
            
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # input length for each batch
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            output_RNN = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            output_RNN = tf.nn.dropout(output_RNN, self.dropout_param)

            return output_RNN

class BidafAttention():
    """
    Module for Bidaf Attention
    https://arxiv.org/pdf/1611.01603.pdf
    """
    def __init__(self, size, dropout_param):
        """
        Inputs: 
            size: Dimension of encoded embeddings (2*hidden states)
            dropout_param:  Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.dropout_param = dropout_param
        self.size = size
        
    def add_layer(self, context_batch, context_mask, question_batch, question_mask, scopename ):
        """
        Inputs:
          context_batch: context matrix, shape: (batch_size, num_context_words = N, size)
          context_mask: shape (batch_size, N)
            1s where there's real input, 0s where there's padding
          question_batch: question matrix, shape: (batch_size, num_question_words = M, size)
          question_mask: shape (batch_size, M).
            1s where there's real input, 0s where there's padding
        Outputs:
          output: Tensor of shape (batch_size, N, size*3 = hidden_size*6)
            This is the attention output.
        """
        with vs.variable_scope(scopename, reuse=tf.AUTO_REUSE):
            #parameter to learn
            S_W = tf.get_variable('S_W', [self.size*3], tf.float32, tf.contrib.layers.xavier_initializer())
             
            # Calculating similarity matrix
            c_expand = tf.expand_dims(context_batch,2)  #[batch,N,1,2h]
            q_expand = tf.expand_dims(question_batch,1)  #[batch,1,M,2h]
            c_pointWise_q = c_expand * q_expand  #[batch,N,M,2h]

            c_input = tf.tile(c_expand, [1, 1, tf.shape(question_batch)[1], 1]) #[batch, N, M, 2h]
            q_input = tf.tile(q_expand, [1, tf.shape(context_batch)[1], 1, 1]) #[batch, N, M, 2h]

            concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [batch,N,M,6h]
            
            similarity=tf.reduce_sum(concat_input * S_W, axis=3)  #[batch,N,M]
            
            # Calculating context to question attention
            similarity_mask = tf.expand_dims(question_mask, 1) # [batch, 1, M]
            _,c2q_dist = masked_softmax(similarity, similarity_mask, 2) # [batch, N, M]
            c2q = tf.matmul(c2q_dist, question_batch) #[batch, N, 2h]
            
            # Calculating question to context attention
            S_max = tf.reduce_max(similarity, axis = 2)
            _,q2c_dist = masked_softmax(S_max, context_mask, 1) # [batch, N]
            q2c_dist_expand = tf.expand_dims(q2c_dist, 1) # [batch, 1, N]
            q2c = tf.matmul(q2c_dist_expand, context_batch) # [batch_size, 1, 2h]
            
            #Combining c2q and q2c with context_batch
            context_c2q = context_batch * c2q # [batch, N, 2h]
            context_q2c = context_batch * q2c # [batch, N, 2h]
            
            #Concatenating to get the final output
            output_Bidaf = tf.concat([context_batch, c2q, context_c2q, context_q2c], axis=2) # (batch, N, 8h)
            
            # Apply dropout
            output_Bidaf = tf.nn.dropout(output_Bidaf, self.dropout_param)

            return output_Bidaf
            
            
            
             
        
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def add_layer(self, inputs, masks, scopename):
        """
        Reduces the dimensionality to 1 using a single layer, then softmax
        Inputs:
          inputs: shape: (batch_size, seq_len, hidden_size_fully_connected)
          masks: shape: (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.
        Outputs:
          logits: Tensor of shape (batch_size, seq_len)
            logits is the result of the fully connected layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor of  shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope(scopename):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

        
class Highway():
    """
    Module for highway layer 
    https://arxiv.org/pdf/1507.06228.pdf
    """
    def __init__(self, size, transform_bias):
        """
        Inputs:
          size: dimension of word embeddings
          transform_bias: Intial bias for transform gate
        """
        self.size = size
        self.transform_bias = transform_bias
        self.bias = 0.1
        
    def add_layer(self, inputs, scopename):
        """Inputs:
            inputs: seq matrix, shape: (batch_size, seq_len, embedding_dim)
            scopename : name of scope
          Outputs:
            Tensor of shape (batch_size, seq_len, embedding_dim)
            Size remains same. This is the highway layer output
        """
        with tf.variable_scope(scopename, reuse=tf.AUTO_REUSE):
            #Weights and Bias for Transform Gate
            W_T = tf.Variable(tf.truncated_normal([self.size, self.size], stddev=0.1), name="weight_transform")
            b_T = tf.Variable(tf.constant(self.transform_bias, shape=[self.size]), name="bias_transform")

            #Weights and Bias for Activation
            W = tf.Variable(tf.truncated_normal([self.size, self.size], stddev=0.1), name="weight")
            b = tf.Variable(tf.constant(self.bias, shape=[self.size]), name="bias")
            
            T = tf.sigmoid(matrix_multiplication(inputs, W_T) + b_T, name="transform_gate")
            H = tf.nn.relu(matrix_multiplication(inputs, W) + b, name="activation")
            C = tf.subtract(1.0, T, name="carry_gate")
            #print("shape H, T: ", H.shape, T.shape)
        
            output_highway = tf.add(tf.multiply(H, T), tf.multiply(inputs, C), "output_highway")
            return output_highway
        
        
class BasicAttentionLayer(object):
    """Module for basic attention.
    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".
    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.
    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.
        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            print("Basic attn keys", keys.shape)
            print("Basic attn values", values_t.shape)
            print("Basic attn logits", attn_logits.shape)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output 