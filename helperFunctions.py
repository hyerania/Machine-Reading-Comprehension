import tensorflow as tf

def masked_softmax(logits, mask, dim):
        """
        Takes masked softmax over the given input
        Inputs:
          logits: Numpy array
          mask: Numpy array of same shape as logits
            Has 1s where there's real data in logits, 0 where there's padding
          dim: int. dimension over which to take softmax
        Returns:
          masked_logits: Numpy array same shape as logits
            This is the same as logits, but with 1e30 subtracted
            (i.e. very large negative number) in the padding locations.
          prob_dist: Numpy array same shape as logits.
            The result of taking softmax over masked_logits in given dimension.
            Should be 0 in padding locations.
            Should sum to 1 over given dimension.
        """
        exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
        prob_dist = tf.nn.softmax(masked_logits, dim)
        return masked_logits, prob_dist
    
def matrix_multiplication(mat, weight):
        """
        Multiplies 3D matrix, mat to 2D matrix, weight
        Input:
            mat: 3D matrix -> (i, j, k)
            weight: 2D matrix -> (k, l)
        Output:
            Output: 3D matrix -> (i,j,l)
        """
        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p