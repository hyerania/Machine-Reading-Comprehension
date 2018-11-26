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
    

def create_char_dicts(CHAR_PAD_ID=0, CHAR_UNK_ID = 1, _CHAR_PAD = '*', _CHAR_UNK = '}' ):

        unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                        '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                        'e' , 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        '~', ]  # based on analysis in jupyter notebook

        

        idx2char = dict(enumerate(unique_chars, 2))  ##reserve first 2 spots
        idx2char[CHAR_PAD_ID] = _CHAR_PAD
        idx2char[CHAR_UNK_ID] = _CHAR_UNK

        ##Create reverse char2idx
        char2idx = {v: k for k, v in idx2char.items()}
        return char2idx, idx2char, len(idx2char)

def word_to_token_ids(word):
        """Turns a word into char idxs
            e.g. "know" -> [9, 32, 16, 96]
            Note any token that isn't in the char2idx mapping gets mapped to the id for UNK_CHAR
        """
        char2idx, idx2char, _ =  create_char_dicts()
        char_tokens = list(word)  # list of chars in word
        char_ids = [char2idx.get(w, 1) for w in char_tokens]
        return char_tokens, char_ids


def padded_char_ids(batch, token_ids, id2word, word_len):  # have to use token_ids since only those are padded

        charids_batch = []
        for i in range(batch.batch_size):
            charids_line = []
            #for each example
            token_row = token_ids[i,:]
            # print("Each token row is", token_row)
            # print("Shape token row is ", token_row.shape)
            for j in range(len(token_row)):
                id = token_row[j]
                # print("each id is:" ,id)
                word = id2word[id] # convert token id to word
                _, char_ids = word_to_token_ids(word)
                # for each word we get char_ids but they maybe different_length
                if len(char_ids) < word_len: #pad with CHAR pad tokens
                    while len(char_ids) < word_len:
                        char_ids.append(0)
                    pad_char_ids = char_ids

                else:  # if longer truncate to word max len
                    pad_char_ids = char_ids[:word_len]

                charids_line.append(pad_char_ids)
            charids_batch.append(charids_line)

        return charids_batch