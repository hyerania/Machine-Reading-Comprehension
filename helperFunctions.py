import tensorflow as tf

def masked_softmax(logits, mask, dim):
        """
        Performs softmax over a dim for logits with mask
        Returns:
            masked_logits : returns logits with -large value where there is a padding
            prob_dist : softmax distribution with 0 at places of padding
        """
    
        exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
        prob_dist = tf.nn.softmax(masked_logits, dim)
        return masked_logits, prob_dist
    
def matrix_multiplication(mat1, mat2):
        """
        Multiplies 3D matrix, mat1 to 2D matrix, mat2
        Input:
            mat1: 3D matrix -> (i, j, k)
            mat2: 2D matrix -> (k, l)
        Output:
            Output: 3D matrix -> (i,j,l)
        """
        mat1_shape = mat1.get_shape().as_list()  #[i, j, k]
        mat2_shape = mat2.get_shape().as_list()  #[k, l]
        mat1_reshape = tf.reshape(mat1, [-1, mat1_shape[-1]])  #[batch_size * n, m]
        mul = tf.matmul(mat1_reshape, mat2)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat1_shape[1], mat2_shape[-1]])  # reshape to batch_size, seq_len, p
    

def create_char_dicts(CHAR_PAD_ID=0, CHAR_UNK_ID = 1, _CHAR_PAD = '*', _CHAR_UNK = '}' ):
        """
        Creates the char to id dictionaries for char embedding
        """

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


def padded_char_ids(batch, token_ids, id2word, word_len): 
        """
        Return char ID representation for each batch
        Input : 
            token_ids - [batch, seq_len]
            id2word : id to word dictionary
            word_len : max word length allowed
        Output:
            charids_batch = [batch, seq_len, word_len]
        
        """

        charids_batch = []
        for i in range(batch.batch_size):
            charids_line = []
            token_row = token_ids[i,:]
            for j in range(len(token_row)):
                id = token_row[j]
                word = id2word[id] # convert token id to word
                _, char_ids = word_to_token_ids(word)
                # for each word we get char_ids but they maybe different_length
                if len(char_ids) < word_len: #pad with CHAR pad tokens
                    while len(char_ids) < word_len:
                        char_ids.append(0)
                    pad_char_ids = char_ids

                else:
                    pad_char_ids = char_ids[:word_len]

                charids_line.append(pad_char_ids)
            charids_batch.append(charids_line)

        return charids_batch #[batch, seq_len, word_len]