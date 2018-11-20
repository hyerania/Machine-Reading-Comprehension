import numpy as np
import random
from embedding import PAD_ID, UNK_ID


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens
        #IS IT REQUIRED????????????????????????????????????????????
        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


def sentence_to_token_ids(sentence, word2id):
    """Turns a sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    Input: 
        sentence: string
        word2id: dictionary for word to id mapping
    Output:
        tokens: list of words in the sentence
        ids : list of ids of each word in the sentence
    """
    ####WHYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY??????????????
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]
    """
    tokens = sentence.strip().split()
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def padded(token_batch, batch_pad):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to.
    Returns:
      List (length batch_size) of padded lists of ints.
    """
    return map(lambda token_list: token_list + [PAD_ID] * (batch_pad - len(token_list)), token_batch)


def add_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, discard_long ):
    """
    Adds batches into the "batches" list.
    Each time it starts from the where it left the files last time and fills in 160 batches into the batches list
    It will in nothing when the EOF has reached
    It will add less than 160 batches when the file has less training samples left
    Inputs:
      batches: list to add batches to, empty list
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
    """
    print("Adding batches start.....................")
    examples = [] # list of (context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens)
    context, ques, ans = context_file.readline(), qn_file.readline(), ans_file.readline() # read the next line from each
    #Each line has a training sample (Context[i], Question[i], Answerspan[i])
    #Each of context, qn, ans are strings at the moment

    while context and ques and ans: # Keep adding till all the training samples are covered

        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context, word2id) #[Number of words in context]
        ques_tokens, ques_ids = sentence_to_token_ids(ques, word2id) #[Number of words in question]
        ans_span = intstr_to_intlist(ans) #[2]

        # read the next line from each file
        context, ques, ans = context_file.readline(), qn_file.readline(), ans_file.readline()

        # get ans_tokens from ans_span
        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] #[Number of words in answer]

        # discard too-long questions
        if len(ques_ids) > question_len:
            if discard_long:
                continue
            else: # truncate
                ques_ids = ques_ids[:question_len]
        
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else: # truncate
                context_ids = context_ids[:context_len]
        
        # Add the training sample to example
        examples.append((context_ids, context_tokens, ques_ids, ques_tokens, ans_span, ans_tokens))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Exits loop if 16 batches have been filled into example or end of file has been reached

    # Sort by question length
    # Note: if you sort by context length, then you'll have batches which contain the same context many times (because each context appears several times, with different questions)
    examples = sorted(examples, key=lambda e: len(e[2]))

    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, ques_ids_batch, ques_tokens_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start:batch_start+batch_size])
        
        # Pad context_ids and ques_ids
        ques_ids = padded(ques_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len

        # Make ques_ids into a np array and create ques_mask
        ques_ids = np.array(ques_ids) # [question_len, batch_size/<batchsize]
        ques_mask = (ques_ids != PAD_ID).astype(np.int32) # [question_len, batch_size/<batchsize]

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids) # [context_len, batch_size/<batchsize]
        context_mask = (context_ids != PAD_ID).astype(np.int32) # [context_len, batch_size/<batchsize]

        # Make ans_span into a np array
        ans_span = np.array(ans_span) # [batch_size/<batchsize, 2]

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, ques_ids, ques_mask, ques_tokens, ans_span, ans_tokens)
        
        batches.append(batch)

    # shuffle the batches
    random.shuffle(batches)
    print("Added ",len(batches)," batches")
    return

def get_batch_generator(word2id, context_path, qn_path, ans_path, batch_size, context_len, question_len, discard_long):
    """This function is a generator. Here is how it runs:
        1. When the function is called first time, none of the code is run, only a generator is returned
        2. Now when this object is used in for loop like, 'for i in generator', the whole function runs till reaches yeild
        3. Every time the state is saved, and it runs from the next iter
        4. The generator stops returning anything when it can no more reach yield
    """
    
    #Open the files for context, question and answer
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    
    batches = []

    while True:
        #When the generator is used for the first time in for loop or when batches gets empty (after 160*batchsize)
        if len(batches) == 0: 
            add_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, discard_long)
        #When the list is empty and we have reached the end of training files, prevent it from going to yield
        if len(batches) == 0:
            break
        
        yield batches.pop(0)

    return














































