import json
import nltk
import numpy as np
import os
from tqdm import tqdm
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize


### Preprocessing Data
def load_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def tokenize_data(string):
    tokens = [token.replace("''", '" ').replace("``", '" ').lower() for token in nltk.word_tokenize(string)]
    return tokens

def get_Word_Index(context, context_tokens):
    result = ''
    current_word_index = 0
    wordMap = dict()

    for char_index, char in enumerate(context):
        if char !='\n' and char !=' ':
            result += char
            context_token = context_tokens[current_word_index]
            if result == context_token:
                start = char_index - len(result) + 1
                for char_position in range(start, char_index+1):
                    wordMap[char_position] = (result, current_word_index)
                result = ''
                current_word_index += 1
                
    if current_word_index != len(context_tokens):
        return None
    else:
        return wordMap

def preprocess(dataset, datatype, jsonDir):
    num_map_problem = 0
    num_token_problem = 0
    num_align_problem = 0
    num_examples = 0
    examples = []
    
    for eventID in tqdm(range(len(dataset['data']))):
        event_Paragraphs = dataset['data'][eventID]['paragraphs']
        for paragraphID in range(len(event_Paragraphs)):
            # Context Data
            context = event_Paragraphs[paragraphID]['context']
            context = context.replace("''", '" ').replace("``", '" ').lower()
            context_tokens = tokenize_data(context)
            
            wordIndex = get_Word_Index(context, context_tokens)
            if wordIndex is None:
                num_map_problem += len(event_Paragraphs[paragraphID]['qas'])
                continue
            
            # Question and Answer Data
            qaSet = event_Paragraphs[paragraphID]['qas']
            for qID in qaSet:
                question = qID['question']
                question = question.replace("''", '" ').replace("``", '" ').lower()
                question_tokens = tokenize_data(question)
                
                ans_text = qID['answers'][0]['text']
                ans_text = ans_text.lower()
                ans_start_index = qID['answers'][0]['answer_start']
                ans_end_index = ans_start_index + len(ans_text)
                
                if context[ans_start_index:ans_end_index] != ans_text:
                    num_align_problem += 1
                    continue
                
                ans_start_word = wordIndex[ans_start_index][1]
                ans_end_word = wordIndex[ans_end_index-1][1]
                ans_tokens = context_tokens[ans_start_word:ans_end_word+1]
                if("".join(ans_tokens) != "".join(ans_text.split())):
                    num_token_problem += 1
                    continue
                
                examples.append((" ".join(context_tokens), " ".join(question_tokens), " ".join(ans_tokens), " ".join([str(ans_start_word), str(ans_end_word)])))
                num_examples += 1
    
    # Creating files for context, questions, answers, and answer span indexes
    index = list(range(len(examples)))
    np.random.shuffle(index)
    with open(os.path.join(jsonDir, datatype +'.context'), 'w', encoding="utf-8") as context_file,           open(os.path.join(jsonDir, datatype +'.question'), 'w', encoding="utf-8") as question_file,         open(os.path.join(jsonDir, datatype +'.answer'), 'w', encoding="utf-8") as answer_file,          open(os.path.join(jsonDir, datatype +'.span'), 'w', encoding="utf-8") as span_file:
        
        for i in index:
            (context, question, answer, span_index) = examples[i]
            context_file.write(context + '\n')
            question_file.write(question + '\n')
            answer_file.write(answer + '\n')
            span_file.write(span_index + '\n')
    
    # Returning results
    print ("Number of triples ignored due to token mapping problems: ", num_map_problem)
    print ("Number of triples ignored due to unalignment with tokenization problems: ", num_token_problem)
    print ("Number of triples ignored due to span alignment problems: ", num_align_problem)
    print ("Processed examples: %i out of %i" % (num_examples, num_examples+num_map_problem+num_token_problem+num_align_problem))


### GloVe Vectors
# NEED TO BE REPLACED WITH STUTI'S FILE
def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.
    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path
    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _START_VOCAB = [_PAD, _UNK]
    PAD_ID = 0
    UNK_ID = 1

    print ("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r', encoding="utf-8") as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return id2word, word2id, emb_matrix