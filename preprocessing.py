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