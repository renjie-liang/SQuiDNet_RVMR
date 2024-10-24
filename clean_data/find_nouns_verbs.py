from utils.basic_utils import load_json, save_json
import os
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Specify your custom directory
nltk_data_dir = '/red/bianjiang/liang.renjie/.nltk_data'
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)

def find_nouns_verbs(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    
    # Get part of speech tagging for each word
    pos_tags = pos_tag(words)
    
    # Define lists to hold nouns and verbs
    nouns = []
    verbs = []
    
    # NLTK POS tags for nouns and verbs
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    
    # Loop through the tagged words and separate nouns and verbs
    for word, tag in pos_tags:
        if tag in noun_tags:
            nouns.append(word)
        elif tag in verb_tags:
            verbs.append(word)
    
    return nouns, verbs



data = load_json("data/TVR-Ranking/train_top01.json")

for record in tqdm(data):
    query = record['query']
    nouns, verbs = find_nouns_verbs(query)
    record.update({'noun': nouns, 'verb': verbs})

save_json(data, "data/TVR-Ranking/train_top01_noun_predicate.json")