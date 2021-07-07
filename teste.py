from collections import defaultdict
from lxml import etree
from multiprocessing import Pool, TimeoutError
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean 
from time import time
from tqdm import tqdm

import numpy as np
import json
import os
import string

# define the folder path that contain the data
# FOLDER_PATH = "Define folder path that contain threads folder and test.json"
FOLDER_PATH = "INF8111_2020_fall_tp1_dataset/dataset/"
#FOLDER_PATH = "INF8111_2020_fall_tp1_dataset_subset/dataset/"
PAGE_FOLDER = os.path.join(FOLDER_PATH, 'bug_reports')
# Load the evaluation dataset
test = json.load(open(os.path.join(FOLDER_PATH, "test.json")))

def extract_data_from_page(pagepath):

    parser = etree.HTMLParser()
    #parser = etree.LXMLParser()
    tree = etree.parse(pagepath, parser)
    main = tree.getroot()[1][0][2]
    report_id = 0
    dup_id = 0
    component = ""
    product = ""
    summary = ""
    description = ""
    creation_date = 0
    dictionary = {}
    
    leaf = main.find('.//*[@id="field-value-bug_id"]')
    if (leaf is not None):
        numbers = ""
        for char in leaf[0].text:
            if char.isdigit():
                numbers+=char
        report_id = (int(numbers))

    leaf = main.findall('.//*[@id="field-value-short_desc"]')
    if (leaf is not None):
        summary = " ".join(leaf[-1].text.split())

    leaf = main.find('.//*[@id="product-name"]')
    if (leaf is not None):
        product = " ".join(leaf.text.split())
        
    leaf = main.find('.//*[@id="component-name"]')
    if (leaf is not None):
        component = " ".join(leaf.text.split())
        
    leaf = main.find('.//*[@id="field-value-status-view"]')
    if (leaf is not None):
        if (leaf.text.find("DUPLICATE") > 0):
            numbers = ""
            for char in leaf[0].text:
                if char.isdigit():
                    numbers+=char
            dup_id = (int(numbers))
        else:
            dup_id = None
        
    leaf = main.find('.//*[@data-time]')
    if (leaf is not None):        
        numbers = ""
        for char in str(leaf.attrib).split(':')[-1]:
            if char.isdigit():
                numbers+=char
        creation_date = (int(numbers))
        
    leaf = main.find('.//*[@id="ct-0"]')
        
    for child in leaf.iter('*'):
        if len(child) == 0:
            description = description + child.text
        
    return  {
            "report_id": report_id,
            "dup_id": dup_id,
            "component": component, 
            "product": product, 
            "summary": summary, 
            "description": description, 
            "creation_date": creation_date
}

# Index each thread by its id
index_path = os.path.join(PAGE_FOLDER, 'bug_reports.json')

if os.path.isfile(index_path):
    # Load threads that webpage content were already extracted.
    report_index = json.load(open(index_path))
else:
    # Extract webpage content

    # This can be slow (around 10 minutes). Test your code with a small sample. lxml parse is faster than html.parser
    files = [os.path.join(PAGE_FOLDER, filename) for filename in os.listdir(PAGE_FOLDER)]
    reports = [extract_data_from_page(f) for f in tqdm.tqdm(files)]
    report_index = dict(((report['report_id'], report) for report in reports ))

    # Save preprocessed bug reports
    json.dump(report_index, open(index_path,'w'))
    
def tokenize_space(text):
    """
    Tokenize the tokens that are separated by whitespace (space, tab, newline). 
    We consider that any tokenization was applied in the text when we use this tokenizer.    
    For example: "hello\tworld of\nNLP" is split in ['hello', 'world', 'of', 'NLP']
    """
    if (type(text) == str):
        return (text.lower().split())
    else:
        return None
        
def tokenize_nltk(text):
    """
    This tokenizer uses the default function of nltk package (word_tokenize) to tokenize the text. [https://www.nltk.org/api/nltk.tokenize.html]
    """
    if (type(text) == str):
        return (word_tokenize(text.lower()))
    else:
        return None

def tokenize_space_punk(text):
    """
    This tokenizer replaces punctuation to spaces and then tokenizes the tokens that are separated by whitespace (space, tab, newline).
    """
    if (type(text) == str):
        nopunc = text.translate(str.maketrans('', '', string.punctuation))
        return (nopunc.lower().split())
    else:
        return None   

def filter_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for w in tokens:
        if w not in stop_words: 
            filtered_tokens.append(w) 
    return (filtered_tokens)

def transform_count_bow(X):
    """
    This method preprocesses the data using the pipeline object, relates each token to a specific integer and  
    transforms the text in a vector. Vectors are weighted using the token frequencies in the sentence.

                                X: document tokens. e.g: [['I','will', 'be', 'back', '.'], ['Helllo', 'world', '!'], ['If', 'you', 'insist', 'on', 'using', 'a', 'damp', 'cloth']]

    :return: vector representation of each document (document-term matrix)
    """    
    #matrix_path = os.path.join(FOLDER_PATH, 'bow_matrix.npz')

    #if os.path.isfile(matrix_path):
    #    bow_matrix = load_npz(matrix_path)

    #else:
    sentence_count = 0
    row = np.empty(0)
    col = np.empty(0)
    data = np.empty(0)  
    word_list = []     

    print("Calculating BOW matrix")
    for sentence in tqdm(X):            
        for token in sentence:
            if (token not in word_list):
                word_list.append(token)
            
        sentence_vector = np.zeros(len(word_list))
        for word in sentence:            
            for i,token in enumerate(word_list):
                if token == word:
                    sentence_vector[i]+=1
                    if sentence_vector[i]==1:
                        col = np.append(col,i)
                        row = np.append(row, sentence_count)
        data = np.append(data,sentence_vector[sentence_vector>0])
        sentence_count+=1
        
    bow_matrix = csr_matrix((data, (row, col)), shape=(len(X), len(word_list)))
    #save_npz(matrix_path, bow_matrix)    

    return bow_matrix

def transform_tf_idf_bow(X):
    """
    This method preprocesses the data using the pipeline object, calculates the IDF and TF and 
    transforms the text in vectors. Vectors are weighted using TF-IDF method.

    X: document tokens. e.g: [['I','will', 'be', 'back', '.'], ['Helllo', 'world', '!'], ['If', 'you', 'insist', 'on', 'using', 'a', 'damp', 'cloth']]

    :return: vector representation of each document
    """    
    #matrix_path = os.path.join(FOLDER_PATH, 'tfidf_matrix.npz')

    #if os.path.isfile(matrix_path):
    #    tfidf_matrix = load_npz(matrix_path)

    #else:
    bow_matrix = transform_count_bow(X)
    tfidf_matrix = bow_matrix

    number_of_documents = bow_matrix.shape[0]    

    print("Calculating TFIDF matrix")
    for i, j in tqdm(zip(*bow_matrix.nonzero())):
        tf = (bow_matrix[i,j] / np.diff(bow_matrix.indptr)[i])
            
            ## talvez colocar um np.bincount(bow_matrix.indices)[j] + 1 aqui, para suavizar a divisão
            ## https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
            
            ## ou somar +1 no final
            ## https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf
            
        idf = np.log (number_of_documents / np.bincount(bow_matrix.indices)[j])
        tfidf = tf * idf
        tfidf_matrix[i,j] = tfidf

    #save_npz(matrix_path, tfidf_matrix)    
        
    return tfidf_matrix

def nlp_pipeline(bug_reports, tokenization_type, vectorizer_type, enable_stop_words, enable_stemming):
    """
    Preprocess and vectorize the threads.
    
    bug_reports: list of all bug reports([dict()]).
    tokenization_type: two possible values "space_tokenization" and "nltk_tokenization".
                            - space_tokenization: tokenize_space function is used to tokenize.
                            - nltk_tokenization: tokenize_nltk function is used to tokenize.
                            - space_punk_tokenization: tokenize_space_punk is used to tokenize.
                            
    vectorizer_type: two possible values "count" and "tf_idf".
                            - count: use transform_count_bow to vectorize the text
                            - tf_idf: use transform_tf_idf_bow to vectorize the text
                            
    enable_stop_words: Enables or disables the insignificant stop words removal
    enable_stemming: Enables or disables steming
    
    return: tuple ($p$, $c$, $M$)
    """
    all_thread_ids = []
    X = []
    
    if tokenization_type == 'space_tokenization':
        tkn_func = tokenize_space
    elif tokenization_type == 'nltk_tokenization':
        tkn_func = tokenize_nltk
    elif tokenization_type == 'space_punk_tokenization':
        tkn_func = tokenize_space_punk
    
    product_vocab = defaultdict(int)
    component_vocab = defaultdict(int)
    
    p = np.zeros((len(bug_reports),))
    c = np.zeros((len(bug_reports),))
    
    for idx, report in enumerate(bug_reports):
        product_vocab.setdefault(report['product'], len(product_vocab))
        component_vocab.setdefault(report['component'], len(component_vocab))
        
        p[idx] = product_vocab[report['product']]
        c[idx] = component_vocab[report['component']]
                 
        text = report['summary'] +"\n" + report["description"]
        tkns =  tkn_func(text)
        
        if enable_stop_words:
            tkns = filter_tokens(tkns)
            
        if enable_stemming:
            tkns = (stemmer.stem(tkn) for tkn in tkns)
                    
        X.append(list(tkns))    
    
    if vectorizer_type == 'count':
        vectorizer_func = transform_count_bow
    elif vectorizer_type == 'tf_idf':
        vectorizer_func = transform_tf_idf_bow
        
    X = vectorizer_func(X)
    
    return p, c, X

def rank(query_idx, p, c, X, w1, w2, w3):
    """
    Return a list of reports indexes sorted by similarity of the bug reports (candidates) and new bug report (query)
    Cosine similarity is used to compare bug reports. 
    
    query_idx: query indexes
    p: product values of all bug reports (list)
    c: component values of all bug reports  (list)
    X: textual data representation of all bug reports  (Matrix)
    
    w1: parameter that controls the impact of the product
    w2: parameter  that controls the impact of the component
    w3: parameter  that controls the impact of textual similrity
    
    return: ranked list of indexes. 
    """
    query_p = p[query_idx]
    query_c = c[query_idx]
    query_X = X[query_idx]    
    
    ranking = np.empty(shape=[0,2])
    fpqr = 0
    fcqr = 0
    cos_sim = 0
    
    for i in range(len(p)):
        if i != query_idx:
            if (p[i] == p[query_idx]):
                fpqr = 1
            else:
                fpqr = 0
                
            if (c[i] == c[query_idx]):
                fcqr = 1
            else:
                fcqr = 0
            
            cos_sim = float(cosine_similarity(X[i], X[query_idx])[0])
            
            simqr = w1*fpqr + w2*fcqr + w3*cos_sim

            ranking = np.vstack((ranking,[i,simqr]))            
            ranking = ranking[ranking[:,1].argsort()[::-1]]

    list_of_indexes = [int(x) for x in ranking[:,0].tolist()]
            
    return list_of_indexes

def calculate_map(x):
    res = 0.0
    n = 0.0
    
    for query_id, corrects, candidate_ids in x:
        precisions = []
        for k, candidate_id in enumerate(candidate_ids):
            
            if candidate_id in corrects:
                prec_at_k = (len(precisions) + 1)/(k+1)
                precisions.append(prec_at_k)
                
            if len(precisions) == len(corrects):
                break
                            
        res += mean(precisions)
        n += 1
    
    return res/n            

def eval(tokenization_type, vectorizer, enable_stop_words, enable_stemming, w1=0.1, w2=0.1, w3=2):
    reports = [r for r in report_index.values()]
    report_ids = [r["report_id"] for r in report_index.values()]
    prod_v, comp_v, M = nlp_pipeline(reports, tokenization_type, vectorizer, enable_stop_words, enable_stemming)
    report2idx = dict([(r['report_id'], idx) for idx,r in enumerate(reports)])
    rank_lists = []
    print("Calculating rankings from test set")
    for query_id, corrects in tqdm(test):
        query_idx =  report_ids.index(query_id)
        candidate_idxs = rank(query_idx, prod_v, comp_v, M, w1, w2, w3)
        candidate_ids = [ report_ids[idx] for idx in candidate_idxs]                
        rank_lists.append((query_id, corrects, candidate_ids))

        
    return calculate_map(rank_lists)
    
print(eval("space_tokenization", "count", False, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("nltk_tokenization", "count", False, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "count", False, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "count", True, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "count", True, True, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "tf_idf", False, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "tf_idf", True, False, w1=0.1, w2=0.1, w3=2 ))
print(eval("space_punk_tokenization", "tf_idf", True, True, w1=0.1, w2=0.1, w3=2 ))