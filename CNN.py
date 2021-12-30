# keras imports
# from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers.normalization import BatchNormalization
# from keras.layers.embeddings import Embedding
# from keras.layers.merge import concatenate

# from keras.callbacks import TensorBoard
# from keras.models import load_model
# from keras.models import Model
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import gc
from operator import itemgetter
from keras.models import load_model
# import pandas as pd
from string import punctuation
from gensim.models import KeyedVectors

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# std imports
import time
import gc
import os



def ner():
    mask = []
    with open("C:/Users/User/PycharmProjects/SEARCH/d/ner.json",encoding="utf-8") as json_file:
        data = json.load(json_file)
        # for p in data['entities']:
        #   if(p['type'] == 'PER'):
        #     mask.append(p['name'])
        for p in data['entities']:
            #   if(p['type'] == 'PER'):
            mask.append(p['name'])
    return mask


def masking(sentence, n):
    sen = sentence.split()
    #   print(sen)
    newb = ""
    for s in sen:
        tr = False
        tr = s in n
        if (tr is True):
            sentence = sentence.replace(s, "")
    return sentence


def pre(text):
    text = stopword.remove(text)
    output = ''.join(c for c in text if not c.isdigit())
    #     output = ' '.join([w.lower() for w in word_tokenize(output)])
    output = output.lower()
    output = ''.join(c for c in output if c not in punctuation)
    return output


def init_lists(folder, n):
    f = open(folder, "r", encoding="utf-8")
    d_list = []
    for line in f:
        a = line.strip().split("\n")
        # sen = masking(a[0],n)
        output = pre(a[0])
        #     sentence=tokenizer.tokenize(a[0])
        #      sentence = [word.lower() for word in sentence
        d_list.append(output)
    # string = string + str(a[0]) + "\n"
    return d_list


def init_pasal(folder):
    f = open(folder, "r")
    d_list = []
    for line in f:
        a = line.strip().split("\n")
        d_list.append(a[0])
    return d_list


def collect(docs, query, pasal, pasald):
    sentence1 = []
    sentence2 = []
    i = 0
    is_similar = []
    state = False

    # make data
    for i in range(len(query)):
        benar = str(pasal[i]).strip().split(",")
        for j in range(len(pasald)):
            sentence1.append(str(query[i]))
            sentence2.append(str(docs[j]))
            state = pasald[j] in benar
            if (state == True):
                is_similar.append('1')
                state = False
            elif (state == False):
                is_similar.append('0')
    return sentence1, sentence2, is_similar


def train_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        embedding_dim (int): outpu wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    #     word_vectors = KeyedVectors.load("/content/gdrive/My Drive/new/d/we/idwiki_cbow_word2vec_100.model.wv.vectors.npy", mmap='r')
    # model = Word2Vec.load("/content/gdrive/My Drive/new/d/we/idwiki_cbow_word2vec_100.model")
    model = Word2Vec.load("/content/gdrive/My Drive/new/d/we/idwiki_cbow_word2vec_50.model")
    #     model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    #     del model
    return word_vectors


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector

    Returns:

    """
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def word_embed_meta_data(documents, embedding_dim):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
        embedding_dim (int): embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [x.lower().split() for x in documents]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents)
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0].lower() for x in test_sentences_pair]
    test_sentences2 = [x[1].lower() for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    #     leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
    #                   for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    #     leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2


def _count_performance(rel, rtv, precisions, recalls):
    #     stored value corectness 0 / 1
    matched = []
    # true positif
    tp = 0
    # boolean of match
    match = False
    for a in rtv:
        for b in rel:
            if (a == b):
                match = True
                continue
        if (match == True):
            matched.append(1)
            tp = tp + 1
            #       print(str(tp))
            match = False
        else:
            matched.append(0)

    precision = float(tp / len(rtv))
    recall = float(tp / len(rel))
    precisions = precisions + float(tp / len(rtv))
    recalls = recalls + float(tp / len(rel))

    return precision, recall, precisions, recalls


def Nmaxelements(list1, rank):
    #     for i in list1:
    #       print(i)
    ranking = np.argsort(-list1) + 1
    #     print("ppppppp")
    #     for i in ranking:
    #       print (i)
    rtv = []
    r = 0
    for idx in ranking:
        if (r < rank):
            #     r= int(ranking[i])
            p = int(int(idx) - 1)
            #       print(p)
            rtv.append(str(pasald[p]))
            r = r + 1

    return rtv


def experiment(preds, r, jum, query, pasal):
    count = 0
    rank = r
    pred_q = []
    precisions = 0
    recalls = 0
    c = 0
    nil = 0
    # print("pred"+str(len(preds)))
    for j in range(len(preds)):
        #   print(len(preds))
        pred_q.append(float(preds[j]))
        c = c + 1
        # jum = 97
        if (c % 9409 == 0):
            # print(str(j) +"===================")
            avg = np.zeros(jum)
            counter = 0

            # jika diaverage per kolom

            for k in range(len(pred_q)):
                if (counter <= jum - 1):
                    avg[counter] = avg[counter] + pred_q[k]
                    counter = counter + 1
                    if (counter == jum - 1):
                        counter = 0
            for l in range(len(avg)):
                avg[l] = avg[l] / jum

                # jika diambil hanya nilai representasi dokumen

                # for l in range(len(avg)):
                #   if(l == 0):
                #     avg[l]= nil
                #   else:
                #     nil = nil + jum + 1
                #     avg[l] = nil
                # # print(avg)
                # for k in range(len(pred_q)):
                #   # print(k)
                #   if(k == int(avg[counter])):
                #     # print(avg[counter])
                #     # print(pred_q[k])
                #     avg[counter] = pred_q[k]
                #     counter = counter + 1




                #     print(str(j))
            #       pred_q.append(preds[j])
            #     print(pred_q)
            #     for i in range(len(pred_q)):
            #       print(str(i) + " = "+ str(pred_q[i]))
            # print("Q" + str(count) + " : " + str(query[count]))
            rel = str(pasal[count]).strip().split(",")
            #       rtv=[]
            #  asc

            # a = np.array(pred_q)
            # print(len(avg))
            #     index_doc = (a).argsort()[-rank:]
            rtv = Nmaxelements(avg, rank)
            #     print("---------------------------")
            #     print(ini)
            #     print(indexing)
            #   np.argsort(-pred_q)[-rank:]
            #     print(index_doc)
            pred_q = []
            #       for k in index_doc:
            #         rtv.append(pasald[k])
            # print("pasal retrived : " + str(rtv))
            # print("pasal relevan : " + str(rel))
            recall = 0
            precision = 0
            precision, recall, precisions, recalls = _count_performance(rel, rtv, precisions, recalls)
            # print("precision : "+ str(precision) +"\trecall : " + str(recall))
            count = count + 1
            #     elif(j % jum-1 != 0):

    map = float(precisions / len(query))
    mar = float(recalls / len(query))
    f1 = float(2 * ((map * mar) / (map + mar)))

    # print("\n Maka \nMean Average Precision : " + str(map) + "\nMean Average Recall : "+ str(mar)+ "\nF1 score : " + str(f1))

    return mar

def init():
    embedding_dim = 50
    max_sequence_length = 30
    #  number_lstm_units = 50
    #  rate_drop_lstm = 0.3
    #  number_dense_units = 50
    #  activation_function = 'tanh'
    #  rate_drop_dense = 0.2
    #  validation_split_ratio = 0.1

    sentences1, sentences2, is_similar = collect(docs, query, pasal, pasald)
    ######## Word Embedding ############
    tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2, embedding_dim)
    embedding_meta_data = {
        'tokenizer': tokenizer,
        'embedding_matrix': embedding_matrix
    }
    ## creating sentence pairs
    test_sentence_pairs = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
    # del sentences1
    #  del sentences2
    # print(test_sentence_pairs)
    jum = int(len(pasald))
    model = load_model('C:/Users/User/PycharmProjects/SEARCH/CNN/100NER',custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
    # test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]
    test_data_x1, test_data_x2 = create_test_data(tokenizer,test_sentence_pairs,  max_sequence_length)
    preds = list(model.predict([test_data_x1, test_data_x2], verbose=1).ravel())
    r =experiment(preds, a[i], jum, query, pasal)

    return r
# print("lagi mulai-----")
# a = [5,10,20,30,40,50,60,70,80,90,97]
# # a = [5]
# j=[]
# # print(preds)
# for i in range(len(a)):
#   r =experiment(preds, a[i], jum, query, pasal)
#   j.append(r)
# for k in j:
#   print(round(k,4))

n = ner()
docs = init_lists('C:/Users/User/PycharmProjects/SEARCH/d/listed_doc.txt', n)
query = init_lists('C:/Users/User/PycharmProjects/SEARCH/d/qtest_20.txt', n)
pasal = init_pasal('C:/Users/User/PycharmProjects/SEARCH/d/pqtest_20.txt')
# pasaln = init_pasal('/content/gdrive/My Drive/new/d/20/pntrain_20.txt')
pasald = init_pasal('C:/Users/User/PycharmProjects/SEARCH/d/listed_pasal.txt')
def init():