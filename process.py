import random

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from transformers import BertModel,BertTokenizer, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sentence_transformers import SentenceTransformer
import torch
from sklearn import preprocessing

def normalize(data_preprocessed):
    scaler = preprocessing.StandardScaler()
    data_preprocessed = scaler.fit_transform(data_preprocessed)
    return data_preprocessed
def w2v(sentences, mxlen):
    # tagged_documents = [TaggedDocument(sentence, [i]) for i, sentence in enumerate(sentences)]
    # model = Doc2Vec(tagged_documents, vector_size=mxlen, window=5, min_count=1, workers=4, epochs=40)
    # embeddings = []
    # for sentence in sentences:
    #     embeddings.append(model.infer_vector(sentence))
    # return np.array(embeddings)
    model = Word2Vec(hs=1, min_count=1, window=5, vector_size=100)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=5)
    embeddings = []
    for sentence in sentences:
        embedding = []
        for word in sentence:
            embedding.append(model.wv.get_vector(word).sum())
        if len(embedding) < mxlen:
            embedding.extend([0] * (mxlen - len(embedding)))
        embeddings.append(embedding)
    return np.array(embeddings)

def TFIDF(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # feature_names = vectorizer.get_feature_names()
    tfidf_array = tfidf_matrix.toarray()
    return tfidf_array

def GloVe(sentences):
    def load_glove_embeddings(file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    glove_file_path = 'glove/glove.6B.300d.txt'
    embeddings_index = load_glove_embeddings(glove_file_path)

    def get_word_vector(word, embeddings_index):
        vector = embeddings_index.get(word)
        if vector is not None:
            return vector
        else:
            return np.zeros(100)
    def log_to_vector(sentence, embeddings_index):
        word_vectors = [np.mean(get_word_vector(word.lower(), embeddings_index), axis=0) for word in sentence]
        # print(np.mean(word_vectors, axis=0).shape)
        return np.mean(word_vectors, axis=0)

    ret = np.zeros((len(sentences), 100))
    for i, sentence in enumerate(sentences):
        ret[i] = log_to_vector(sentence, embeddings_index)
    return ret
def BERT(log_data):
    # tokenizer = AutoTokenizer.from_pretrained("./deberta-v3-base")
    # model = AutoModel.from_pretrained("./deberta-v3-base")
    # pipe = pipline("fill-mask", model="microsoft/deberta-v3-base")
    # model = AutoModel.from_pretrained("./deberta")
    model = AutoModel.from_pretrained("../bert-base-uncased")

    sentences = []
    # tokenizer = AutoTokenizer.from_pretrained("./deberta")
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
    for i in log_data:
        sentences.append(str(i))

    rets = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            rets.append(outputs['last_hidden_state'].cpu().detach().numpy()[0].mean(axis=0).reshape(1, -1))
    # predictions = torch.argmax(logits, dim=-1)
    # print(outputs.shape)
    rets = np.concatenate(rets, axis=0)
    return rets
def read(path):
    data = pd.read_csv(path)

    try:
        label = data['label']
    except KeyError:
        label_ = data['Label']
        label = np.zeros(len(label_), int)
        for i in range(len(label_)):
            label[i] = 0 if label_[i] == '-' else 1

    EventTemplate = data['EventTemplate']
    sentences = []
    for event in EventTemplate:
        i = event.split(' ')
        sentences.append(i)
    w2v_data = w2v(sentences, EventTemplate)
    tfidf_data = TFIDF(sentences)
    # glove_data = GloVe(sentences)
    bert_data = BERT(EventTemplate)
    lt = [i for i in range(0, data.shape[0])]
    random.seed(10000)
    random.shuffle(lt)
    w2v_data = w2v_data[lt]
    tfidf_data = tfidf_data[lt]
    bert_data = bert_data[lt]
    label = label[lt]

    return w2v_data, tfidf_data, bert_data, label
    # print(glove_data.shape)
def node_embedding(logs, num=4):
    sentences = []
    mxlen = 0
    for event in logs:
        i = event.split(' ')
        sentences.append(i)
        mxlen = max(mxlen, len(i))
    w2v_data = w2v(sentences, mxlen)
    tfidf_data = TFIDF(logs)
    # glove_data = GloVe(sentences)
    bert_data = BERT(logs)
    # return [normalize(np.array(w2v_data)), normalize(np.array(tfidf_data)), normalize(np.array(bert_data)), normalize(np.array(glove_data))]
    # return [ normalize(np.array(bert_data))]
    return [normalize(np.array(w2v_data)), normalize(np.array(tfidf_data)), normalize(np.array(bert_data))]
    # return [normalize(np.array(w2v_data)), normalize(np.array(tfidf_data))]
    # return [normalize(np.array(w2v_data)), normalize(np.array(bert_data))]
    # return [normalize(np.array(tfidf_data)), normalize(np.array(bert_data))]