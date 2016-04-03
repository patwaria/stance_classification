#!/usr/bin/python

from __future__ import division, unicode_literals, print_function

import os
import sys
import string
import nltk
import pprint
from scipy.sparse import csr_matrix

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score, jaccard_similarity_score
import numpy as np

import pickle
import json

from textblob import TextBlob

from nltk.parse.stanford import StanfordDependencyParser
from collections import defaultdict
import re

import os

java_path = "C:/Program Files/Java/jre1.8.0_77/bin"  # replace this
os.environ['JAVAHOME'] = java_path



def prettyPrint(data):
    pprint.pprint(data)

def remove_punct(text):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in text if ch not in exclude)
    return s


def get_sentiment(text):
    t = TextBlob(text)
    return t.sentiment.polarity, t.sentiment.subjectivity


jarpath = "D://ML//stanford-parser//stanford-parser.jar"
modelpath = "D://ML//stanford-parser//stanford-parser-3.6.0-models.jar"
parser = StanfordDependencyParser(path_to_jar=jarpath, path_to_models_jar=modelpath,java_options='-mx10000m',)

def dependency_parse(text):
    t = TextBlob(text)

    # print(t.raw_sentences)
    dependencies = []
    for sent in t.raw_sentences:
        result = parser.raw_parse(sent)
        for res in result:
            for (head, rel, tail) in res.triples():
                dep = (head[0], rel, tail[0])
                # print (dep)
                dependencies.append(dep)
    print (dependencies)
    return dependencies
    # prettyPrint ([list(res.triples()) for res in result])


def read_file(filepath):
    if not (os.path.isfile(filepath)):
        return None
    with open(filepath, encoding='utf8') as f:
        content = f.readlines()
        return content


def parse_meta(meta):
    # print (meta)
    id = int(meta[0].strip('\n').split('=')[1])
    parent = int(meta[1].strip('\n').split('=')[1])
    try:
        stance = int(meta[2].strip('\n').split('=')[1])
    except ValueError:
        stance = +1

    rebuttal = (meta[3].strip('\n').split('=')[1])
    return {'id': id, 'pid': parent, 'stance': stance, 'rebuttal': rebuttal}


def create_model(counts, maxfeat=1000, minfreq=3):
    model = {}
    fdist = nltk.FreqDist(counts)
    for (item, freq) in fdist.most_common(maxfeat):
        if freq > minfreq:
            model[item] = freq
    return model


def extract_model_features(model, tokens):
    features = {x: 0 for x in model.keys()}
    for token in tokens:
        if token in features:
            features[token] += 1
    return features


def extract_individual_features(data):
    # features taken from Anand et al.
    # 1. Unigram freqs, bigram freqs,
    # 2. Cue words: Intitial unigram, bigram, trigram
    # 3. sentiment polarity, subjectivity
    for c in string.ascii_uppercase:
        if len(data[c]) == 0: continue
        for i in range(1, 500):
            if i in data[c]:
                features = {}
                unigram_feat = extract_model_features(model=data['unigram_model'], tokens= data[c][i]['unigrams'])
                bigram_feat = extract_model_features(model=data['bigram_model'], tokens=data[c][i]['bigrams'])
                polarity, subjectivity = get_sentiment(data[c][i]['text'])

                # print (unigram_feat)
                # print (bigram_feat)
                features.update(unigram_feat)
                features.update(bigram_feat)
                rebuttal = data[c][i]['meta']['rebuttal']

                reb = 0
                if rebuttal == 'oppose':
                    reb = -1
                elif rebuttal == 'support':
                    reb = 1
                features.update({('sentiment_value', ):polarity, ('subjectivity_value', ): subjectivity,
                                 ('rebuttal_value', ): reb})

                # pid = data[c][i]['meta']['pid']
                data[c][i]['features'] = features
                # prettyPrint(features)
    return data


def read_data(type):
    datapath = '../data/' + type + '/'
    data = {}
    maxindex = 500
    count = 0
    unigrams = []
    bigrams = []
    dependecies = []
    for c in string.ascii_uppercase:
        data[c] = {}
        for i in range(1, maxindex):
            filename = datapath + c + str(i)
            txtpath = filename + '.data'
            metapath = filename + '.meta'
            text = read_file(txtpath)

            meta = read_file(metapath)
            if text is not None:
                count += 1
                # print (count)
                data[c][i] = {'text': text[0], 'meta': parse_meta(meta)}
                tokens = nltk.word_tokenize(text[0])

                data[c][i]['tokens'] = tokens
                data[c][i]['length'] = len(tokens)
                s = remove_punct(text[0])
                tokens = nltk.word_tokenize(remove_punct(s.lower()))

                data[c][i]['unigrams'] = list(nltk.ngrams(tokens, 1))
                data[c][i]['bigrams'] = list(nltk.ngrams(tokens, 2))

                # data[c][i]['dependencies'] = dependency_parse(text[0])
                # deppath = filename + '.dep'
                # with open (deppath, 'w') as f:
                #     json.dump(data[c][i]['dependencies'],f)
                # with open (deppath, 'r') as f:
                #     data[c][i]['dependencies'] = json.load(f)


                unigrams.extend(data[c][i]['unigrams'])
                bigrams.extend(data[c][i]['bigrams'])
                # dependecies.extend(data[c][i]['dependencies'])

        data[c]['sequences'] = gen_sequences(data[c])
        data['unigram_model'] = create_model(unigrams, maxfeat=5000, minfreq=3)
        data['bigram_model'] = create_model(bigrams, maxfeat=5000, minfreq=3)
        # data['dependencies'] = create_model(dependecies, maxfeat=5000, minfreq=3)

    # pprint.pprint (data['unigram_model'])
    # pprint.pprint (data['bigram_model'])
    # pprint.pprint (data['dependencies'])

    # print(type, count)
    return data


def dfs(root, children, sequences, seq):
    newseq = seq + [root]
    if len(children[root]) == 0:
        # print(newseq)
        sequences.append(newseq)
        return
    for child in children[root]:
        dfs(child, children, sequences, newseq)


def gen_sequences(thread):
    n = len(thread)
    # print (n)
    roots = []
    children = {}
    for ind in thread:
        post = thread[ind]
        id = post['meta']['id']
        pid = post['meta']['pid']
        if pid in children:
            children[pid].append(id)
        if pid == -1:
            roots.append(id)
        children[id] = []

    sequences = []
    for root in roots:
        # dfs to create sequences
        dfs(root, children, sequences, [])

    # print(sequences)
    return sequences


def read_authors(type):
    datapath = '../data/authors/' + type + '/'
    authors = defaultdict(list)
    for c in string.ascii_uppercase:
        filename = datapath + c + '.author'
        # print(filename)
        lines = read_file(filename)
        if lines == None: continue
        for line in lines:
            line = line.strip('\n')
            line = line.split(" ")
            id, author = line[0], line[1]
            authors[author].append(id)

    # prettyPrint(authors)
    return authors

def read_folds(type):

    datapath = '../data/folds/' + type + '_folds/'
    folds = []
    for i in range(1, 6):
        filepath = datapath + 'Fold-' + str(i)
        fold = read_file(filepath)
        fold = [f.strip('\n') for f in fold]
        folds.append(fold)

    return folds

def parse_filename(name):

    name = name.strip('\n')
    c = name[0]
    ind = int(name[1:])
    return (c, ind)

def prep_svm_struct_datfiles(type, data, folds, test_fold = 1):

    print ('Prepping type ' + type + '...')

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    train_seqid = []
    test_seqid = []
    train_file_id = []
    test_file_id = []
    seqid = 0
    for c in string.ascii_uppercase:
        sequences = data[c]['sequences']
        # if root is in test then whole sequence in test
        for seq in sequences:
            seqid += 1
            # print (len(seq))
            start = c + str(seq[0])
            # print (start)
            test = False
            if start in folds[test_fold]:
                test = True
                # print ("Found in test")
            for id in seq:
                stance = data[c][id]['meta']['stance']
                if stance == -1:
                    stance = 2
                if test == False:
                    X_train.append(data[c][id]['features'])
                    y_train.append(stance)
                    train_seqid.append(seqid)
                    train_file_id.append(c + str(id))
                else:
                    X_test.append(data[c][id]['features'])
                    y_test.append(stance)
                    test_seqid.append((seqid))
                    test_file_id.append(c + str(id))

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.fit_transform(X_test)

    # print (X_train.nonzero())
    row, col = X_train.nonzero()
    with open(type + '_train.dat', 'w') as f:

        for i in range(len(X_train)):
            line = ""
            line += str(y_train[i]) # stance class
            line += ' qid:' + str(train_seqid[i])
            for j in range(len(X_train[i])):
                if X_train[i][j] != 0:
                    line += " " + str(j + 1) + ":" + str(X_train[i][j])
            f.write(line+'\n')

    with open(type + '_test.dat', 'w') as f:
        for i in range(len(X_test)):
            line = ""
            line += str(y_test[i]) # stance class
            line += ' qid:' + str(test_seqid[i])
            for j in range(len(X_test[i])):
                if X_test[i][j] != 0:
                    line += " " + str(j + 1) + ":" + str(X_test[i][j])
            f.write(line + '\n')

    # print (X_test.nonzero())
    print ("Done")
    return (test_file_id, y_test)

import subprocess
def run_process(args):
    fnull = open(os.devnull, 'w')
    subprocess.call(args, stdout=fnull, stderr=fnull, shell=False)

def learn_classify_svm_structured(type, c, e, test_file_id, y_true):

    args = "svm_hmm_learn.exe -c " + str(c) + " -e 0.5 " + type + "_train.dat " + type + ".model"
    run_process(args)

    args = "svm_hmm_classify.exe " + type + "_test.dat " + type + ".model " + type + "_output"
    run_process(args)


    lines = read_file(type + "_output")

    test_ids = {}
    for i, id in enumerate(test_file_id):
        if id in test_ids: continue
        test_ids[id] = {}
        test_ids[id]['1'] = 0
        test_ids[id]['2'] = 0
        test_ids[id]['true'] = y_true[i]
        test_ids[id]['pred'] = -1

    for i, line in enumerate(lines):
        res = line.strip('\n')
        test_ids[test_file_id[i]][res] += 1

    correct = 0.0
    for id in test_ids:
        pred = 2
        if test_ids[id]['1'] > test_ids[id]['2']:
            pred = 1
        if pred == test_ids[id]['true']:
            correct += 1.0
        test_ids[id]['pred'] = pred

    accuracy = correct / len(test_ids)
    # print ('SSVM Accuracy: ', accuracy)
    return test_ids, accuracy

def learn_classify__svm_individual(data, folds, test_fold=4):

    test_folds = [0, 1, 2, 3, 4]

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in test_folds:
        if i == test_fold: continue
        for name in folds[i]:
            c, ind = parse_filename(name)
            X_train.append(data[c][ind]['features'])
            y_train.append(data[c][ind]['meta']['stance'])

    for i in test_folds:
        if i != test_fold: continue
        for name in folds[i]:
            c, ind = parse_filename(name)
            X_test.append(data[c][ind]['features'])
            y_test.append(data[c][ind]['meta']['stance'])

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.fit_transform(X_test)

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def author_constraint_results(authors, test_ids):

    correct = 0.0
    for a in authors:
        ids = authors[a]
        stance1 = 0; stance2 = 0
        for id in ids:
            if id in test_ids:
                if test_ids[id]['pred'] == 1:
                    stance1 += 1
                else:
                    stance2 += 1

        res = 1
        if stance2 > stance1:
            res = 2
        for id in ids:
            if id in test_ids:
                if test_ids[id]['true'] == res:
                    correct += 1.0

    accuracy = correct / (len(test_ids))
    # print('SSVM + AC Accuracy: ', accuracy)
    return accuracy


def main():
    types = ['abortion', 'gayRights', 'marijuana', 'obama']


    for type in types:
        # if type != 'abortion': continue

        print ('--------- ' + type+  ' --------------')
        data = read_data(type)
        data = extract_individual_features(data)
        folds = read_folds(type)
        authors = read_authors(type)
        test_folds = [0, 1, 2, 3, 4]
        for test_fold in test_folds:
            print ('Fold', test_fold)
            svmacc = learn_classify__svm_individual(data,folds, test_fold)
            print("SVM: ", svmacc)
            test_file, y_true = prep_svm_struct_datfiles(type, data, folds, test_fold)

            maxauthacc = 0.0; maxssvmacc = 0.0
            for c in range(1, 10, 1):
                # print ('C: ' , c)
                test_ids, ssvm_acc = learn_classify_svm_structured(type, c, 0.5, test_file, y_true)
                auth_acc = author_constraint_results(authors, test_ids)
                if auth_acc > maxauthacc:
                    maxauthacc = auth_acc
                    maxssvmacc = ssvm_acc
            print("SSVM: ", maxssvmacc)
            print("SSVM+Auth: ", maxauthacc)


if __name__ == '__main__':
    main()
