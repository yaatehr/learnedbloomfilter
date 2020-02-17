import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from ascii_regression_classifier import *

import json
from collections import Counter
import pandas as pd
from tqdm import tqdm

def gen_random_len_string(n=10):
    return ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = n))



def build_features(num_features, string_len):
    random_strings = list(set([gen_random_len_string(string_len) for i in range(num_features)]))
    # print(random_strings)
    labels = np.array([(1 if random.randint(0,1) else 0) for i in range(len(random_strings))])
    ascii_vectors = np.array([[ord(c) for c in string] for string in random_strings])
    return np.array(random_strings), labels, ascii_vectors

def build_case_separated_features(num_features, string_len, noise_ratio):
    in_strings = list(set([gen_random_len_string(string_len) for i in range(num_features)]))
    # print(random_strings)
    out_strings = list(set([''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = string_len)) for i in range(num_features)]))
    
    labels = np.vstack((np.ones((len(in_strings),1)), np.zeros((len(out_strings) ,1))))
    # random boolean mask for which values will be changed
    mask = np.random.choice([0,1],size=labels.shape, p=((1-noise_ratio), noise_ratio)).astype(np.bool)

    # random matrix the same shape of your data
    r = np.random.randint(2, size=labels.shape)

    # use your mask to replace values in your input array
    labels[mask] = r[mask]
    all_strings = in_strings + out_strings
    ascii_vectors = np.array([[ord(c) for c in string] for string in all_strings])
    print(ascii_vectors.shape, labels.shape)
    return np.array(all_strings), labels, ascii_vectors
    

def topNError(output, labels, ns, percent=True):
    sortedOutputs = output.topk(k = max(ns), dim=1, sorted=True)[1]
    topNs = torch.cat((sortedOutputs, labels.view(labels.shape + (1,))), dim=1)
    results = [[row[-1] in row[:n] for row in torch.unbind(topNs, dim=0)] for n in ns]
    errors = [len(res) - np.sum(res) for res in results]
    return np.array([error/len(labels) for error in errors] if percent else errors)

def confusionMatrix(outputs, labels):
    return confusion_matrix(labels, torch.argmax(outputs, dim=1))

def saveErrorGraph(trainErrors, valErrors, outfile):
    trainClassificationErrors, trainTop2Errors = trainErrors[:,0], trainErrors[:,1]
    valClassificationErrors, valTop2Errors = valErrors[:,0], valErrors[:,1]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors')
    epochs = np.arange(1, trainErrors.shape[0] + 1)
    plt.plot(epochs, trainClassificationErrors, label="Train Classification Error")
    plt.plot(epochs, trainTop2Errors, label="Train Top 2 Error")
    plt.plot(epochs, valClassificationErrors, label="Validation Classification Error")
    plt.plot(epochs, valTop2Errors, label="Validation Top 2 Error")
    plt.legend(loc='best')
    plt.savefig(outfile)



class AsciiStringData(Dataset):
    def __init__(self, num_features, string_len):
        super(Dataset, self).__init__()
        self.Strings, self.Y, self.X = build_features(num_features, string_len)
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # print(self.X)
        # print(self.Y)
        # print(idx)
        ascii_vec = self.X[idx, :]
        # print(ascii_vec)
        label = self.Y[idx]
        return ascii_vec, label


class AsciiStringDataCaps(Dataset):
    def __init__(self, num_features, string_len, noise_ratio=.05):
        super(Dataset, self).__init__()
        self.Strings, self.Y, self.X = build_case_separated_features(num_features, string_len, noise_ratio)
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # print(self.X)
        # print(self.Y)
        # print(idx)
        ascii_vec = self.X[idx, :]
        # print(ascii_vec)
        label = self.Y[idx]
        return ascii_vec, label

dataset_size = 1000

ten_feature_ascii_set = AsciiStringData(dataset_size, 10)
twentyfive_feature_ascii_set = AsciiStringData(dataset_size, 25)
fifty_feature_ascii_set = AsciiStringData(dataset_size, 50)

datasets = [ten_feature_ascii_set, twentyfive_feature_ascii_set, fifty_feature_ascii_set]

def get_dataset(i):
    return datasets[i%3]



def write_dataset(filepath):
    strings, labels, asci_vec = build_case_separated_features(100000, 25, .3)
    n,d = asci_vec.shape
    with open(filepath, 'w+') as outfile:
        for i in range(n):
            for j in range(d):
                outfile.write(str(asci_vec[i][j]))
        

def tokenize(args, input_text):
    tokenized = args.compiled_url_regex.split(input_text)
    if len(tokenized) > args.max_tokens:
        return tokenized[:args.max_tokens].join(" ")
    return tokenized.join(" ")



def load_data(args):
    # chunk your dataframes in small portions
    chunks = pd.read_csv(args.data_path,
                         usecols=[args.text_column, args.label_column],
                         chunksize=args.chunksize,
                         encoding=args.encoding,
                         nrows=args.max_rows,
                         sep=args.sep)
    texts = []
    labels = []
    for df_chunk in tqdm(chunks):
        aux_df = df_chunk.copy()
        aux_df = aux_df.sample(frac=1)
        aux_df = aux_df[~aux_df[args.text_column].isnull()]
        aux_df = aux_df[(aux_df[args.text_column].map(len) > 1)]
        aux_df['processed_text'] = (aux_df[args.text_column]
                                    .map(lambda text: tokenize(args, text)))
        texts += aux_df['processed_text'].tolist()
        labels += aux_df[args.label_column].astype('int16').tolist()


    number_of_classes = len(set(labels))

    print(
        f'data loaded successfully with {len(texts)} rows and {number_of_classes} labels')
    print('Distribution of the classes', Counter(labels))

    return texts, labels, number_of_classes


class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.texts = texts
        self.labels = labels
        self.length = len(self.texts)

        self.vocabulary = args.alphabet
        self.number_of_characters = args.number_of_characters
        self.max_length = args.max_length
        self.max_tokens = args.max_tokens
        self.identity_mat = np.identity(self.number_of_characters)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]

        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.max_length, self.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label
