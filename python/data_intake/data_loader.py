#GENERAL IMPORTS
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
# from p_tqdm import p_map, p_imap #TODO remove this and the commented out statements for it
from torch.utils.data import Dataset
import torch
import concurrent.futures
from collections import Counter
import numpy as np
import gensim
import copy


#My packages
from data_intake.processing import *

def get_sample_weights(labels):
    counter = Counter(labels)
    counter = dict(counter)
    for k in counter:
        counter[k] = 1 / counter[k]
    sample_weights = np.array([counter[l] for l in labels])
    return sample_weights




class Helper():
    def __init__(self, args=None):
        self.args = args
        self._process_fn = None
        #TODO make this modular or add arg switches
        self.tokenizer = UrlTokenizer(args)


    def call_helper(self, chunk):
        if not self.args:
            raise RuntimeError("Args must be specified before helper is called")
        return tqdm_helper(args)

    def tqdm_helper(self, chunk):
        chunk = chunk.copy()
        chunk = chunk.sample(frac=1)
        chunk = chunk[~chunk[self.args.text_column].isnull()]
        chunk = chunk[(chunk[self.args.text_column].map(len) > 1)]
        chunk['processed_text'] = (chunk[self.args.text_column]
                                    .map(lambda text: self.process_text(text)))
        texts = chunk['processed_text'].tolist()
        # labels += aux_df[self.args.label_column].astype('int16').tolist()
        labels = chunk[self.args.label_column].tolist()
        chunk['tokens'] = (chunk['processed_text']
                            .map(lambda text: self.tokenizer.tokenize(text)))
        tokens = chunk['tokens'].tolist()

        return texts, labels, tokens

    def process_text(self, text):
        if self._process_fn is not None:
            return self._process_fn(text)
        return text

def load_data(args):
    print(args.data_path)
    chunks = pd.read_csv(args.data_path,
                         usecols=[args.text_column, args.label_column],
                         iterator=True,
                         chunksize=args.chunksize,
                         encoding=args.encoding,
                         nrows=args.max_rows,
                         sep=args.sep)
    texts = []
    labels = []
    tokens = []
    token_lens = []
    helper = Helper(args)

    # for chunk in tqdm(chunks):
    #     text, label = tqdm_helper(chunk, args)
    # results = p_imap(tqdm_helper, chunks, args, num_cpus=args.num_text_processing_threads)
    # p_
    # for pair in results:
    #     print(pair)
    # texts, labels = zip(results)
    
    with Pool(args.num_text_processing_threads) as p:
      for text, label, token in  list(tqdm(p.imap(helper.tqdm_helper, chunks))):
          texts += text
          labels += label
          tokens += token
        #   print(token)
        #   token_lens.append(len(token))
      p.close()
      p.join()
      p.close()


    # texts, labels = zip(r)
    # for chunk in tqdm(chunks):

    number_of_classes = len(set(labels))

    print(
        'data loaded successfully with {len(texts)} rows and {number_of_classes} labels')
    print('Distribution of the classes', Counter(labels))

    sample_weights = get_sample_weights(labels)


    return texts, labels, tokens, number_of_classes, sample_weights


class EncodedDataset(Dataset):
    def __init__(self, texts, labels, args, tokens=None):
        self.texts = texts
        self.labels = labels
        self.length = len(self.texts)
        self.tokens = tokens
        self.use_char_encoding = args.use_char_encoding


        if self.use_char_encoding:
            self.vocabulary = args.alphabet + args.extra_characters
            self.number_of_characters = args.number_of_characters + \
                len(args.extra_characters)
            self.identity_mat = np.identity(self.number_of_characters)
        else:
            #TODO switch cases for differnt delims'
            if args.debug:
                print("DEBUG: initializing token word2vec model")

            self.tokens = tokens
            self.tokenizer = UrlTokenizer(args)
            # see https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py for attributes & use
            self.embedding_size = args.embedding_size    # Word vector dimensionality  
            self.embedding_window = args.embedding_window          # Context window size                                                                                    
            self.min_word_count = args.min_word_count   # Minimum word count                        
            self.model = gensim.models.Word2Vec(self.tokens, size=self.embedding_size, 
                          window=self.embedding_window, min_count=self.min_word_count, workers=4, iter=50)



            print(self.model)
            

            if args.debug:
                print("DEBUG: word2vec model initialized")


                similar_words = {search_term: [item[0] for item in self.model.wv.most_similar([search_term], topn=5)]
                  for search_term in ['http', 'com', 'google', 'index', 'php', 'uk']}
                print(similar_words)
            self.vocabulary = self.tokenizer.alphabet
            pass
            #TODO make an ngram encoding
            # self.vocabulary = #TODO
            #self.number_of_characters = #TODO
        #TODO(Stretch) make an encoding with glove?

        self.max_length = args.max_length
        # self.preprocessing_steps = args.steps

    def __len__(self):
        if self.use_char_encoding:
            return self.length
        else:
            return len(self.tokens)

    def __getitem__(self, index):

        if self.use_char_encoding:
            raw_text = self.texts[index]
            data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                            dtype=np.float32) #TODO why is this backwards?
            if len(data) > self.max_length:
                data = data[:self.max_length]
            elif 0 < len(data) < self.max_length:
                data = np.concatenate(
                    (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
            elif len(data) == 0:
                data = np.zeros(
                    (self.max_length, self.number_of_characters), dtype=np.float32)

        else:
            tokens = self.tokens[index]
            data = np.array([self.model.wv[t] for t in tokens[::-1] if t is not None])
            if len(data) > self.max_length:
                data = data[:self.max_length]
            elif 0 < len(data) < self.max_length:
                data = np.concatenate(
                    (data, np.zeros((self.max_length - len(data), self.embedding_size), dtype=np.float32))) #TODO is this the right dtype?
            elif len(data) == 0:
                data = np.zeros(
                    (self.max_length, self.embedding_size), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label

class EncodedStringLabelDataset(Dataset):
    def __init__(self, urls_by_category, args):
        self.args = args
        self.tokenizer = UrlTokenizer(args)
        self._init_private_vars(urls_by_category)
        self._init_embedding()
        # self.selectSubset(labelSubset=args.in_set_labels, normalizeWeights=False)

    def __len__(self):
        if self.args.use_char_encoding:
            return len(self.texts)
        else:
            return len(self.tokens)

    def __getitem__(self, index):

        if self.args.use_char_encoding:
            raw_text = self.texts[index]
            data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                            dtype=np.float32) #TODO why is this backwards?
            if len(data) > self.max_length:
                data = data[:self.max_length]
            elif 0 < len(data) < self.max_length:
                data = np.concatenate(
                    (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
            elif len(data) == 0:
                data = np.zeros(
                    (self.max_length, self.number_of_characters), dtype=np.float32)
        else:
            tokens = self.tokens[index]
            data = np.array([self.model.wv[t] for t in tokens[::-1] if t is not None])
            if len(data) > self.max_length:
                data = data[:self.max_length]
            elif 0 < len(data) < self.max_length:
                data = np.concatenate(
                    (data, np.zeros((self.max_length - len(data), self.args.embedding_size), dtype=np.float32))) #TODO is this the right dtype?
            elif len(data) == 0:
                    data = np.zeros(
                        (self.max_length, self.args.embedding_size), dtype=np.float32)

        label = self.string_labels[index]
        data = torch.Tensor(data)
        return data, label

    def selectSubset(self, labelSubset=None, normalizeWeights=False):              
        indToRemove = copy.copy(self.class_indices)
        existing_classes = set(self.string_labels)
        for label in labelSubset:
            if label not in existing_classes:
                raise Exception("Invalid class name in labelSubset: " + label)
            del indToRemove[label]
        indices = []
        for l in [indToRemove[key] for key in indToRemove.keys()]:
            indices.extend(l)
        indices = np.array(indices)
        self.texts = np.delete(np.array(self.texts), indices).tolist()
        self.string_labels = np.delete(np.array(self.string_labels), indices).tolist()
        if self.tokens:
            self.tokens = np.delete(np.array(self.tokens), indices).tolist()

        assert len(self.texts) == len(self.string_labels)
        if self.tokens:
            assert len(self.texts) == len(self.tokens)

        #build new class_indices structure
        min_label_samples = min([len(self.class_indices[i]) for i in labelSubset])
        self.class_indices = {}
        self.counter = Counter()
        new_texts = []
        new_labels = []
        new_tokens = []
        for i, text in enumerate(self.texts):
            label = self.string_labels[i]
            if(normalizeWeights):
                if self.counter[label] == min_label_samples:
                    continue
                new_texts.append(text)
                new_labels.append(label)
                if self.tokens:
                    new_tokens.append(self.tokens[i])
            self.counter[label] +=1
            if label in self.class_indices.keys():
                self.class_indices[label].append(i)
            else:
                self.class_indices[label] = [i]
            if label in self.args.in_set_labels:
                self.labels.append(1)
            else:
                self.labels.append(0)

        if(normalizeWeights):
            self.texts = new_texts
            self.string_labels = new_labels
            if self.tokens:
                self.tokens = new_tokens
        assert len(self.texts) == len(self.string_labels), "impath length %d and imclass length %d " % (len(self.texts), len(self.string_labels))
        if self.tokens:
            assert len(self.texts) == len(self.tokens)
        print("Selected the following distribution: ", self.counter)

    def _init_private_vars(self, urls_by_category):
        texts = []
        labels = []
        tokens = []
        for key in urls_by_category.keys():
            for url in urls_by_category[key]:
                url = url.strip()
                try:
                    token = self.tokenizer.tokenize(url)
                    tokens.append(token)
                    texts.append(url)
                    labels.append(key)
                except:
                    pass

        assert len(texts) == len(labels)
        assert len(tokens) == len(texts)
        self.texts = texts
        self.string_labels = labels
        if not self.args.use_char_encoding:
            self.tokens = tokens
        else:
            self.tokens = None

        #init private vars
        self.labels = []
        self.tokens = tokens
        self.class_indices = {}
        self.counter = Counter()
        self.max_length = self.args.max_length

        #populate class index dictionary and binary labels
        for i, text in enumerate(texts):
            label = self.string_labels[i]
            self.counter[label] += 1
            if label not in self.class_indices.keys():
                self.class_indices[label] = [i]
            else:
                self.class_indices[label].append(i)
            if label in self.args.in_set_labels:
                self.labels.append(1)
            else:
                self.labels.append(0)

    def _init_embedding(self):
        if self.args.use_char_encoding:
            self.vocabulary = self.args.alphabet + self.args.extra_characters
            self.number_of_characters = self.args.number_of_characters + \
                len(self.args.extra_characters)
            self.identity_mat = np.identity(self.number_of_characters)
        else:
            #TODO switch cases for differnt delims'
            if self.args.debug:
                print("DEBUG: initializing token word2vec model")
            # see https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py for attributes & use                                                                                 
            self.model = gensim.models.Word2Vec(self.tokens, size=self.args.embedding_size, 
                        window=self.args.embedding_window, min_count=self.args.min_word_count, workers=4, iter=50)

            print(self.model)
            print(self.counter)

            if self.args.debug:
                print("DEBUG: word2vec model initialized")
                similar_words = {search_term: [item[0] for item in self.model.wv.most_similar([search_term], topn=5)]
                for search_term in ['http', 'com', 'google', 'index', 'php', 'uk']}
                print(similar_words)
            self.vocabulary = self.tokenizer.alphabet
