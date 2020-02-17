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
        f'data loaded successfully with {len(texts)} rows and {number_of_classes} labels')
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

