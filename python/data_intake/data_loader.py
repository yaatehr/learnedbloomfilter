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
        self.selectSubset(labelSubset=None, balanceWeights=True) #balance binary classifications

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

    def selectSubset(self, labelSubset=None, balanceWeights=False):
        if labelSubset:              
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
            self.labels = np.delete(np.array(self.labels), indices).astype(int).tolist()
            if self.tokens:
                self.tokens = np.delete(np.array(self.tokens), indices).tolist()

            self._check_assertions()

        #build new class_indices structure
        num_in_inset = np.sum(self.labels)
        num_in_outset = 0 #counter to balance weights
        self.class_indices = {}
        self.counter = Counter()
        new_texts = []
        new_string_labels = []
        new_labels = []
        new_tokens = []
        for i, text in enumerate(self.texts):
            label = self.string_labels[i]
            if(balanceWeights):
                if not self.labels[i]:
                    num_in_outset +=1
                elif num_in_outset >= num_in_inset:
                    continue
                new_texts.append(text)
                new_string_labels.append(label)
                if self.tokens:
                    new_tokens.append(self.tokens[i])
            self.counter[label] +=1
            if label in self.class_indices.keys():
                self.class_indices[label].append(i)
            else:
                self.class_indices[label] = [i]
            if self.labels[i]:
                new_labels.append(1)
            else:
                new_labels.append(0)

        if(balanceWeights):
            self.texts = new_texts
            self.string_labels = new_string_labels
            self.labels = new_labels
            if self.tokens:
                self.tokens = new_tokens
            num_positive_labels = np.sum(self.labels)
            num_negative_labels = (len(self.labels) - num_positive_labels)
            num_labels_to_add = num_positive_labels - num_negative_labels
            print("DEBUG: Difference in labels %d" % num_labels_to_add)
            if num_labels_to_add != 0:
                if num_labels_to_add > 0:
                    self._add_negative_samples(num_labels_to_add)
                else:
                    self._remove_negative_samples(num_labels_to_add*-1)
            assert num_positive_labels == (len(self.labels) - num_positive_labels), "number of positive samples %d should match negative %d" % (num_positive_labels, len(self.labels) - num_positive_labels)
        self._check_assertions()
        print("Selected the following distribution: ", self.counter)
        print("With %d positive labels and %d negative labels" % (num_positive_labels, (len(self.labels) - num_positive_labels)))

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

        self.texts = texts
        self.string_labels = labels
        if not self.args.use_char_encoding:
            self.tokens = tokens
        else:
            self.tokens = None

        #init private vars
        self.tokens = tokens
        self.max_length = self.args.max_length

        #populate class index dictionary and binary labels
        self._init_class_indices_and_counter(build_labels=True)

        self._check_assertions()

    def _init_embedding(self):
        if self.args.use_char_encoding:
            self.vocabulary = self.args.alphabet + self.args.extra_characters
            self.number_of_characters = self.args.number_of_characters + \
                len(self.args.extra_characters)
            self.identity_mat = np.identity(self.number_of_characters)
        elif self.args.use_word2vec_encoding:
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
        else:
            pass
            # raise Exception("not implemented")
            #TODO add glove embedding
            

    def _check_assertions(self):
        assert len(self.texts) == len(self.string_labels), "texts length %d and string_labels length %d " % (len(self.texts), len(self.string_labels))
        assert len(self.texts) == len(self.labels), "texts length %d and binary labels length %d " % (len(self.texts), len(self.labels))
        if self.tokens:
            assert len(self.texts) == len(self.tokens), "texts length %d and tokens length %d " % (len(self.texts), len(self.tokens))
        num_indexes = 0
        for l in self.class_indices.values():
            num_indexes += len(l)
        assert len(self.texts) == num_indexes, "num texts %d must match num class indexes % d" % (len(self.texts), num_indexes)
 
    def _add_negative_samples(self, num_to_add):
        #TODO make this process more clear (ie rename the input path param)
        print(len(self.texts))
        print(sum(self.labels))
        texts, labels, tokens, number_of_classes, sample_weights = load_data(self.args)
        print("length of all loaded texts is", len(texts))
        assert len(texts) >= num_to_add, "insufficient urls loaded to pad inputs, need %d more" % (num_to_add - len(texts))
        #TODO extract only the negative things
        indices = np.random.choice(len(texts), len(texts) - num_to_add, replace=False)

        print(num_to_add)
        texts = np.delete(np.array(texts), indices).tolist()
        string_labels = ["misc"]*num_to_add
        labels = np.delete(np.array(labels), indices).astype(int).tolist()
        if self.tokens:
            tokens = np.delete(np.array(tokens), indices).tolist()

        print(len(texts), len(string_labels), len(labels), len(tokens))

        self.texts.extend(texts)
        self.string_labels.extend(string_labels)
        self.labels.extend(labels)
        if self.tokens:
            self.tokens.extend(tokens)

        misc_indexes = range(len(self.texts), len(self.texts) + num_to_add)

        if "misc" in self.class_indices.keys():
            self.class_indices["misc"].extend(misc_indexes)
        else:
            self.class_indices["misc"] = list(misc_indexes)

        self._check_assertions()

    def _remove_negative_samples(self, num_to_remove):
        #TODO make this process more clear (ie rename the input path param)

        negative_sample_indices = []
        for key in self.class_indices.keys():
            if key in self.args.in_set_labels:
                continue
            for class_index in self.class_indices[key]:
                negative_sample_indices.append(class_index)

        indices = np.random.choice(negative_sample_indices, num_to_remove, replace=False)

        self.texts = np.delete(np.array(self.texts), indices).tolist()
        self.string_labels = np.delete(np.array(self.string_labels), indices).tolist()
        self.labels = np.delete(np.array(self.labels), indices).astype(int).tolist()
        if self.tokens:
            self.tokens = np.delete(np.array(self.tokens), indices).tolist()

        self._init_class_indices_and_counter()

        self._check_assertions()

    def _init_class_indices_and_counter(self, build_labels=False):
        self.class_indices = {}
        self.counter = Counter()
        if build_labels:
            self.labels = []
        for i, text in enumerate(self.texts):
            label = self.string_labels[i]
            self.counter[label] +=1
            if label in self.class_indices.keys():
                self.class_indices[label].append(i)
            else:
                self.class_indices[label] = [i]

            if build_labels:
                if label in self.args.in_set_labels:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
