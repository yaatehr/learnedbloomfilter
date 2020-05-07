import json
import os
import pickle
import re
import string
import sys

import requests


def process_text(data):
    # TODO make an actuall text processing thing
    return data


class UrlTokenizer:
    def __init__(self, args, debug=False):
        self.args = args

        # delimiting urls
        self.url_delimeters = "".join(
            set(args.url_delimeters).intersection(set(args.alphabet))
        )
        self.alphabet = "".join(set(args.alphabet).difference(set(args.url_delimeters)))
        self.all_chars = "".join(set(args.alphabet).union(set(args.url_delimeters)))
        if args.debug and debug:
            print("BEGIN UrlTokenizer Debug Info: \n")
            print("inititalizing url delimeters to be: \n ", self.url_delimeters)
            print("inititalizing alphabet to be: \n ", self.alphabet)
            print("END DEBUG INFO \n\n")
        self.max_num_tokens = args.max_embedding_length
        self.regex_str = "[" + self.url_delimeters + "]"

    def tokenize(self, text):
        """
        public function, all tokenizers must have this function
        """
        return self._tokenize_url(text)

    def _tokenize_url(self, string):
        # print(string)
        # print(self.url_delimeters)
        # Filter to remove the empty string tokens
        tokens = [
            x
            for x in re.split(self.regex_str, string, maxsplit=self.max_num_tokens)
            if x
        ]
        # print(tokens)
        if len(tokens) == 0:
            return None  # TODO figure out if none is the best thing to return
        return tokens[0 : self.max_num_tokens]

    def p_tokenize_tuples(self, inputs):
        """
        Take in list of tuples (label, string)
        return list of tuplets 
        """
        pass

    def p_tokenize_tuples(self, inputs):
        """
        Take in list of tuples with
        """
        pass


class WebCrawlExtractor: #TODO make a function of the similarity comparison and use jaccard similarity on tokenization to improve equivalence
    def __init__(
        self, args, max_similar_entries=1, prefix_length=12, current_position=0
    ):
        """
        Note that the prefixes map to a count, You can modify the max 
        number of unique prefixes that can be extracted from the crawl
        """
        self.root = args.root
        self.prefixes = {}
        self.urls = []
        self.max_similar_entries = max_similar_entries
        self.prefix_length = 15
        self.tokenizer = UrlTokenizer(args)
        self.current_position = current_position
        # self.search = re.compile('[^' + self.tokenizer.all_chars +']').search #TODO do we want to filter here?

    def process_crawl(
        self,
        crawl_path="/mnt/hdd1/datasets/web_crawl_urls_v2.txt",
        save_path="python-iap/data_intake/crawl_checkpoint.pkl",
        print_path="input/webcrawl_unique_len_12_prefixes.txt",
    ):
        # filename = os.path.join(self.root, "input/clean_dedup_urls.txt")
        filename = crawl_path
        checkpoint_path = os.path.join(self.root, save_path)
        print_path = os.path.join(self.root, print_path)

        position = 0
        num_urls = 0

        # resume from checkpoint if necessary
        if os.path.exists(checkpoint_path):
            (
                prefixes,
                urls,
                root,
                max_similar_entries,
                prefix_length,
                tokenizer,
                current_position,
                num_urls,
            ) = pickle.load(open(checkpoint_path, "rb"))
            self.prefixes = prefixes
            self.urls = urls
            self.root = root
            self.max_similar_entries = max_similar_entries
            self.prefix_length = prefix_length
            self.tokenizer = tokenizer
            self.current_position = current_position
            write_access = "a"
        else:
            write_access = "w+"

        if not os.path.exists(filename):
            raise Exception("can't open the input file")
        with open(print_path, write_access) as pp:
            with open(filename, "r") as f:
                for position, line in enumerate(f):
                    # fast forward
                    if position < self.current_position:
                        continue

                    # update checkpoints
                    if position % 10000 == 0:
                        print(
                            "Saved {} urls after {} lines, updating checkpoint: {}\n".format(
                                num_urls, position, checkpoint_path
                            )
                        )
                        pickle.dump(
                            (
                                self.prefixes,
                                self.urls,
                                self.root,
                                self.max_similar_entries,
                                self.prefix_length,
                                self.tokenizer,
                                position,
                                num_urls,
                            ),
                            open(checkpoint_path, "wb"),
                        )

                    line = line.strip()
                    prefix = self.gen_prefix(line)
                    if self.contains_prefix(prefix):
                        count = self.prefixes[prefix]
                        if count > self.max_similar_entries:
                            continue
                        else:
                            self.urls.append(line)
                            pp.write(line + "\n")
                            count = self.prefixes[prefix]
                            self.prefixes[prefix] = count + 1
                    else:
                        self.prefixes[prefix] = 1
                        self.urls.append(line)
                        pp.write(line + "\n")

                    num_urls += 1

        print("Crawl Processed!!\n\n")
        print(
            "Saved {} urls after {} lines, updating checkpoint: {}\n".format(
                num_urls, position, checkpoint_path
            )
        )

        pickle.dump(
            (
                self.prefixes,
                self.urls,
                self.root,
                self.max_similar_entries,
                self.prefix_length,
                self.tokenizer,
                position,
                num_urls,
            ),
            open(checkpoint_path, "wb"),
        )
        return self.prefixes, self.urls

    def gen_prefix(self, string):
        if len(string) < self.prefix_length:
            return string
        else:
            return string[: self.prefix_length]

    def contains_prefix(self, prefix):
        return prefix in self.prefixes.keys()

    # def special_match(strg):
    #     return not bool(self.search(strg))


class UrlClassifier:
    def __init__(self, args):
        self.api_key = args.api_key
        self.num_malicious = 0
        self.num_benign = 0
        self.num_failed_queries = 0
        self.max_sublist_size = 500
        self.max_timeout = args.query_timeout
        self.timeout = args.query_timeout
        self.query_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

    def query(self, urls):
        query_urls = self.format_query(urls)
        dict_sublists = self.gen_sublists(query_urls, self.max_sublist_size)
        url_sublists = self.gen_sublists(urls, self.max_sublist_size)
        malicious_urls = []
        benign_urls = []
        for i, l in enumerate(dict_sublists):
            if not len(l):
                continue
            payload = {
                "client": {"clientId": "mycompany", "clientVersion": "0.1"},
                "threatInfo": {
                    "threatTypes": [
                        "THREAT_TYPE_UNSPECIFIED",
                        "MALWARE",
                        "SOCIAL_ENGINEERING",
                        "UNWANTED_SOFTWARE",
                        "POTENTIALLY_HARMFUL_APPLICATION",
                    ],
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": [
                        "THREAT_ENTRY_TYPE_UNSPECIFIED",
                        "URL",
                        "EXECUTABLE",
                    ],
                    "threatEntries": l,
                },
            }
            params = {"key": self.api_key}
            r = requests.post(self.query_url, params=params, json=payload).json()
            matches = []
            if "error" in r.keys():
                self.num_failed_queries += 1
                self.timeout -= 1
                print(r["error"])
                continue
            
            if "matches" in r.keys():
                for match in r["matches"]:
                    matches.append(match["threat"]["url"])
                self.timeout = self.max_timeout
            if len(matches) > 1:
                malicious_urls.extend(matches)
                safe_urls = list(set(url_sublists[i]).difference(set(matches)))
            else:
                safe_urls = url_sublists[i]
            benign_urls.extend(safe_urls)
            self.num_benign += len(safe_urls)
            self.num_malicious += len(matches)
        print(
            "Classified %d malicious and %d benign urls with %d errors"
            % (self.num_malicious, self.num_benign, self.num_failed_queries)
        )
        return malicious_urls, benign_urls, self.timeout == 0

    def format_query(self, urls):
        sb_query = []
        for url in urls:
            sb_query.append({"url": url})
        return sb_query

    def get_urls(self, json):
        urls = []
        for obj in json:
            urls.append(obj["url"])
        return urls

    def gen_sublists(self, urls, max_list_size):

        sublists = []

        sublist_size = int((len(urls) / max_list_size) + 1)

        for i in range(1, sublist_size + 1):
            if (i * max_list_size) > len(urls):
                sublists.append(urls[(i - 1) * max_list_size : len(urls)])
            else:
                sublists.append(urls[(i - 1) * max_list_size : i * max_list_size])

        return sublists
