# we will make categories out of.... 
# 
import os
import pickle
from collections import Counter

def load_all_categories():
    path= "/Users/yaatehr/Documents/course6/research/shallablacklists/"
    url_paths_by_category = {}
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root)
        # print(files)
        dirname = os.path.basename(root)
        for name in files:

            if "url" in name:
                # print(name)
                # print(dirname)
                url_paths_by_category[dirname] = os.path.join(root, name)

    urls_by_category = {}
    url_counter = Counter()

    for key in url_paths_by_category.keys():
        path = url_paths_by_category[key]
        current_category_url = []
        errorcounter = 0
        with open(path, 'rb') as fp:
            for bline in fp:
                try:
                    line = bline.decode().strip()
                    current_category_url.append(line)
                    url_counter[key] += 1
                except UnicodeDecodeError as e:
                    # process error
                    errorcounter += 1

            print("Category %s had %d errors" % (key, errorcounter))
            urls_by_category[key] = current_category_url
        
    print(url_counter)

    with open("url_load_backup.pkl", 'wb') as fp:
        pickle.dump(urls_by_category, fp) 


load_all_categories()