import exrex
import os
import sys

url_regex = '^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$'
url_regex = '^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$'
# url_regex = '^((http[s]?|ftp):\/)'

SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(SRC_PATH, "generated_good_urls.txt"), 'w+') as outfile:
    for i in range(1000000):
        url = exrex.getone(url_regex)
        outfile.write(url + "\n")
    