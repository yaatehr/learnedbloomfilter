import numpy as np
import os

# Adapted from https://github.com/minimaxir/char-embeddings

# TODO Try other glove vectors?

data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "input")
print(data_root)

glove_vector_set = "glove.6B.50d.txt"
file_path = os.path.join(data_root, glove_vector_set)
print(file_path)

vectors = {}
with open(file_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        #increment sum and counts

        for char in word:
            if ord(char) < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                    
                else:
                    vectors[char] = (vec, 1)

base_name = os.path.join(data_root, glove_vector_set[:-4]) + '-char.txt'
print(base_name)
with open(base_name, 'w') as f2:
    for char in vectors:
        avg_vector = np.round(
            (vectors[char][0] / vectors[char][1]), 6).tolist()
        f2.write(char + " " + " ".join(str(x) for x in avg_vector) + "\n")