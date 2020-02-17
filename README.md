<p align="center">
  <img src="./docs/Dsail-logo-2.png"/>
</p>

# Learned Bloom Filter

This is a work in progress under MIT's DSAIL lab
Please contact yaatehr@mit.edu if you have any questions or concerns

<!-- - [Learned Bloom Filter](#learned-bloom-filter) -->
  - [Project Overview](#project-overview)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
    - [Plotting to Tensorboard](#plotting-to-tensorboard)
  - [Further Reading/Citations](#further-readingcitations)

## Project Overview

The overall aim of the project is to make Learned Bloom Filters more accessible to non-specialists. Despite not natively offering the same guarantees as their classic counterparts (data agnosticism and no false negatives), researchers have reported space reductions from 10-70% compared to classical filters with the same false positive rate.

## Getting Started

1. **Install The Conda Environment**.
    ```bash
    conda env create -f environment_lbf.yml
    ```
2. **Install [Pytorch](https://pytorch.org/) Nightly Preview**
3. **Download Sample Datasets** #TODO make dropbox link to large files
4. **Configure C++ Dependencies** #TODO reinstall on another computer, write down steps
5. Run tests for classifier #TODO make tests
6. Run tests for benchmark #TODO make tests

## Usage

This package is a learned bloom filter meant to be a drop in replacement for existing generic bloom filters. It features the ability to train a sclassifer on user data in python, export that classifier to c++, and benchmark the new learned bloom filter in c++. 

### Plotting to Tensorboard

**Locally**

Run this command at the root of the project:

```bash
tensorboard --logdir=./logs/ --port=6006
```

**Remotely**

1) Start the remote server and run tensorboard on the server
```bash
tensorboard --logdir=./logs/ --host $SERVER_IP --port $SERVER_PORT
```
2) SSH tunnel the port to your laptop

```bash
ssh uname@hostname.edu -L 6006:$SERVER_IP:$SERVER_PORT
```

Finally, view the charts at http://localhost:6006 (or whatever host you're using)

## Further Reading/Citations

[1] T. Kraska, A. Beutel, E. H. Chi, J. Dean, and N. Polyzotis, “The case for learned index structures,” CoRR, vol. abs/1712.01208, 2017.

[2] M. Mitzenmacher, “Optimizing learned bloom filters by sandwiching,” arXiv preprint arXiv:1803.01474, 2018.

[3] K. Cho, B. Van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using rnn encoder-decoder for statistical machine translation,” arXiv preprint arXiv:1406.1078, 2014.

[4] J.W.Rae, S.Bartunov, and T.P.Lillicrap,“Meta-learningneuralbloom filters,” arXiv preprint arXiv:1906.04304, 2019.

[5] S. Macke, A. Beutel, T. Kraska, M. Sathiamoorthy, D. Z. Cheng, and H. Chi, “Lifting the curse of multidimensional data with learned existence indexes,”

[6] Singhal, Karan, and Philip Weiss. "DeepBloom Building a Novel Learned Index Bloom Filter."

[7] A. Partow, “General Purpose Hash Function Algorithms - By Arash Partow.”, http://www.partow.net/programming/hashfunctions/index.html

[8] A, Besbes, "Character Based CNN"
 https://github.com/ahmedbesbes/character-based-cnn


Note this is readme in not finalized and some citations may be missing. 