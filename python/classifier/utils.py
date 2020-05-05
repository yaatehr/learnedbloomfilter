# NOTE CITATION https://github.com/ahmedbesbes/character-based-cnn

import math
import json
import re
import numpy as np
from sklearn import metrics
from classifier import bloom_calc, pytorch_modelsize
# text-preprocessing
import sys
from types import ModuleType, FunctionType
from gc import get_referents

def lower(text):
    return text.lower()


def remove_hashtags(text):
    clean_text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    return clean_text


preprocessing_setps = {
    "remove_hashtags": remove_hashtags,
    "remove_urls": remove_urls,
    "remove_user_mentions": remove_user_mentions,
    "lower": lower,
}


def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)
    return text


# metrics // model evaluations


def get_evaluation(y_true, y_prob, args, list_metrics):
    y_pred = np.round(y_prob)
    output = {}
    if "accuracy" in list_metrics:
        output["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    if "f1" in list_metrics:
        output["f1"] = metrics.f1_score(y_true, y_pred, average="weighted")
    if "bloom_threshold_accuracy" in list_metrics:
        bloom_pred = (y_prob > args.tau).astype(np.float32)
        # print(bloom_pred)
        output["bloom_threshold_accuracy"] = metrics.accuracy_score(y_true, bloom_pred)

    return output


def mitzenmacher_theorem(alpha, fpr, fnr, b, zeta, m):
    """
        alpha - usually predetermined (~0.6185), the rate at which fpr falls in a generic bloom filter is alpha^b where b is bits per item
        fpr - lbf false positive rate (emp.)
        fnr - lbf false negative rate (emp.)
        b - bits per item
        zeta - size of learned classifier
        m - number of keys in the in set ()
        returns - scalar value, if it is greater than or equal to 0, the LBF can potentiallly save space!
    """

    lhs = zeta / float(m)
    inner_term = fpr + (1 - fpr)*alpha**(b/ fnr)
    rhs = math.log(inner_term, alpha) -b
    return rhs - lhs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# preprocess input for prediction


def preprocess_input(args):
    raw_text = args.text
    steps = args.steps
    for step in steps:
        raw_text = preprocessing_setps[step](raw_text)

    number_of_characters = args.number_of_characters + len(args.extra_characters)
    identity_mat = np.identity(number_of_characters)
    vocabulary = list(args.alphabet) + list(args.extra_characters)
    max_length = args.max_length

    processed_output = np.array(
        [
            identity_mat[vocabulary.index(i)]
            for i in list(raw_text[::-1])
            if i in vocabulary
        ],
        dtype=np.float32,
    )
    if len(processed_output) > max_length:
        processed_output = processed_output[:max_length]
    elif 0 < len(processed_output) < max_length:
        processed_output = np.concatenate(
            (
                processed_output,
                np.zeros(
                    (max_length - len(processed_output), number_of_characters),
                    dtype=np.float32,
                ),
            )
        )
    elif len(processed_output) == 0:
        processed_output = np.zeros(
            (max_length, number_of_characters), dtype=np.float32
        )
    return processed_output


# cyclic learning rate scheduling


def cyclical_lr(stepsize, min_lr=1.7e-3, max_lr=1e-2):

    # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x):
        return 1.0

    # Lambda function to calculate the LR
    def lr_lambda(it):
        return min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_bf_size(target_fpr, projected_eles):
    #NOTE m_calc is a sanity check based on empirically best alpha (.6185)
    m_calc = math.log(target_fpr)*projected_eles/(math.log(0.6185)) # compute table size in bits 
    k,m,n,p = bloom_calc.km_from_np(projected_eles, target_fpr)
    a = sizeof_fmt(m_calc, suffix="b")
    b = sizeof_fmt(m, suffix="b")
    # print(f"calculated m to be {a}")
    print(f"calculated optimal bloom filter size to be {a, b}")
    return m
    # print(f"hurst calculated m to be {b}")


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

def get_model_size(model, args, input_features=None):
    input_size = (1, 1, args.max_length*args.embedding_size) if input_features == None else list(input_features.size())
    print(input_size)
    try:
        input_size[0] = 1
        input_size = tuple(input_size)
    except:
        pass
    print(input_size)
    print(pytorch_modelsize.summary_string(model, input_size=input_size)[0])
    print("\n\n\n\n\n\n\n")

    print("python size estimate: ", getsize(model))

    (total_size, total_input_size, total_output_size, total_params_size), (total_params, trainable_params) = pytorch_modelsize.summary_tuple(model, input_size=input_size)
    print("total size estimate ", total_size)
    print("params: ", total_params_size) # bits taken up by parameters
    print("forward backwards: ", total_output_size) # bits stored for forward and backward
    print("input bits: ", total_input_size) # bits for input
    return float(total_size)

    