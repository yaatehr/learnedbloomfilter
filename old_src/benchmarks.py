from bloom_filter import BloomFilter
import numpy as np
import random
import time
import timeit
import sys

def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer


def csvtimerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        print(runtime, end="|")
        return value
    return function_timer

# instantiate BloomFilter with custom settings,
# max_elements is how many elements you expect the filter to hold.
# error_rate defines accuracy; You can use defaults with
# `BloomFilter()` without any arguments. Following example
# is same as defaults:


error_rate = 0.001

bloom1k = BloomFilter(max_elements=1000, error_rate=error_rate)
bloom10k = BloomFilter(max_elements=10000, error_rate=error_rate)
bloom100k = BloomFilter(max_elements=100000, error_rate=error_rate)
bloom1m = BloomFilter(max_elements=1000000, error_rate=error_rate)
# fileprefix = "mmape-3"
# bloom1k = BloomFilter(max_elements=1000, error_rate=error_rate, filename=(fileprefix + "1.txt",-1))
# bloom10k = BloomFilter(max_elements=10000, error_rate=error_rate, filename=(fileprefix + "2.txt",-1))
# bloom100k = BloomFilter(max_elements=100000, error_rate=error_rate, filename=(fileprefix + "3.txt",-1))
# bloom1m = BloomFilter(max_elements=1000000, error_rate=error_rate, filename=(fileprefix + "4.txt",-1))

filters = [bloom1k, bloom10k, bloom100k, bloom1m]
filter_names = ["bloom1", "bloom2", "bloom3", "bloom4"]

epochs = 100
num_query_eles = 100000
num_insert_eles = 100   


key_eles = np.random.randint(low=0, high=9999999, size=10000).tolist()
# print(key_eles)

distribution = list(range(1000000))
# print(distribution)

negatives = list(set(distribution).difference(set(key_eles)))

# negatives = [ele for ele in key_eles]

key_eles = [str(ele) for ele in key_eles]
negatives = [str(ele) for ele in negatives]

random.shuffle(negatives)

@csvtimerfunc
def test_filter_queries(bloom, n):
    """
    query 1000 elements so we get a good sense of ammortized runtime 
    return false pos rate
    """
    false_pos = 0
    for i in range(n):
        if negatives[i] in bloom is True:
            false_pos +=1
    return float(false_pos)/n

@csvtimerfunc
def insert_elements(bloom, n):
    """
    insert 100 elements so we get a good sense of ammortized runtime 
    """

    for i in range(n):
        bloom.add(key_eles[i])
    
#baseline test, query an empty filter
# print("empty filter baseline \n")


print("now evaluating on the following criteria, num_insert_eles %d, num_query_eles %d \n\n\n" % (num_insert_eles, num_query_eles))


print("epoch|filter_name|fp_rate|size|ideal_number|insert_time|num_inserted_eles|query_time|num_query_eles|exp_fpr|sys_getsizeof")

for idx,filter in enumerate(filters):

    print("0|%s|%f|%d|%d|0|0|" % (filter_names[idx], filter.error_rate_p, filter.num_bits_m, filter.ideal_num_elements_n), end="")
    fpr = test_filter_queries(filter, num_query_eles)
    print(str(num_query_eles) + "|" +str(fpr) + "|" + str(sys.getsizeof(filter)))

inserted_elements = set()

for i in range(epochs):
    inserted_elements.update(key_eles[0:100])
    for idx,filter in enumerate(filters):
        print(str(i+1) + "|%s|%f|%d|%d|" % (filter_names[idx], filter.error_rate_p, filter.num_bits_m, filter.ideal_num_elements_n), end="")
        insert_elements(filter, num_insert_eles)
        print(len(inserted_elements), end="|")
        fpr = test_filter_queries(filter, num_query_eles)
        print(str(num_query_eles) + "|" +str(fpr) + "|" + str(sys.getsizeof(filter)))
         
        # print("had an empirical false positive rate of ~ {} ~ on # {} # elements " % (test_filter_queries(filter, num_query_eles), num_query_eles))
        
    random.shuffle(key_eles)
    random.shuffle(negatives)