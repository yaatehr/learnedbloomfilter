#define MIN_TAU 0.25
#define MAX_TAU 0.95
#define MIN_FPR 0.0001
#define MAX_FPR 0.05
#define PROJECTED_ELE_COUNT 10
#define COMPOUND_MODEL_SIZE  6812
#define ARG_LENGTH 15

#define DATASET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/timestamp_dataset"
// #ifndef USER_DEBUG_STATEMENTS
// #define USER_DEBUG_STATEMENTS
// #endif

#include <iostream>
#include <string>
#include <benchmark/benchmark.h>
#include <vector>
#include <map>
#include <fstream>
#include <numeric>
#include <random>
#include <iterator>
#include <memory>
#include <chrono>
// #include <unistd.h>

#include "Filters/myUtils.cpp"
#include "Filters/learned_bloom.cpp"
#include "Filters/bloom_filter.hpp"



      void sanityCheck() {
            std::cout << " TESTING BF SIZE: " << std::endl;
            bloom_parameters parameters;
            parameters.projected_element_count = 1000000;
            parameters.false_positive_probability = .001;
            if (!parameters)
            {
            std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
            return;
            }
            parameters.compute_optimal_parameters();

            bloom_filter* gbf = new bloom_filter(parameters);
            std::cout << " Filter size was : " << gbf->size() << std::endl;
      }

// main function to measure elapsed time of a C++ program 
// using chrono library
int main()
{
	// auto start = chrono::steady_clock::now();

	// auto end = chrono::steady_clock::now();

	// std::cout << "Elapsed time in nanoseconds : " 
	// 	<< chrono::duration_cast<chrono::nanoseconds>(end - start).count()
	// 	<< " ns" << endl;


      std::ofstream output_file;
      output_file.open("timestamp_lstm_1.csv");
      // write header to file
      output_file << "empirical_fpr,num_hashes,table_size,tau,lbf_size,target_fpr,insert_time,query_time,num_eles\n";
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture init";
            std::cout << " with max num val int: " << std::numeric_limits<int>::max() << std::endl;

#endif

      LearnedBloomFilter *filter;
      std::vector<std::string> key_strings = load_dataset(DATASET_PATH);
      std::shared_ptr<torch::Tensor> data;
      std::shared_ptr<torch::Tensor> labels;
      std::vector<int> validIndices;
      std::vector<int> invalidIndices;
      std::vector<double> tau = linspace(MIN_TAU, MAX_TAU, ARG_LENGTH - 1);
      tau.push_back(1);
      std::vector<double> fpr = linspace(MAX_FPR, MIN_FPR, ARG_LENGTH);

      std::shared_ptr<torch::jit::script::Module> classifier = LearnedBloomFilter::load_classifier(MODEL_PATH);
      std::tie(data, labels, validIndices, invalidIndices) = LearnedBloomFilter::load_tensor_container(DATA_PATH, PROJECTED_ELE_COUNT);


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "loaded " << key_strings.size() << " strings from dataset" << std::endl;
#endif

      for(int i = 0 ; i < ARG_LENGTH; i++) { // tau index
            for( int j = 0; j < ARG_LENGTH; j++) {// fpr index


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture setup entered";
#endif
            filter = new LearnedBloomFilter(PROJECTED_ELE_COUNT, fpr[j], classifier, data, labels, validIndices, invalidIndices, key_strings);
            double t = tau[i];
            filter->set_tau(t);
      

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Entering TestBloomFilterStringQuery loop" << std::endl;
#endif

            double numFalsePos = 0.0;


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "setting numItems " << std::endl;
#endif
            int numItems = PROJECTED_ELE_COUNT;

            auto valid_tensor_indices = filter->validIndices;
            auto invalid_tensor_indices = filter->invalidIndices;

#ifdef USER_DEBUG_STATEMENTS
            std::cout << "inserting valid indices into compound model" << std::endl;
#endif
            // insert all valid tensors an strings
	auto insert_start = std::chrono::steady_clock::now();
            filter->insert(valid_tensor_indices);
	auto insert_end = std::chrono::steady_clock::now();

	auto insert_timing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_start).count();

            // query all invalid tensors and strings
	auto query_start = std::chrono::steady_clock::now();
            numFalsePos = filter->batch_query_count(invalid_tensor_indices, false);
	auto query_end = std::chrono::steady_clock::now();

      	auto query_timing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(query_end - query_start).count();
            double exp_fpr = (double) numFalsePos * 100 / (double)(numItems);
            auto num_hashes = filter->filter->hash_count();
            auto table_size = filter->filter->size();

            output_file << exp_fpr  << "," << num_hashes << "," << table_size  << ",";
            output_file << tau[i] <<  ",";
            output_file << COMPOUND_MODEL_SIZE << ",";
            output_file << fpr[j] << "," << insert_timing_ns << "," << query_timing_ns << "\n";

#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture teardown entered";
#endif
            delete filter;

            }
      }
      output_file.close();

};

