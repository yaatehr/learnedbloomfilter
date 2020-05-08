#define MIN_TAU 0.35
#define MAX_TAU 0.95
#define MIN_FPR 0.0001
#define MAX_FPR 0.05
#define PROJECTED_ELE_COUNT 10000
#define COMPOUND_MODEL_SIZE  6812
#define ARG_LENGTH_T 20
#define ARG_LENGTH_F 30
#define MAX_SANITY_CHECK_SIZE 50000

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
// args must be entered in order
// arg0 - dataset_path
// arg1 - 
int main(int argc, char *argv[])
{
	// auto start = chrono::steady_clock::now();

	// auto end = chrono::steady_clock::now();

	// std::cout << "Elapsed time in nanoseconds : " 
	// 	<< chrono::duration_cast<chrono::nanoseconds>(end - start).count()
	// 	<< " ns" << endl;

#ifdef USER_DEBUG_STATEMENTS
for(int i = 0; i < argc; i++) {
      std::cout << argv[i] << std::endl;
}
#endif

      std::string dataset_path = argc < 2 ? DATASET_PATH : argv[1];
      std::string model_path = argc < 3 ? MODEL_PATH : std::string(argv[1]).append("/").append(argv[2]).append(".pt");
      std::string container_path = argc < 3 ? CONTAINER_PATH : std::string(argv[1]).append("/").append(argv[2]).append("_container.pt");


      std::ofstream output_file;
      output_file.open(std::string(argv[2]).append(".csv"));
      // write header to file
      output_file << "empirical_fpr,num_hashes,table_size,tau,lbf_size,target_fpr,insert_time,query_time,num_eles_tested,projected_fallback_count,projected_fallback_percentage,fallback_count,gbf_effective_fpr\n";
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture init";
            std::cout << " with max num val int: " << std::numeric_limits<int>::max() << std::endl;

#endif

      LearnedBloomFilter *filter;
      std::vector<std::string> key_strings;
      std::vector<int> plaintext_labels; 
      std::tie(plaintext_labels, key_strings) = load_dataset(dataset_path);
      std::shared_ptr<torch::Tensor> data;
      std::shared_ptr<torch::Tensor> labels;
      std::vector<int> validIndices;
      std::vector<int> invalidIndices;


      std::vector<double> tau;
      std::vector<double> tau_fallback_percentage;
      std::tie(tau, tau_fallback_percentage) = load_tau_to_fpr(dataset_path);

      // std::vector<double> tau = linspace(MIN_TAU, MAX_TAU, ARG_LENGTH_T - 1);
      tau.push_back(1);
      tau_fallback_percentage.push_back(1);
      int arg_length_t = tau.size();
      std::vector<double> fpr = linspace(MAX_FPR, MIN_FPR, ARG_LENGTH_F);

      std::shared_ptr<torch::jit::script::Module> classifier = LearnedBloomFilter::load_classifier(model_path);
      std::tie(data, labels, validIndices, invalidIndices) = LearnedBloomFilter::load_tensor_container(container_path, -1);


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "loaded " << key_strings.size() << " strings from dataset" << std::endl;
#endif

      for(int i = 0 ; i < arg_length_t; i++) { // tau index
            std::cout << "progress bar: " << i << "/" << arg_length_t << std::endl;
            for( int j = 0; j < ARG_LENGTH_F; j++) {// fpr index


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture setup entered";
#endif
            bool evaluate_filter = i + j == 0;// only evaluate on the first run

            double t = tau[i];
            int projected_ele_count = tau_fallback_percentage[i]*PROJECTED_ELE_COUNT;
            filter = new LearnedBloomFilter(projected_ele_count, fpr[j], classifier, data, labels, validIndices, invalidIndices, key_strings, plaintext_labels, evaluate_filter);
            filter->set_tau(t);

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Entering TestBloomFilterStringQuery loop" << std::endl;
#endif

            double numFalsePos = 0.0;


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "setting numItems " << std::endl;
#endif
            int numItems = PROJECTED_ELE_COUNT;

            auto valid_tensor_indices = select_random_vector_subset(filter->validIndices, numItems);
            auto invalid_tensor_indices = select_random_vector_subset(filter->invalidIndices, numItems);
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "inserting valid indices into compound model" << std::endl;
#endif
            // insert all valid tensors an strings
	auto insert_start = std::chrono::steady_clock::now();
            auto num_bypassed_inserts = filter->batch_insert(valid_tensor_indices);
	auto insert_end = std::chrono::steady_clock::now();
	auto insert_timing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_start).count();

            // auto numFalseNeg = filter->batch_query_count(valid_tensor_indices, true);
            std::cout << "number of bypassed inserts in the reduced size compound model" << num_bypassed_inserts << std::endl;
            if(evaluate_filter) {
                  auto f = new LearnedBloomFilter(PROJECTED_ELE_COUNT, fpr[j], classifier, data, labels, validIndices, invalidIndices, key_strings, plaintext_labels, evaluate_filter);
                  f->set_tau(t);

                  auto num_bypassed_inserts = f->batch_insert(valid_tensor_indices);
                  std::cout << "number of bypassed inserts in the reduced size compound model" << num_bypassed_inserts << std::endl;
                  auto nfp = f->batch_query_count(invalid_tensor_indices, false);
                  double expfpr = (double) numFalsePos * 100 / (double)(numItems);

                  std::cout << "number of false positives for the eval model (with full sized backup):\n" << nfp << "\nWith an fpr of: " << expfpr << std::endl;
            }



            // query all invalid tensors and strings
	auto query_start = std::chrono::steady_clock::now();
            numFalsePos = filter->batch_query_count(invalid_tensor_indices, false);
	auto query_end = std::chrono::steady_clock::now();

      	auto query_timing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(query_end - query_start).count();
            double exp_fpr = (double) numFalsePos * 100 / (double)(numItems);
            auto num_hashes = filter->filter->hash_count();
            auto table_size = filter->filter->size();
            double gbf_effective_fpr = filter->filter->effective_fpp();
            int num_fallback_eles = filter->filter->element_count();

            if(evaluate_filter) {
                  std::cout << "number of false positives for modified compound model (with smaller backup) :\n" << numFalsePos << "\n with fpr of: " << exp_fpr << std::endl;
            }

            output_file << exp_fpr  << "," << num_hashes << "," << table_size  << ",";
            output_file << tau[i] <<  ",";
            output_file << COMPOUND_MODEL_SIZE << ",";
            output_file << fpr[j] << "," << insert_timing_ns << "," << query_timing_ns << "," << numItems << ",";
            output_file << projected_ele_count << "," << tau_fallback_percentage[i] << "," << num_fallback_eles << "," << gbf_effective_fpr << "\n";

#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture teardown entered";
#endif
            delete filter;

            }
      }
      output_file.close();

};

