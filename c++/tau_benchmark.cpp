#define MIN_TAU 0.25
#define MAX_TAU 0.95
#define MIN_FPR 0.0001
#define MAX_FPR 0.05
#define PROJECTED_ELE_COUNT 10490
#define COMPOUND_MODEL_SIZE  6812
#define ARG_LENGTH 30

#define DATASET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/timestamp_dataset"
#ifndef USER_DEBUG_STATEMENTS
#define USER_DEBUG_STATEMENTS
#endif

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
#include "Filters/myUtils.cpp"
#include "Filters/learned_bloom.cpp"
#include "Filters/bloom_filter.hpp"

class MyFixtureLearned : public benchmark::Fixture
{
public:
      LearnedBloomFilter *filter;
      std::map<std::string, std::vector<int>> valid_index_map;
      std::map<std::string, std::vector<int>> invalid_index_map;
      std::vector<std::string> key_strings;
      std::shared_ptr<torch::Tensor> data;
      std::shared_ptr<torch::Tensor> labels;
      std::vector<int> validIndices;
      std::vector<int> invalidIndices;
      std::vector<double> tau;
      std::vector<double> fpr;
      int compount_model_size;
      std::shared_ptr<torch::jit::script::Module> classifier;

      MyFixtureLearned()
      {
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture init";
            std::cout << " with max num val int: " << std::numeric_limits<int>::max() << std::endl;

#endif

            MyFixtureLearned::valid_index_map = {};
            MyFixtureLearned::invalid_index_map = {};
            MyFixtureLearned::key_strings = load_dataset(DATASET_PATH);
            MyFixtureLearned::tau = linspace(MIN_TAU, MAX_TAU, ARG_LENGTH - 1);
            MyFixtureLearned::tau.push_back(1); //TODO tau of 1 should circuit break and just return false in the preediction so th eonly false prediction is in the dataset
            MyFixtureLearned::fpr = linspace(MAX_FPR, MIN_FPR, ARG_LENGTH);

   
            classifier = LearnedBloomFilter::load_classifier(MODEL_PATH);
            std::tie(data, labels, validIndices, invalidIndices) = LearnedBloomFilter::load_tensor_container(DATA_PATH);


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "loaded " << key_strings.size() << " urls from dataset" << std::endl;
#endif
      };

      void SetUp(const ::benchmark::State &state)
      {
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture setup entered";
#endif
            filter = new LearnedBloomFilter(PROJECTED_ELE_COUNT, fpr[state.range(0)], classifier, data, labels, validIndices, invalidIndices, key_strings);
            double t = tau[state.range(1)];
            filter->set_tau(t);
      }
      void TearDown(const ::benchmark::State &state)
      {
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture teardown entered";
#endif
            delete filter;
      }
private:
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
};

BENCHMARK_DEFINE_F(MyFixtureLearned, TestBloomFilterStringQuery)
(benchmark::State &st)
{
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Entering TestBloomFilterStringQuery loop" << std::endl;
#endif
      for (auto _ : st)
      {
            std::cout << "top of for loop" << std::endl;
            st.PauseTiming(); // Stop timers. They will not count until they are resumed.
            double numFalsePos = 0.0;


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "setting numItems " << std::endl;
#endif
            int numItems = PROJECTED_ELE_COUNT;
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "setting counters " << std::endl;
#endif

#ifdef USER_DEBUG_STATEMENTS
            std::cout << "getting tensor indices from map" << std::endl;
            // std::cout << MyFixtureLearned::valid_index_map << std::endl;
#endif
            auto valid_tensor_indices = MyFixtureLearned::filter->validIndices;
            auto invalid_tensor_indices = MyFixtureLearned::filter->invalidIndices;

#ifdef USER_DEBUG_STATEMENTS
            std::cout << "inserting valid indices into compound model" << std::endl;
#endif
            // insert all valid tensors an strings
            MyFixtureLearned::filter->insert(valid_tensor_indices);

            st.ResumeTiming();
            // query all invalid tensors and strings

            numFalsePos = MyFixtureLearned::filter->batch_query_count(invalid_tensor_indices, false);

            double exp_fpr = round_to_digits((double) numFalsePos * 100 / (double)(numItems), 3);
            double num_hashes = (double)MyFixtureLearned::filter->filter->hash_count();
            long table_size = (long)MyFixtureLearned::filter->filter->size();

            #ifdef USER_DEBUG_STATEMENTS
            std::cout << "fpr: " << exp_fpr << " numhashes: " << num_hashes << " table_size: " << table_size << std::endl;
            std::cout << "tau: " << tau[st.range(1)] << std::endl;
            std::cout << "lbf_size: " << COMPOUND_MODEL_SIZE << std::endl;
            std::cout << "target fpr: " << MyFixtureLearned::fpr[st.range(0)] << std::endl;
            //st.counters.insert({{"fpr", fpr}, {"num_hashes", num_hashes}, {"table_size", table_size}, {"tau",  tau[st.range(1)]}, {"lbf_size", COMPOUND_MODEL_SIZE}, {"target_fpr", MyFixtureLearned::fpr[st.range(0)]}});
         
            #endif
            // st.counters.insert({{"fpr", fpr}, {"num_hashes", num_hashes}, {"table_size", table_size}, {"tau",  tau[st.range(1)]}, {"lbf_size", COMPOUND_MODEL_SIZE}, {"target_fpr", MyFixtureLearned::fpr[st.range(0)]}});
            st.counters["fpr"] = (double) exp_fpr;
            st.counters["num_hashes"] = num_hashes;
            st.counters["table_size"] = (double) table_size;
            st.counters["tau"] = (double) round_to_digits(tau[st.range(1)], 3);
            st.counters["lbf_size"] = COMPOUND_MODEL_SIZE;
            st.counters["target_fpr"] =  MyFixtureLearned::fpr[st.range(0)];

      }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 0; i < ARG_LENGTH; i++)
    for (int j = 0; j < ARG_LENGTH; j++)
      b->Args({i, j});
}

// BENCHMARK(BM_generate_random_string)
//     ->Ranges({{8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_compute_optimal_params)
//    ->Ranges({{2, 2 << 10}, {1000, 1000000}});

/* BarTest is NOT registered */
// range {false positive rate^-1, num_projected eles, number of eles for testing fixture, ele length in characters}
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {50, 1000000}, {8, 8 << 10}});
BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Apply(CustomArguments);


BENCHMARK_MAIN();



