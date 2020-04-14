#define TORCH_INPUT_LEN 124
#define INSET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/clean_dedup_urls.txt"
#define OUTSET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/clean_dedup_urls.txt"
#define DATASET_PATH "/Users/yaatehr/Programs/learnedbloomfilters/input/timestamp_dataset"
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
#include "Filters/myUtils.cpp"
#include "Filters/learned_bloom.cpp"
#include "Filters/bloom_filter.hpp"

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds


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
   
            classifier = LearnedBloomFilter::load_classifier(MODEL_PATH);
            // std::tie(data, labels, validIndices, invalidIndices) = LearnedBloomFilter::load_tensor_container(DATA_PATH);


#ifdef USER_DEBUG_STATEMENTS
            std::cout << "loaded " << key_strings.size() << " urls from dataset" << std::endl;
#endif
      };

      void SetUp(const ::benchmark::State &state)
      {
#ifdef USER_DEBUG_STATEMENTS
            std::cout << "fixture setup entered";
#endif
            // filter = new LearnedBloomFilter(state.range(1), state.range(0), classifier, data, labels, validIndices, invalidIndices, key_strings);
            // std::string key = "k" + std::to_string(state.range(2)) + std::to_string(TORCH_INPUT_LEN);
            // if (valid_index_map.count(key) == 0)
            // {
            //       std::vector<int> valid_index_subset = select_random_vector_subset(MyFixtureLearned::filter->validIndices, state.range(2));
            //       std::vector<int> invalid_index_subset = select_random_vector_subset(MyFixtureLearned::filter->invalidIndices, state.range(2));
            //       valid_index_map.emplace(key, std::move(valid_index_subset));
            //       invalid_index_map.emplace(key, std::move(invalid_index_subset));
            // }
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
//             st.PauseTiming(); // Stop timers. They will not count until they are resumed.
//             double numFalsePos = 0.0;

// #ifdef USER_DEBUG_STATEMENTS
//             std::cout << "setting numItems " << std::endl;
// #endif
//             int numItems = st.range(2);
// #ifdef USER_DEBUG_STATEMENTS
//             std::cout << "setting counters " << std::endl;
// #endif

//             std::string key = "k" + std::to_string(st.range(2)) + std::to_string(TORCH_INPUT_LEN);
// #ifdef USER_DEBUG_STATEMENTS
//             std::cout << "getting tensor indices from map" << std::endl;
//             // std::cout << MyFixtureLearned::valid_index_map << std::endl;
// #endif
//             auto valid_tensor_indices = MyFixtureLearned::valid_index_map[key];
//             auto invalid_tensor_indices = MyFixtureLearned::invalid_index_map[key];

// #ifdef USER_DEBUG_STATEMENTS
//             std::cout << "inserting valid indices into compound model" << std::endl;
// #endif
//             // insert all valid tensors an strings
//             MyFixtureLearned::filter->insert(valid_tensor_indices);

//             st.ResumeTiming();
//             // query all invalid tensors and strings

//             numFalsePos = MyFixtureLearned::filter->batch_query_count(invalid_tensor_indices, false);

//             double fpr = (double) numFalsePos * 100 / (double)(numItems);
//             double num_hashes = (double)MyFixtureLearned::filter->filter->hash_count();
//             double table_size = (double)MyFixtureLearned::filter->filter->size();

//             // #ifdef USER_DEBUG_STATEMENTS
//             // std::cout << "fpr: " << fpr << " numhashes: " << num_hashes << " table_size: " << table_size << std::endl;
//             // #endif
//             st.counters.insert({{"fpr", fpr}, {"num_hashes", num_hashes}, {"table_size", table_size}});
          std::this_thread::sleep_for(std::chrono::seconds(30));

      }
}

// BENCHMARK(BM_generate_random_string)
//     ->Ranges({{8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_compute_optimal_params)
//    ->Ranges({{2, 2 << 10}, {1000, 1000000}});

/* BarTest is NOT registered */
// range {false positive rate^-1, num_projected eles, number of eles for testing fixture, ele length in characters}
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {50, 1000000}, {8, 8 << 10}});
BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {4, 60000}, {8, 29318}});


BENCHMARK_MAIN();



