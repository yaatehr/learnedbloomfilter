#define TORCH_INPUT_LEN 124
#define INSET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/clean_dedup_urls.txt"
#define OUTSET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/clean_dedup_urls.txt"
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

class MyFixtureLearned : public benchmark::Fixture
{
public:
   LearnedBloomFilter *filter;
   std::map<std::string, std::vector<int>> valid_index_map;
   std::map<std::string, std::vector<int>> invalid_index_map;
   std::string *str_list;
   std::string *invalid_str_list;
   std::vector<std::string> parsed_phishing_urls;
   std::vector<std::string> parsed_good_urls;

   MyFixtureLearned()
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "fixture init";
      std::cout << "max num val int: " << std::numeric_limits<int>::max();

#endif

      MyFixtureLearned::valid_index_map = {};
      MyFixtureLearned::invalid_index_map = {};

   };

   void SetUp(const ::benchmark::State &state)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "fixture setup entered";
#endif
      filter = new LearnedBloomFilter(state.range(1), state.range(0));
      std::string key = "k" + std::to_string(state.range(2)) + std::to_string(TORCH_INPUT_LEN);
      if (valid_index_map.count(key) == 0)
      {
         auto valid_index_subset = select_random_vector_subset(MyFixtureLearned::filter->validIndices, state.range(2));
         auto invalid_index_subset = select_random_vector_subset(MyFixtureLearned::filter->invalidIndices, state.range(2));
         valid_index_map.emplace(key, std::move(valid_index_subset));
         invalid_index_map.emplace(key, std::move(invalid_index_subset));
      }
   }
   void TearDown(const ::benchmark::State &state)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "fixture teardown entered";
#endif
      delete filter;
      // LearnedBloomFilter *filter;
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
      st.PauseTiming(); // Stop timers. They will not count until they are resumed.
      double numFalsePos = 0.0;

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "setting numItems " << std::endl;
#endif
      int numItems = st.range(2);
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "setting counters " << std::endl;
#endif

      // st.counters["fpr"] = 0;
      // st.counters["num_hashes"] = 0;
      // st.counters["table_size"] = 0;
      std::string key = "k" + std::to_string(st.range(2)) + std::to_string(st.range(3));
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "getting tensor from map" << std::endl;
#endif
      std::vector<int> valid_tensor_indices = MyFixtureLearned::valid_index_map[key];
      std::vector<int> invalid_tensor_indices = MyFixtureLearned::invalid_index_map[key];


#ifdef USER_DEBUG_STATEMENTS
      std::cout << "getting accessors from tensor" << std::endl;
#endif


      auto tensor_accessor = MyFixtureLearned::filter->X->accessor<float, 3>();
      #ifdef USER_DEBUG_STATEMENTS
      std::cout << "data accessor created of size: " << tensor_accessor.size(0) << std::endl;
#endif

// iterate through tensor and call one at a time

         for (auto i : valid_tensor_indices)
         {
            torch::Tensor valid_tensor = torch::from_blob(tensor_accessor[i].data(), {124, 32});
            MyFixtureLearned::filter->predict(valid_tensor);
         }

//       for (auto inp : valid_inputs)
//       {
// // auto inp = valid_inputs->at(i);
// #ifdef USER_DEBUG_STATEMENTS
//          std::cout << "inputing element: " << inp << std::endl;
// #endif
//          // std::cout << MyFixtureLearned::filter->query("stringoflengthasdfgh") << std::endl;
//          MyFixtureLearned::filter->insert(inp);
//       }
      st.ResumeTiming();

      // numFalsePos = MyFixtureLearned::filter->batch_query_count(*invalid_inputs);

      double fpr = numFalsePos * 100 / (double)(numItems);
      double num_hashes = (double)MyFixtureLearned::filter->filter->hash_count();
      double table_size = (double)MyFixtureLearned::filter->filter->size();

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "fpr: " << fpr << " numhashes: " << num_hashes << " table_size: " << table_size << std::endl;
#endif
      st.counters.insert({{"fpr", fpr}, {"num_hashes", num_hashes}, {"table_size", table_size}});

      // st.counters["fpr"] = fpr;
      // st.counters["num_hashes"] = MyFixtureLearned::filter->hash_count();
      // st.counters["table_size"] = MyFixtureLearned::filter->size();
   }
}

// BENCHMARK_DEFINE_F(MyFixtureLearned, TestBloomFilterStringInsertion)
// (benchmark::State &st)
// {
//    MyFixtureLearned::filter->filter->clear();
//    for (auto _ : st)
//    {
//       st.PauseTiming(); // Stop timers. They will not count until they are resumed.
//       MyFixtureLearned::filter->filter->clear();
//       std::string key = "k" + std::to_string(st.range(2)) + std::to_string(st.range(3));
//       // if(!MyFixtureLearned::valid_index_map[key])
//       //    std::cout << "string array is null" << std::endl;
//       // else {
//       //    std::cout << MyFixtureLearned::valid_index_map[key].get()->at(0) << std::endl;
//       // }

//       auto valid_inputs = MyFixtureLearned::valid_index_map[key].get();
//       st.ResumeTiming();

//       MyFixtureLearned::filter->insert(*valid_inputs);

//    }
// }

// BENCHMARK_DEFINE_F(MyFixtureLearned, TestBloomFilterIntInsertion)
// (benchmark::State &st)
// {
//    MyFixtureLearned::filter->clear();
//    for (auto _ : st)
//    {
//       for (int i = 0; i < st.range(2); i++)
//       {
//          MyFixtureLearned::filter->insert(i);
//       }
//       st.PauseTiming(); // Stop timers. They will not count until they are resumed.
//       MyFixtureLearned::filter->clear();
//       st.ResumeTiming();
//    }
// }

// BENCHMARK_DEFINE_F(MyFixtureLearned, TestBloomFilterIntQuery)
// (benchmark::State &st)
// {
//    for (int i = 0; i < 8 << 10; i++)
//    {
//       MyFixtureLearned::filter->insert(i);
//    }
//    for (auto _ : st)
//    {
//       double numFalsePos = 0.0;
//       int numItems = st.range(2);
//       st.counters["fpr"] = 0;
//       st.counters["num_hashes"] = 0;
//       st.counters["table_size"] = 0;
//       for (int i = -1; i > st.range(2) * -1; i--)
//       {
//          MyFixtureLearned::filter->contains(i);
//       }
//       st.counters["fpr"] = numFalsePos*100/(double) numItems;
//       st.counters["num_hashes"] = MyFixtureLearned::filter->hash_count();
//       st.counters["table_size"] = MyFixtureLearned::filter->size();
//    }
// }

// BENCHMARK(BM_generate_random_string)
//     ->Ranges({{8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_compute_optimal_params)
//    ->Ranges({{2, 2 << 10}, {1000, 1000000}});

/* BarTest is NOT registered */
// range {false positive rate^-1, num_projected eles, number of eles for testing fixture, ele length in characters}
BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_example01);

BENCHMARK_MAIN();
