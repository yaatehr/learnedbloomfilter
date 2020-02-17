#define TORCH_INPUT_LEN 50
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
   std::map<std::string, std::shared_ptr<std::vector<std::string>>> valid_string_map;
   std::map<std::string, std::shared_ptr<std::vector<std::string>>> invalid_string_map;
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

      MyFixtureLearned::valid_string_map = {};
      MyFixtureLearned::invalid_string_map = {};

      MyFixtureLearned::valid_string_map = {};
      MyFixtureLearned::invalid_string_map = {};
      MyFixtureLearned::parsed_phishing_urls = {};

      // std::ifstream inSetFile(INSET_PATH);
      // std::ifstream outSetFile(OUTSET_PATH);
      std::string line;

      // if(inSetFile.is_open()) {
      //    while (inSetFile.good()) {
      //             getline(inSetFile,line);
      //             // long linepos = infile.tellg();
      //             // std::cout << line << std::endl;
      //             MyFixtureLearned::parsed_phishing_urls.push_back(line);
      //          }
      //       inSetFile.close();
      // } else {
      //            std::cout << "Unable to open inset file";
      // }
      // if(outSetFile.is_open()) {
      //    while (outSetFile.good()) {
      //             getline(outSetFile,line);
      //             // long linepos = infile.tellg();
      //             // std::cout << line << std::endl;
      //             MyFixtureLearned::parsed_phishing_urls.push_back(line);
      //          }
      //       outSetFile.close();
      // } else {
      //            std::cout << "Unable to open outset file";
      // }

      std::ifstream file("../input/randlabelurls.txt");
      std::string line;

      while (std::getline(file, line))
      {
         std::stringstream linestream(line);
         std::string data;
         int val1;
         std::string url;

         // If you have truly tab delimited data use getline() with third parameter.
         // If your data is just white space separated data
         // then the operator >> will do (it reads a space separated word into a string).
         // std::getline(linestream, data, '\t');  // read up-to the first tab (discard tab).

         // Read the integers using the operator >>
         linestream >> val1 >> url;
         if (val1 != 1)
         {
            MyFixtureLearned::parsed_good_urls.push_back(url);
         }
         else
         {
            MyFixtureLearned::parsed_phishing_urls.push_back(url);
         }
      }

      if (MyFixtureLearned::parsed_phishing_urls.empty())
         std::cout << "parsed strings is empty" << std::endl;

      if (MyFixtureLearned::parsed_good_urls.empty())
         std::cout << "parsed strings is empty" << std::endl;
   };

   void SetUp(const ::benchmark::State &state)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "ficture setup entered";
#endif
      filter = new LearnedBloomFilter(state.range(1), state.range(0));
      std::string key = "k" + std::to_string(state.range(2)) + std::to_string(TORCH_INPUT_LEN);
      if (valid_string_map.count(key) == 0)
      {
         auto gvec = select_random_vector_subset(MyFixtureLearned::parsed_phishing_urls, state.range(2));
         std::shared_ptr<std::vector<std::string>> s = std::make_shared<std::vector<std::string>>(gvec);
         auto pvec = select_random_vector_subset(MyFixtureLearned::parsed_good_urls, state.range(2));
         std::shared_ptr<std::vector<std::string>> i = std::make_shared<std::vector<std::string>>(pvec);
         valid_string_map.insert(std::map<std::string, std::shared_ptr<std::vector<std::string>>>::value_type(key, s));
         invalid_string_map.insert(std::map<std::string, std::shared_ptr<std::vector<std::string>>>::value_type(key, i));
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

// MyFixtureLearned::valid_string_map = new std::map<<>>();
// MyFixtureLearned::invalid_string_map = new std::map<>();

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
      std::cout << "getting strings from keys" << std::endl;
#endif
      std::vector<std::string> valid_inputs = *MyFixtureLearned::valid_string_map[key].get();
      auto invalid_inputs = MyFixtureLearned::invalid_string_map[key].get();
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "strings access successful! numvalidInputs: " << valid_inputs.size() << std::endl;
#endif
      for (auto inp : valid_inputs)
      {
// auto inp = valid_inputs->at(i);
#ifdef USER_DEBUG_STATEMENTS
         std::cout << "inputing element: " << inp << std::endl;
#endif
         // std::cout << MyFixtureLearned::filter->query("stringoflengthasdfgh") << std::endl;
         MyFixtureLearned::filter->insert(inp);
      }
      st.ResumeTiming();

      numFalsePos = MyFixtureLearned::filter->batch_query_count(*invalid_inputs);

      double fpr = numFalsePos * 100 / (double)(numItems);
      double num_hashes = (double)MyFixtureLearned::filter->hash_count();
      double table_size = (double)MyFixtureLearned::filter->size();

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
//       // if(!MyFixtureLearned::valid_string_map[key])
//       //    std::cout << "string array is null" << std::endl;
//       // else {
//       //    std::cout << MyFixtureLearned::valid_string_map[key].get()->at(0) << std::endl;
//       // }

//       auto valid_inputs = MyFixtureLearned::valid_string_map[key].get();
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
BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_example01);

BENCHMARK_MAIN();
