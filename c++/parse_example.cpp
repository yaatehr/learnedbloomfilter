/*
 *********************************************************************
 *                                                                   *
 *                           Open Bloom Filter                       *
 *                                                                   *
 * Description: Basic Bloom Filter Usage                             *
 * Author: Arash Partow - 2000                                       *
 * URL: http://www.partow.net                                        *
 * URL: http://www.partow.net/programming/hashfunctions/index.html   *
 *                                                                   *
 * Copyright notice:                                                 *
 * Free use of the Open Bloom Filter Library is permitted under the  *
 * guidelines and in accordance with the MIT License.                *
 * http://www.opensource.org/licenses/MIT                            *
 *                                                                   *
 *********************************************************************
*/

/*
   Description: This example demonstrates basic usage of the Bloom filter.
                Initially some values are inserted then they are subsequently
                queried, noting any false positives or errors.
*/

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
#include <torch/torch.h>



#include "Filters/bloom_filter.hpp"
#include "Filters/myUtils.cpp"


// std::map<std::string, std::vector<std::string>> *valid_string_map = new std::map<std::string, std::vector<std::string>>();
// std::shared_ptr<std::map<std::string, std::vector<std::string>>> valid_string_map = {};

// std::cout << "shared pointer made";
// std::map<std::string, std::vector<std::string>> invalid_string_map = {};





static void BM_generate_random_string(benchmark::State &state)
{
   for (auto _ : state)
      for (int i = 0; i < state.range(0); i++)
         random_string(state.range(1));
}
// const std::vector<std::pair<int64_t, int64_t>> range_vec_string_gen{new std::pair<int64_t, int64_t>(8, 8<<10), new std::pair<int64_t, int64_t>(10, 100)};

// ->Args({5000, 100});

static void BM_compute_optimal_params(benchmark::State &state)
{
   bloom_parameters parameters;
   parameters.random_seed = 0xA5A5A5A5;
   for (auto _ : state)
   {

      // How many elements roughly do we expect to insert?
      parameters.projected_element_count = state.range(0);

      // Maximum tolerable false positive probability? (0,1)
      parameters.false_positive_probability = 1 / (float)state.range(1);

      // Simple randomizer (optional)

      if (!parameters)
      {
         std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
         return;
      }

      parameters.compute_optimal_parameters();
   }
}

// std::vector<std::pair<int64_t, int64_t>> range_vec{new std::pair<int64_t, int64_t>(2, 2<<10), new std::pair<int64_t, int64_t>(1000, 10000000)};

static void BM_example01(benchmark::State &state)
{
   for (auto _ : state)
   {
      double falsePos = 0.0;

      bloom_parameters parameters;

      // How many elements roughly do we expect to insert?
      parameters.projected_element_count = 1000;

      // Maximum tolerable false positive probability? (0,1)
      parameters.false_positive_probability = 0.0001; // 1 in 10000

      // Simple randomizer (optional)
      parameters.random_seed = 0xA5A5A5A5;

      if (!parameters)
      {
         std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
         return;
      }

      parameters.compute_optimal_parameters();

      //Instantiate Bloom Filter
      bloom_filter filter(parameters);

      std::string str_list[] = {"AbC", "iJk", "XYZ"};

      // Insert into Bloom Filter
      {
         // Insert some strings
         for (std::size_t i = 0; i < (sizeof(str_list) / sizeof(std::string)); ++i)
         {
            filter.insert(str_list[i]);
         }

         // Insert some numbers
         for (std::size_t i = 0; i < 100; ++i)
         {
            filter.insert(i);
         }
      }

      std::string invalid_str_list[] = {"AbCX", "iJkX", "XYZX"};

      // Query Bloom Filter
      {
         // Query the existence of strings
         for (std::size_t i = 0; i < (sizeof(str_list) / sizeof(std::string)); ++i)
         {
            if (filter.contains(str_list[i]))
            {
               // falsePos++;
               // std::cout << "BF contains: " << str_list[i] << std::endl;
            }
         }

         // Query the existence of numbers
         for (std::size_t i = 0; i < 100; ++i)
         {
            if (filter.contains(i))
            {
               // falsePos++;
               // std::cout << "BF contains: " << i << std::endl;
            }
         }

         // Query the existence of invalid strings
         for (std::size_t i = 0; i < (sizeof(invalid_str_list) / sizeof(std::string)); ++i)
         {
            if (filter.contains(invalid_str_list[i]))
            {
               falsePos++;
               // std::cout << "BF falsely contains: " << invalid_str_list[i] << std::endl;
            }
         }

         // Query the existence of invalid numbers
         for (int i = -1; i > -100; --i)
         {
            if (filter.contains(i))
            {
               falsePos++;
               // std::cout << "BF falsely contains: " << i << std::endl;
            }
         }
      }
      // std::cout << "BF false pos rate: " << falsePos/(double)((sizeof(invalid_str_list) / sizeof(std::string)) + 100 ) << std::endl;
   }
}



class MyFixture : public benchmark::Fixture
{
public:
   bloom_parameters parameters;
   // Simple randomizer (optional)
   bloom_filter* filter;
   // std::map<std::string, std::vector<std::string>> valid_string_map;
   std::map<std::string, std::shared_ptr<std::vector<std::string>>> valid_string_map;
   std::map<std::string, std::shared_ptr<std::vector<std::string>>> invalid_string_map;
   std::vector<std::string> parsed_phishing_urls;
   std::vector<std::string> parsed_good_urls;
   // std::map<std::string, std::string*> valid_string_map;
   // std::map<std::string, std::shared_ptr<std::string*>> valid_string_map;

   std::string *str_list;
   std::string *invalid_str_list;

   MyFixture(){
      std::cout << "fixture made";
      MyFixture::valid_string_map = {};
      MyFixture::invalid_string_map = {};
      MyFixture::parsed_phishing_urls = {};

      // std::ifstream infile("../../datasets/clean_dedup_urls2.txt");
      std::ifstream infile("/home/yaatehr/programs/learnedbloomfilter/clean_dedup_urls.txt");
      std::ifstream goodUrlFile("/home/yaatehr/programs/learnedbloomfilter/clean_dedup_urls.txt");
      // std::cout << infile.peek() << std::endl;
      std::string line;
   
      if(infile.is_open()) {
         while (infile.good()) {
                  getline(infile,line);
                  // long linepos = infile.tellg();
                  // std::cout << line << std::endl;
                  MyFixture::parsed_phishing_urls.push_back(line);
               }
            infile.close();
      } else {
                 std::cout << "Unable to open file"; 
      }
      if(goodUrlFile.is_open()) {
         while (goodUrlFile.good()) {
                  getline(goodUrlFile,line);
                  // long linepos = infile.tellg();
                  // std::cout << line << std::endl;
                  MyFixture::parsed_phishing_urls.push_back(line);
               }
            goodUrlFile.close();
      } else {
                 std::cout << "Unable to open file"; 
      }
      

      torch::jit::script::Module module;
      try {
         // Deserialize the ScriptModule from a file using torch::jit::load().
         module = torch::jit::load("/Users/yaatehr/Programs/learnedbloomfilters/traced_ascii_regression.pt");
      }
      catch (const c10::Error& e) {
         std::cerr << "error loading the model\n";
      }
      // while (true) {
      //    if (!getline(infile, line)) break;
      //    // long linepos = infile.tellg();
      //    std::cout << line << std::endl;
      //    MyFixture::parsed_phishing_urls.push_back(line);
      // }
      

      if(MyFixture::parsed_phishing_urls.empty())
         std::cout << "parsed strings is empty" << std::endl;

      if(MyFixture::parsed_good_urls.empty())
         std::cout << "parsed strings is empty" << std::endl;

   };

   void SetUp(const ::benchmark::State &state)
   {
      // std::cout << "entering setup" << std::endl;
      // if(MyFixture::valid_string_map.empty())
      //    std::cout << "string map is empty" << std::endl;

      // parameters = new bloom_parameters();

      parameters.random_seed = 0xA5A5A5A5;

      // How many elements roughly do we expect to insert?
      // How many elements roughly do we expect to insert?
      parameters.projected_element_count = state.range(1);

      // Maximum tolerable false positive probability? (0,1)
      parameters.false_positive_probability = 1.0 / (float)state.range(0);

      if (!parameters)
      {
         std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
         return;
      }

      parameters.compute_optimal_parameters();
      //Instantiate Bloom Filter
      filter = new bloom_filter(parameters);
      std::string key = "k" + std::to_string(state.range(2)) + std::to_string(state.range(3));
      if(valid_string_map.count(key) == 0) {
         // std::shared_ptr<std::vector<std::string>> s = std::make_shared<std::vector<std::string>>(gen_string_array(state.range(2), state.range(3), ""));
         // auto vec = gen_string_array(state.range(2), state.range(3), "");
         auto subset = std::make_shared<std::vector<std::string>>(select_random_vector_subset(MyFixture::parsed_phishing_urls, state.range(2)));
         // std::cout << "starting size is" << subset->size() << std::endl;

         // std::shared_ptr<std::vector<std::string>> s = std::make_shared<std::vector<std::string>>(vec);
         std::shared_ptr<std::vector<std::string>> i = std::make_shared<std::vector<std::string>>(select_random_vector_subset(MyFixture::parsed_good_urls, state.range(2)*5));
         // std::shared_ptr<std::vector<std::string>> i = std::make_shared<std::vector<std::string>>(gen_string_array(state.range(2) * 5, state.range(3), "http:\\\\www."));
         // std::cout << "invalid size is" << i->size() << std::endl;

         // invalid_str_list = gen_string_array(state.range(2) * 10, state.range(3), "#");
         // std::vector<std::string> v(str_list, str_list + sizeof(*str_list) / sizeof(std::string));
         // std::vector<std::string> iv(invalid_str_list, invalid_str_list + sizeof(*invalid_str_list) / sizeof(std::string));
         valid_string_map.insert(std::map<std::string, std::shared_ptr<std::vector<std::string>>>::value_type(key, subset));
         invalid_string_map.insert(std::map<std::string, std::shared_ptr<std::vector<std::string>>>::value_type(key, i));
         // valid_string_map.insert(std::map<std::string, std::shared_ptr<std::string*>>::value_type(key, s));
         // std::cout << "shared ptr size is" << MyFixture::valid_string_map[key].get()->size() << std::endl;
         // std::cout << "first ele of array is" << MyFixture::valid_string_map[key].get()->at(0) << std::endl;

         // invalid_string_map[key] = iv;
      }
      // invalid_str_list = new std::string[state.range(3)*10];
   }
   void TearDown(const ::benchmark::State &state)
   {
      // delete parameters;
      bloom_parameters *parameters;
      // Simple randomizer (optional)
      delete filter;
      bloom_filter *filter;
      // delete[] str_list;
      // delete[] invalid_str_list;
   }
};

// MyFixture::valid_string_map = new std::map<<>>();
// MyFixture::invalid_string_map = new std::map<>();

BENCHMARK_DEFINE_F(MyFixture, TestBloomFilterStringQuery)
(benchmark::State &st)
{

//      if (st.thread_index == 0) {
//     // Setup code here.
//   }
   for (auto _ : st)
   {
      st.PauseTiming(); // Stop timers. They will not count until they are resumed.

      double numFalsePos = 0.0;
      int numItems = st.range(2);
      st.counters["fpr"] = 0;
      st.counters["num_hashes"] = 0;
      st.counters["table_size"] = 0;
      std::string key = "k" + std::to_string(st.range(2)) + std::to_string(st.range(3));
      // if(!MyFixture::valid_string_map[key])
      //    std::cout << "string array is null" << std::endl;
      // else {
      //    std::cout << MyFixture::valid_string_map[key].get()->at(0) << std::endl;
      // }

      auto valid_inputs = MyFixture::valid_string_map[key].get();
      auto invalid_inputs = MyFixture::invalid_string_map[key].get();

      // std::cout << "we are almost at insert with " << valid_inputs->size() << " elements" << std::endl;


      for(int i = 0; i < valid_inputs->size(); i++) {
         std::cout << "inserting ele: " << &valid_inputs->at(i) << std::endl;
         return;
      MyFixture::filter->insert(&valid_inputs->at(i));
      }
      st.ResumeTiming();
      // std::cout << "insertion completed with" << valid_inputs->size() << " elements" << std::endl;


      for (int i = 0; i < numItems; i++)
      {
         if(MyFixture::filter->contains(&invalid_inputs[i])){
            numFalsePos += 1;
         }
      }
      // std::cout << "queries completed with" << valid_inputs->size() << " elements inserted"  << std::endl;

      st.counters["fpr"] = numFalsePos*100/(double) numItems;
      st.counters["num_hashes"] = MyFixture::filter->hash_count();
      st.counters["table_size"] = MyFixture::filter->size();

   }
//      if (st.thread_index == 0) {
//     // Teardown code here.
//   }
}

BENCHMARK_DEFINE_F(MyFixture, TestBloomFilterStringInsertion)
(benchmark::State &st)
{
   MyFixture::filter->clear();
   for (auto _ : st)
   {
      for (int i = 0; i < st.range(2); i++)
      {
         MyFixture::filter->insert(MyFixture::str_list[i]);
      }
      st.PauseTiming(); // Stop timers. They will not count until they are resumed.
      MyFixture::filter->clear();
      st.ResumeTiming();
   }
}

BENCHMARK_DEFINE_F(MyFixture, TestBloomFilterIntInsertion)
(benchmark::State &st)
{
   MyFixture::filter->clear();
   for (auto _ : st)
   {
      for (int i = 0; i < st.range(2); i++)
      {
         MyFixture::filter->insert(i);
      }
      st.PauseTiming(); // Stop timers. They will not count until they are resumed.
      MyFixture::filter->clear();
      st.ResumeTiming();
   }
}

BENCHMARK_DEFINE_F(MyFixture, TestBloomFilterIntQuery)
(benchmark::State &st)
{
   for (int i = 0; i < 8 << 10; i++)
   {
      MyFixture::filter->insert(i);
   }
   for (auto _ : st)
   {
      double numFalsePos = 0.0;
      int numItems = st.range(2);
      st.counters["fpr"] = 0;
      st.counters["num_hashes"] = 0;
      st.counters["table_size"] = 0;
      for (int i = -1; i > st.range(2) * -1; i--)
      {
         MyFixture::filter->contains(i);
      }
      st.counters["fpr"] = numFalsePos*100/(double) numItems;
      st.counters["num_hashes"] = MyFixture::filter->hash_count();
      st.counters["table_size"] = MyFixture::filter->size();
   }
}

// BENCHMARK(BM_generate_random_string)
//     ->Ranges({{8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_compute_optimal_params)
//    ->Ranges({{2, 2 << 10}, {1000, 1000000}});

/* BarTest is NOT registered */
BENCHMARK_REGISTER_F(MyFixture, TestBloomFilterStringQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixture, TestBloomFilterStringInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixture, TestBloomFilterIntQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixture, TestBloomFilterIntInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});

// BENCHMARK(BM_example01);

BENCHMARK_MAIN();
