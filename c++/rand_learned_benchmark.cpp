// #define learned_benchmark_VERSION_MAJOR @learned_benchmark_VERSION_MAJOR@
// #define learned_benchmark_VERSION_MINOR @learned_benchmark_VERSION_MINOR@
// #pragma once
// #ifndef USER_DEBUG_STATEMENTS
// #define USER_DEBUG_STATEMENTS
// #endif

#define TORCH_INPUT_LEN 50

#include <iostream>
#include <string>
#include <benchmark/benchmark.h>
#include <vector>
#include <map>
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

    MyFixtureLearned()
    {
#ifdef USER_DEBUG_STATEMENTS
        std::cout << "fixture init";
        std::cout << "max num val int: " << std::numeric_limits<int>::max();

#endif

        MyFixtureLearned::valid_string_map = {};
        MyFixtureLearned::invalid_string_map = {};
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
            auto vec = gen_string_array(state.range(2), TORCH_INPUT_LEN, true);
            std::shared_ptr<std::vector<std::string>> s = std::make_shared<std::vector<std::string>>(vec);
            std::shared_ptr<std::vector<std::string>> i = std::make_shared<std::vector<std::string>>(gen_string_array(state.range(2), TORCH_INPUT_LEN, false));
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
        std::string key = "k" + std::to_string(st.range(2)) + std::to_string(TORCH_INPUT_LEN);
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
        double num_hashes = (double) MyFixtureLearned::filter->filter->hash_count();
        double table_size = (double) MyFixtureLearned::filter->filter->size();

#ifdef USER_DEBUG_STATEMENTS
        std::cout << "fpr: " << fpr << " numhashes: " << num_hashes << " table_size: " << table_size << std::endl;
#endif
        st.counters.insert({{"fpr", fpr}, {"num_hashes", num_hashes}, {"table_size", table_size}});

        // st.counters["fpr"] = fpr;
        // st.counters["num_hashes"] = MyFixtureLearned::filter->hash_count();
        // st.counters["table_size"] = MyFixtureLearned::filter->size();
    }
}

// BENCHMARK_DEFINE_F(MyFixtureLearned, TestBloomFilterIntQuery)
// (benchmark::State &st)
// {
//     for (int i = 0; i < 8 << 10; i++)
//     {
//         MyFixtureLearned::filter->insert(i);
//     }
//     for (auto _ : st)
//     {
//         double numFalsePos = 0.0;
//         int numItems = st.range(2);
//         st.counters["fpr"] = 0;
//         st.counters["num_hashes"] = 0;
//         st.counters["table_size"] = 0;
//         for (int i = -1; i > st.range(2) * -1; i--)
//         {
//             MyFixtureLearned::filter->contains(i);
//         }
//         st.counters["fpr"] = numFalsePos * 100 / (double)numItems;
//         st.counters["num_hashes"] = MyFixtureLearned::filter->hash_count();
//         st.counters["table_size"] = MyFixtureLearned::filter->size();
//     }
// }

/* BarTest is NOT registered */
BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringQuery)->Ranges({{10, 2 << 12}, {10000, 1000000000}, {1000, 2 << 16}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterStringInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntQuery)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});
// BENCHMARK_REGISTER_F(MyFixtureLearned, TestBloomFilterIntInsertion)->Ranges({{2, 2 << 10}, {1000, 1000000}, {8, 8 << 10}, {10, 100}});

BENCHMARK_MAIN();
