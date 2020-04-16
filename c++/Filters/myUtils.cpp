#pragma once

#define learned_benchmark_VERSION_MAJOR @learned_benchmark_VERSION_MAJOR@
#define learned_benchmark_VERSION_MINOR @learned_benchmark_VERSION_MINOR@

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include <numeric>
#include <random>
#include <iterator>
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Tensor.h>
#include <math.h> //for std::round
#include <chrono>

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

std::string random_string(size_t length)
{
   auto randchar = []() -> char {
      const char charset[] =
          "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
   };
   std::string str(length, 0);
   std::generate_n(str.begin(), length, randchar);
   return str;
}
std::string random_in_string(size_t length)
{
   auto randchar = []() -> char {
      const char charset[] =
          "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
   };
   std::string str(length, 0);
   std::generate_n(str.begin(), length, randchar);
   return str;
}

std::string random_out_string(size_t length)
{
   auto randchar = []() -> char {
      const char charset[] =
          "0123456789"
          "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
   };
   std::string str(length, 0);
   std::generate_n(str.begin(), length, randchar);
   return str;
}

std::vector<std::string> gen_string_array(int length, size_t str_len, bool inset)
{
   std::vector<std::string> output = {};
   for (int i = 0; i < length; i++)
   {
      std::string rand_string = inset ? random_in_string(str_len) : random_out_string(str_len);

      output.push_back(rand_string);
   }
   return output;
}


std::vector<unsigned int> gen_ascii_chars(std::string input_string)
{
   std::vector<unsigned int> output;

   for (auto c : input_string)
   {
      output.push_back(int(c));
   }
   return output;
}

torch::Tensor gen_ascii_tensor(std::string input)
{
   int stringLen = input.length();
   std::vector<float> output(stringLen, 0.0);

   for (int j = 0; j < stringLen; j++)
   {
      output[j] = int(input[j]);
   }

   torch::Tensor data_tensor = torch::from_blob(output.data(), {1, stringLen});

   return data_tensor;
}

torch::Tensor gen_ascii_tensor(int numFeatures, int stringLen)
{
   std::vector<float> output(numFeatures * stringLen, 0.0);
   for (int i = 0; i < numFeatures; i++)
   {
      auto randString = random_string(stringLen);
      for (int j = 0; j < stringLen; j++)
      {
         output[i * stringLen + j] = int(randString[j]);
      }
   }
   torch::Tensor data_tensor = torch::from_blob(output.data(), {numFeatures, stringLen});

   return data_tensor;
}
torch::Tensor gen_ascii_tensor(std::shared_ptr<std::vector<std::string>> inputStrings)
{
   int stringLen = inputStrings->at(0).length();
   std::vector<float> output(inputStrings->size() * stringLen, 0.0);
   for (unsigned long i = 0; i < inputStrings->size(); i++)
   {
      auto randString = inputStrings->at(i);
      for (int j = 0; j < stringLen; j++)
      {
         output[i * stringLen + j] = int(randString[j]);
      }
   }
   torch::Tensor data_tensor = torch::from_blob(output.data(), {(int)inputStrings->size(), stringLen});

   return data_tensor;
}

std::vector<std::string> gen_string_array(int length, size_t str_len, const char *prefix)
{
   std::vector<std::string> output = {};
   for (int i = 0; i < length; i++)
   {
      output.push_back(prefix + random_string(str_len));
   }
   // std::vector<std::string> out_vec(output, output + sizeof(*output) / sizeof(std::string));
   return output;
}

std::vector<std::vector<unsigned int>> gen_ascii_string_array(int length, size_t str_len, const char *prefix)
{
   std::vector<std::vector<unsigned int>> output = {};
   for (int i = 0; i < length; i++)
   {
      output.push_back(gen_ascii_chars(prefix + random_string(str_len)));
   }
   return output;
}

template<typename K>
std::vector<unsigned int> gen_random_indices(std::vector<K> &input_vec)
{
   std::vector<unsigned int> indices(input_vec.size());
   std::iota(indices.begin(), indices.end(), 0);
   unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count(); // time seed
   auto rng = std::default_random_engine{seed1};
   std::shuffle(indices.begin(), indices.end(), rng);

   return indices;
}

template<typename K>
std::vector<unsigned int> gen_random_indices(std::vector<K> &input_vec, unsigned int desired_num_eles) {
      std::vector<unsigned int> indices = gen_random_indices(input_vec);
      indices.resize(desired_num_eles);
      return indices;
}

template<typename K>
std::vector<K> select_random_vector_subset(std::vector<K> &input_vec, int desired_num_eles)
{
   std::vector<K> output = std::vector<K>();
   auto indices = gen_random_indices(input_vec, desired_num_eles);
   for (auto i : indices)
   {
      output.push_back(input_vec[i]);
   }
   return output;
}

torch::Tensor select_tensor_subset(torch::Tensor data, std::vector<int> &index_vec, int desired_num_eles)
{
   std::vector<int> indices(desired_num_eles);
   std::copy_n(index_vec.begin(), desired_num_eles, indices.begin());
   auto index_tensor = torch::tensor(indices).toType(torch::kInt64);
   torch::Tensor output = data.index(index_tensor); 
   return output;
}

torch::Tensor select_random_tensor_subset(torch::Tensor data, std::vector<int> &index_vec, int desired_num_eles)
{
   std::vector<int> indices = select_random_vector_subset(index_vec, desired_num_eles);
   return select_tensor_subset(data, indices, desired_num_eles);
}

// Read in the csv file and return keys and labels as vector of tuples.
std::vector<std::tuple<std::string /*key*/, int64_t /*label*/>> ReadCsv(std::string& location) {
    std::fstream in(location, std::ios::in);
    std::string line;
    std::string key;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;

    while (getline(in, line))
    {
        std::stringstream s(line);
        getline(s, key, ',');
        getline(s, label, ',');
                                    // converts string to integer
        csv.push_back(std::make_tuple( key, stoi(label)));
    }
    return csv;
}

std::vector<std::string> load_dataset(std::string location) {
   std::string test_set_location = location + "/test_set.txt";
   std::string validation_set_location = location + "/validation_set.txt";

    std::fstream in(validation_set_location, std::ios::in);
    std::string line;
    std::string key;
    std::string label;
   std::vector<std::string> keys;
   std::vector<int> labels; // TODO validate labels and return tup or pointers for these?
   std::cout << "attempting to load from " + validation_set_location << std::endl;

    while (getline(in, line))
    {
        std::stringstream s(line);
        s >> label >> key;
                                    // converts string to integer
      labels.push_back(stoi(label));
      keys.push_back(key);
    }
    in.close();
   std::cout << "attempting to load from " << test_set_location << std::endl;


   std::fstream in2(test_set_location, std::ios::in);
       while (getline(in2, line))
    {
        std::stringstream s(line);
        s >> label >> key;
                                    // converts string to integer
      labels.push_back(stoi(label));
      keys.push_back(key);
    }

    return keys;
}



template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end);
  return linspaced;
}





// int main()
// {
//    //  auto i = gen_ascii_string_array(5000, 10, "#");
//    //  auto t = gen_ascii_tensor(20, 20);
//    std::vector<std::string> stringVec = {"string1string1123124", "string2str12ing11234"};
//    std::string testString = "string1string1123124";

//    auto learnedBloom = new LearnedBloomFilter(100, .1);
//    std::cout << learnedBloom->query(testString) <<std::endl;
//    std::cout << learnedBloom->batch_query_count(stringVec) << std::endl;


//    // auto t = gen_ascii_tensor(std::make_shared<std::vector<std::string>>(stringVec));
//    // std::vector<torch::jit::IValue> inputs;
//    // inputs.push_back(t);
//    // torch::jit::script::Module module;
//    // try
//    // {
//    //    // Deserialize the ScriptModule from a file using torch::jit::load().
//    //    module = torch::jit::load("/Users/yaatehr/Programs/learnedbloomfilters/traced_ascii_regression.pt");
//    // }
//    // catch (const c10::Error &e)
//    // {
//    //    std::cerr << "error loading the model\n";
//    // }
//    // // std::cout << module.shape();
//    // std::vector<int> outputs;
//    // torch::Tensor out_tensor = module.forward(inputs).toTensor();
//    // auto accessor = out_tensor.accessor<float, 2>();
//    // for (int i = 0; i < accessor.size(0); ++i)
//    // {
//    //    outputs.push_back(std::round(accessor[i][0]));
//    // }
//    // std::cout << out_tensor.size(1);
//    // std::cout << out_tensor[0][0];
// }