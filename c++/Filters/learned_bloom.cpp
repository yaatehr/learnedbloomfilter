// #pragma once

// #ifndef USER_DEBUG_STATEMENTS
// #define USER_DEBUG_STATEMENTS
// #endif

#ifndef MODEL_PATH
// #define MODEL_PATH "/Users/yaatehr/Programs/learnedbloomfilters/CharLevelCnn.pt"
#define MODEL_PATH "/Users/yaatehr/Programs/learnedbloomfilters/python/modelsaves/traced_lstm_non_homogenized.pt"
#endif

#ifndef DATA_PATH
// #define DATA_PATH "/Users/yaatehr/Programs/learnedbloomfilters/container.pt"
#define DATA_PATH "/Users/yaatehr/Programs/learnedbloomfilters/python/modelsaves/traced_lstm_non_homogenized_container.pt"
#endif

#include <iostream>
#include <string>
#include <vector>
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
#include "bloom_filter.hpp"
#include "myUtils.cpp"

class LearnedBloomFilter
{
public:
   bloom_filter *filter;
   std::shared_ptr<torch::jit::script::Module> module;
   std::shared_ptr<at::Tensor> X;
   std::shared_ptr<at::Tensor> Y;
   std::vector<int> validIndices;
   std::vector<int> invalidIndices;

   LearnedBloomFilter(int projected_ele_count, float false_pos_probability)
   {
      try
      {
#ifdef USER_DEBUG_STATEMENTS
         std::cout << "ATTEMPTING TO LOAD CLASSIFIER" << std::endl;
#endif
         // Deserialize the ScriptModule from a file using torch::jit::load().
         module = std::make_shared<torch::jit::script::Module>(torch::jit::load(MODEL_PATH));
         torch::jit::script::Module container = torch::jit::load(DATA_PATH);

         // // Load values by name
         torch::Tensor a = container.hasattr("data") ? container.attr("data").toTensor() : torch::Tensor();
         torch::Tensor b = container.hasattr("labels") ? container.attr("labels").toTensor() : torch::Tensor();
         torch::TensorAccessor<float, 1> accessor = b.accessor<float, 1>();

         X = std::make_shared<torch::Tensor>(a);
         Y = std::make_shared<torch::Tensor>(b);

         int counter = 0;
         for (int i = 0; i < accessor.size(0); i++)
         {
            auto a1 = accessor[i];
            if (a1 > .5)
            {
               validIndices.push_back(counter);
            }
            else
            {
               invalidIndices.push_back(counter);
            }
            counter++;
         }
#ifdef USER_DEBUG_STATEMENTS
         std::cout << "loaded " << validIndices.size() << " positive samples and " << invalidIndices.size() << " negative samples" << std::endl;
#endif
         //   std::cout << tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
      }
      catch (const c10::Error &e)
      {
         std::cerr << "error loading the data/models\n";
         std::cerr << e.what();
      }

      evaluate_classifier();

      bloom_parameters parameters;
      parameters.random_seed = 0xA5A5A5A5;
      parameters.projected_element_count = projected_ele_count;
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "projected_ele_count: " << projected_ele_count << std::endl;
      std::cout << "false_pos_probability: " << 1.0 / (float)false_pos_probability << std::endl;
#endif

      parameters.false_positive_probability = 1.0 / (float)false_pos_probability;
      if (!parameters)
      {
         std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
         return;
      }
      parameters.compute_optimal_parameters();
      filter = new bloom_filter(parameters);
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter init complete~!" << std::endl;
#endif
   }

   bool predict(std::string input)
   {
      auto t = gen_ascii_tensor(input);
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(t);
      torch::Tensor out_tensor = module->forward(inputs).toTensor();
      float prediction = std::round(out_tensor.accessor<float, 2>()[0][0]);
      return prediction > 0.5;
   }

   bool predict(torch::Tensor input) //TODO deprecate
   {
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input);
      torch::Tensor out_tensor = module->forward(inputs).toTensor();
      auto accessor = out_tensor.accessor<float, 2>();
      return accessor[0][0] < accessor[0][1];
   }

   std::vector<bool> predict_batch(torch::Tensor input)
   {
      // std::cout << input << std::endl;
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input);
      torch::Tensor out_tensor = module->forward(inputs).toTensor();
      auto accessor = out_tensor.accessor<float, 2>();
      // std::cout << "first prediction vector: " << out_tensor << std::endl;
      std::vector<bool> outputs;
      for (int i = 0; i < accessor.size(0); i++)
      {
         bool isMalicious = accessor[i][0] < accessor[i][1]; // if label 0 is smaller than label 1, then 0 is less likely
         outputs.push_back(isMalicious);
      }
      return outputs;
   }

   bool query(std::string input)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter query" << std::endl;
#endif
      if (predict(input))
      {
         return true;
      }
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter query returning..." << std::endl;
#endif
      return filter->contains(input);
   }

   std::vector<bool> query(std::vector<std::string> &inputs)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter vector query" << std::endl;
#endif
      std::vector<bool> outputs(inputs.size(), false);
      for (auto s : inputs)
      {
         outputs.push_back(query(s));
      }
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter vector query returning..." << std::endl;
#endif
      return outputs;
   }

   std::vector<bool> batch_query(std::vector<std::string> &input_strings)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query" << std::endl;
#endif
      std::vector<bool> results(input_strings.size(), true);
      auto t = gen_ascii_tensor(std::make_shared<std::vector<std::string>>(input_strings));
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(t);
      std::vector<int> outputs;
      torch::Tensor out_tensor = module->forward(inputs).toTensor();
      auto accessor = out_tensor.accessor<float, 2>();
      for (int i = 0; i < accessor.size(0); ++i)
      {
         outputs.push_back(std::round(accessor[i][0]));
      }
      int index = 0;
      for (auto s : outputs)
      {
         if (!s)
         {
            auto update = filter->contains(input_strings.at(index));
            results[index] = update;
         }
         index++;
      }
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query returning..." << std::endl;
#endif
      return results;
   }

   /**
    * returns a count of the number of predicted positives from ensemble
    */
   int batch_query_count(std::vector<std::string> &input_strings)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query count" << std::endl;
#endif
      std::vector<bool> results(input_strings.size(), true);
      auto t = gen_ascii_tensor(std::make_shared<std::vector<std::string>>(input_strings));
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(t);
      std::vector<int> outputs;
      torch::Tensor out_tensor = module->forward(inputs).toTensor();
      auto accessor = out_tensor.accessor<float, 2>();
      for (int i = 0; i < accessor.size(0); ++i)
      {
         outputs.push_back(std::round(accessor[i][0]));
      }
      int index = 0;
      int sum_false_pos = 0;
      for (auto s : outputs)
      {
         if (s)
         {
            sum_false_pos++;
         }
         else
         {
            auto update = filter->contains(input_strings.at(index));
            if (update)
            {
               sum_false_pos++;
            }
         }
         index++;
      }
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query count returning..." << std::endl;
#endif
      return sum_false_pos;
   }

   void evaluate_classifier()
   {

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter Evaluating classifier on all data" << std::endl;
#endif
      auto label_accessor = Y->accessor<float, 1>();
      unsigned int num_correct = 0;
      int num_positive_samples = 0;
      int num_positive_predictions = 0;
      auto predictions = predict_batch(*X);
      for (unsigned long j = 0; j < predictions.size(); j++)
      {
         if (predictions[j])
         {
            num_positive_predictions++;
         }
         bool label = label_accessor[j] >= .5;
         if (label)
         {
            num_positive_samples++;
         }
         if (predictions[j] == label)
         {
            num_correct++;
         }
      }

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter classifier accuracy was: " << (float)num_correct / (float)label_accessor.size(0) << std::endl;
      std::cout << "with: " << num_positive_samples << " positive samples" << std::endl;
      std::cout << "and: " << num_positive_predictions << " positive predictions" << std::endl;
#endif
   }

   //    /**
   //     * returns a count of the number of predicted positives from ensemble
   //     */
   //    int count_batch_classifier_predictions(std::vector<unsigned int> &indices)
   //    {
   // #ifdef USER_DEBUG_STATEMENTS
   //       std::cout << "Learned bloom filter batch query count" << std::endl;
   // #endif

   //       std::vector<bool> outputs;
   //       auto accessor = X->accessor<float, 3>();
   //       for (auto i : indices) {
   //             torch::Tensor valid_tensor = torch::from_blob(accessor[i].data(), {124, 32});
   //             outputs.push_back(predict(valid_tensor));
   //       }

   //       int index = 0;
   //       int sum_false_pos = 0;
   //       for (auto s : outputs)
   //       {
   //          if (s)
   //          {
   //             sum_false_pos++;
   //          }
   //          else
   //          {
   //             auto update = filter->contains(input_strings.at(index));
   //             if (update)
   //             {
   //                sum_false_pos++;
   //             }
   //          }
   //          index++;
   //       }
   // #ifdef USER_DEBUG_STATEMENTS
   //       std::cout << "Learned bloom filter batch query count returning..." << std::endl;
   // #endif
   //       return sum_false_pos;
   //    }

   void insert(std::string input)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter inserting " << std::endl;
#endif
      if (predict(input))
      {
         return;
      }
      filter->insert(input);
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter insert returning " << std::endl;
#endif
   }

   void insert(std::vector<std::string> &input_strings)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter vector insert NOT IMPLEMENTED" << std::endl;
#endif
      //TODO
   }

   // std::size_t hash_count()
   // {
   //    return filter->hash_count();
   // }
};
