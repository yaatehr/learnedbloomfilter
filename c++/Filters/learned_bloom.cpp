// #pragma once

// #ifndef USER_DEBUG_STATEMENTS
// #define USER_DEBUG_STATEMENTS
// #endif

#ifndef MODEL_PATH
// #define MODEL_PATH "/Users/yaatehr/Programs/learnedbloomfilters/CharLevelCnn.pt"
// #define MODEL_PATH "/Users/yaatehr/Programs/learnedbloomfilters/python/modelsaves/traced_lstm_non_homogenized.pt"
#define MODEL_PATH "/home/yaatehr/programs/learnedbloomfilter/python/modelsaves/timestamp_lstm_1.pt"
#endif

#ifndef DATA_PATH
// #define DATA_PATH "/Users/yaatehr/Programs/learnedbloomfilters/container.pt"
#define DATA_PATH "/home/yaatehr/programs/learnedbloomfilter/python/modelsaves/timestamp_lstm_1_container.pt"
#endif
#ifndef DATASET_PATH
#define DATASET_PATH "/home/yaatehr/programs/learnedbloomfilter/input/timestamp_dataset"
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
   std::shared_ptr<torch::jit::script::Module> classifier;
   std::shared_ptr<at::Tensor> X;
   std::shared_ptr<at::Tensor> Y;
   std::vector<int> validIndices;
   std::vector<int> invalidIndices;
   std::vector<std::string> data_strings;
   double tau;

   static std::tuple<std::shared_ptr<torch::Tensor> /*Data*/,
                     std::shared_ptr<torch::Tensor> /*labels*/,
                     std::vector<int> /*validIndices*/,
                     std::vector<int> /*invalidIndices*/ > load_tensor_container(std::string data_path, int max_num_eles) {
      try
      {
          torch::jit::script::Module container = torch::jit::load(DATA_PATH);

         // check for valid container
         if(! (container.hasattr("data") && container.hasattr("labels"))) {
            throw new std::logic_error("data path " + data_path + " points to a container without data and label attributes. Check your pytorch export \n");
         }
         
         torch::Tensor a = container.attr("data").toTensor();
         torch::Tensor b = container.attr("labels").toTensor();
         torch::TensorAccessor<float, 1> accessor = b.accessor<float, 1>();

         auto X = std::make_shared<torch::Tensor>(a);
         auto Y = std::make_shared<torch::Tensor>(b);

         int counter = 0;
         std::vector<int> validIndices;
         std::vector<int> invalidIndices;

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
         if(max_num_eles > 0) {
            validIndices = select_random_vector_subset(validIndices, max_num_eles);
            invalidIndices = select_random_vector_subset(invalidIndices, max_num_eles);
            // std::vector<int> allIndices;
            // allIndices.insert(allIndices.end(), validIndices.begin(), validIndices.end());
            // allIndices.insert(allIndices.end(), invalidIndices.begin(), invalidIndices.end());

            // X = std::make_shared<torch::Tensor>(select_tensor_subset(a, allIndices, max_num_eles*2));
            // Y = std::make_shared<torch::Tensor>(select_tensor_subset(b, allIndices, max_num_eles*2));
         }
#ifdef USER_DEBUG_STATEMENTS
         std::cout << "loaded " << validIndices.size() << " positive samples and " << invalidIndices.size() << " negative samples" << std::endl;
#endif
         //   std::cout << tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

         return std::make_tuple(X, Y, validIndices, invalidIndices);
      }
      catch (const c10::Error &e)
      {
         std::cerr << "error loading the data container \n" << std::endl;
         std::cerr << e.what() << std::endl;
         throw e;
      }     
   }

   static std::shared_ptr<torch::jit::script::Module> load_classifier(std::string model_path) 
   {
      try
      {
         return std::make_shared<torch::jit::script::Module>(torch::jit::load(MODEL_PATH));
      }
      catch (const c10::Error &e)
      {
         std::cerr << "error loading the classifier \n" << std::endl;
         std::cerr << e.what() << std::endl;
         throw e;
      }     
   }


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
//                                  Constructors
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////


   LearnedBloomFilter(int projected_ele_count, float false_pos_probability)
   {

#ifdef USER_DEBUG_STATEMENTS
         std::cout << "ATTEMPTING TO LOAD CLASSIFIER" << std::endl;
#endif
      classifier = load_classifier(MODEL_PATH);
      std::tie(X, Y, validIndices, invalidIndices) = load_tensor_container(DATA_PATH, projected_ele_count);
      tau = 0.5;
      data_strings = load_dataset(DATASET_PATH);
      //evaluate_classifier();
      init_generic_bloom(projected_ele_count, false_pos_probability);
   }

   LearnedBloomFilter(int p,
                      float f, 
                     std::shared_ptr<torch::jit::script::Module> c,
                     std::shared_ptr<torch::Tensor> x,
                     std::shared_ptr<torch::Tensor> y,
                     std::vector<int> v,
                     std::vector<int> i,
                     std::vector<std::string> d): classifier(c), X(x), Y(y), validIndices(v), invalidIndices(i), data_strings(d) 
   {
      tau = 0.5;
      //evaluate_classifier();
      init_generic_bloom(p, f);
   }

   ~LearnedBloomFilter() {
         delete filter;
         // delete classifier;
         // delete X;
         // delete Y;
         // delete validIndices;
         // delete invalidIndices;
         // delete data_strings;
   }


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
//                                  Member Functions
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////


   void set_tau(double t) {
      if(tau > 1 || tau < 0) { 
         return;
      }
      tau = t;
   }

   bool predict(torch::Tensor input) //TODO deprecate
   {
      if (tau ==1) {
         return false;
      }
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input);
      float out = classifier->forward(inputs).toTensor().item().to<float>();
      return out > tau;
   }
   // Always false prediction override
   // bool predict(torch::Tensor input)
   // {
   //    return false;
   // }

   std::vector<bool> predict_batch(torch::Tensor input)
   {
      if (tau == 1) {
         std::vector<bool> vec(input.accessor<float, 2>().size(0), false);
         return vec;
      }
      // std::cout << input << std::endl;
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input);
      torch::Tensor out_tensor = classifier->forward(inputs).toTensor();
      auto accessor = out_tensor.accessor<float, 1>();
      // std::cout << "first prediction vector: " << out_tensor << std::endl;
      std::vector<bool> outputs;
      for (int i = 0; i < accessor.size(0); i++)
      {
         bool isMalicious = accessor[i] > tau; 
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


   bool query(int index){ 
      std::vector<int> index_vec = {index};
      auto tensor = select_tensor_subset(*X, index_vec, 1);
      auto prediction = predict(tensor);
      if(prediction){
         return true;
      } else {
         return filter->contains(data_strings[index]);
      }
   }
   std::vector<bool> query(std::vector<int> indices){ 
      std::vector<bool> outputs;
      for (auto i : indices) {
         outputs.push_back(query(i));
      }
      return outputs;
   }


   void insert(int index){  // todo is this void?
      std::vector<int> index_vec = {index};
      auto tensor = select_tensor_subset(*X, index_vec, 1);
      auto prediction = predict(tensor);
      if (!prediction)
      {
            filter->insert(data_strings[index]);
      }
   }

   void insert(std::vector<int> indices){ 
      for (auto i : indices) {
         insert(i);
      }
   }

   /**
    * returns a count of the number of predicted positives from ensemble
    */
   int batch_query_count(std::vector<int> indices, bool valid_indices)
   {
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query count" << std::endl;
#endif
      int num_false_positives = 0;
      auto predictions = query(indices);
      for(auto p : predictions) {
         if(p ^ valid_indices){
            num_false_positives++;
         }
      }

#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter batch query count returning..." << std::endl;
#endif
      return num_false_positives;
   }

///////////////////////////////////////////////////?
///////////////////////////////////////////////////?
///////////////////////////////////////////////////?
//       String methods (TODO move to new class)
///////////////////////////////////////////////////?
///////////////////////////////////////////////////?

   bool predict(std::string input)
   {
      auto t = gen_ascii_tensor(input);
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(t);
      torch::Tensor out_tensor = classifier->forward(inputs).toTensor();
      float prediction = std::round(out_tensor.accessor<float, 2>()[0][0]);
      return prediction > 0.5;
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
      torch::Tensor out_tensor = classifier->forward(inputs).toTensor();
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
      torch::Tensor out_tensor = classifier->forward(inputs).toTensor();
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


private:

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

   void init_generic_bloom(int projected_ele_count, float false_pos_probability) {
            bloom_parameters parameters;
      parameters.random_seed = 0xA5A5A5A5;
      parameters.projected_element_count = projected_ele_count;
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "projected_ele_count: " << projected_ele_count << std::endl;
      std::cout << "false_pos_probability: " << 1.0 / (float)false_pos_probability << std::endl;
#endif

      parameters.false_positive_probability = false_pos_probability;
      if (!parameters)
      {
         std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
         return;
      }
      parameters.compute_optimal_parameters();
      filter = new bloom_filter(parameters);//TODO memory leak check, make sure filter is destructed
#ifdef USER_DEBUG_STATEMENTS
      std::cout << "Learned bloom filter init complete~!" << std::endl;
#endif
   }

};
