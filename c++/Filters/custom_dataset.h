#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>

#include "myUtils.cpp"

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        std::vector<std::tuple<std::string /*key*/, int64_t /*label*/>> csv_;

    public:
        explicit CustomDataset(std::string& file_names_csv)
            // Load csv file with file locations and labels.
            : csv_(ReadCsv(file_names_csv)) {

        };

        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            std::string key = std::get<0>(csv_[index]);
            int64_t label = std::get<1>(csv_[index]);


            // Convert the key and label to a tensor.
            //TODO make string embedding logic 

            torch::Tensor string_embedding_tensor;

            torch::Tensor label_tensor = torch::full({1}, label);

            return {string_embedding_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return csv_.size();
        };
};
