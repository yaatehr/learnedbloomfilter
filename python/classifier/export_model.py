import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from classifier import embedding_cnn, embedding_lstm, embedding_rnn, utils, container
from data_intake import data_loader, processing
import pickle
# import torch.jit


def load_lstm(args, model_path):
    model = embedding_lstm.LSTMBasicX(args, 2, model_path)
    model.eval()
    return model

def export_lstm(args, export_dataset=True):
    model_path = os.path.join(args.root, 'python/modelsaves/second_run_train_test_val_split.pth')
    model_save_name = "traced_lstm_non_homogenized"
    model = load_lstm(args, model_path)
    print("model loaded")
    dataset = data_loader.EncodedStringLabelDataset(args, init_tuple=(["test"]*2000, ["label"]*2000, [0]*2000, None))
    export_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }
    export_generator = DataLoader(dataset, **export_params)
    i, (feats, labels) = list(enumerate(export_generator))[0]
    feats = feats.cpu().detach()

    print("attempting to trace model")

    traced_script_module = torch.jit.script(model)
    print("saving model")
    traced_script_module.save(os.path.join(args.root, "python/modelsaves/%s.pt" % model_save_name))
    print("MODEL SAVED!")

    if export_dataset:
        dataset_path = os.path.join(args.root, "input/dataset/train_val_test.pkl")
        try:
            training_set, validation_set, test_set = pickle.load(open(dataset_path, 'rb'))
            print("loaded train set: ", training_set.counter)
            print("loaded val set: ", validation_set.counter)
            print("loaded test set: ", test_set.counter)
        except:
            urls_by_category_path = os.path.join(args.root, "python/scripts/url_load_backup.pkl")
            with open(urls_by_category_path, 'rb') as fp:
                urls_by_category = pickle.load(fp)
            training_set = data_loader.EncodedStringLabelDataset(args, urls_by_category=urls_by_category)
            init_tuple = training_set.split_train_val()
            validation_set = data_loader.EncodedStringLabelDataset(args, init_tuple=init_tuple)
            init_tuple = training_set.split_train_val(split_size=.2222222)
            test_set = data_loader.EncodedStringLabelDataset(args, init_tuple=init_tuple)
            #initialize a train val test split of 70, 10, 20
            validation_set.select_subset(balanceWeights=True)
            test_set.select_subset(balanceWeights=True)

        #combine the validation and test_set pieces of interest
        def extract_data(dataset,tensor_list=None, label_list=None):
            tensor_list = [] if not tensor_list else tensor_list
            label_list = [] if not label_list else label_list
            assert len(tensor_list) == len(label_list), "label lists must match"

            for data, label in dataset:
                tensor_list.append(data)
                label_list.append(label)
            
            return tensor_list, label_list

        tensor_list, label_list = extract_data(validation_set)
        tensor_list, label_list = extract_data(test_set, tensor_list=tensor_list, label_list=label_list)
        tensors = torch.stack(tensor_list, dim=0)
        labels = torch.IntTensor(label_list)

        print("final exported tensor dims", tensors.shape)
        print("final exported label dims", labels.shape)


        export_values = {
            'data': tensors,
            'labels': labels,
            'model_path': model_path
        }
        export_container = torch.jit.script(container.Container(export_values))
        export_container.save(os.path.join(args.root, "python/modelsaves/%s_container.pt" % model_save_name))
        print("container saved successfully")
