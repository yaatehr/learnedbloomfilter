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
    #TODO load the original model and test what the outputs are on this side. they should match. 
    export_model = embedding_lstm.LSTMBasicX(args, 2, model_path)
    base_model = embedding_lstm.LSTMBasic(args, 2)
    export_model.eval()
    base_model.load_state_dict(torch.load(model_path))
    base_model.eval()
    return export_model, base_model

def export_lstm(args, export_dataset=True):
    model_path = os.path.join(args.root, 'python/modelsaves/explicit_lstm_1.pth')
    model_save_name = "explicit_lstm_1"
    model, base_model = load_lstm(args, model_path)
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

        validation_generator = DataLoader(validation_set, **export_params)
        test_generator = DataLoader(test_set, **export_params)

        def test_model(model, dataset_generator):
            total_errors = 0
            num_inset_predicted =0
            for i, (features, labels) in enumerate(dataset_generator):
                predictions = model(features)
                y_true = labels.cpu().numpy().tolist()
                y_pred = torch.max(predictions, 1)[1].cpu().numpy().tolist()
                # print()
                num_correct = np.sum(np.equal(y_true, y_pred))
                num_inset_predicted += np.sum(y_pred)
                num_incorrect = len(y_true) - num_correct
                total_errors += num_incorrect
                #TODO fix this so that we don't need to use the argmax but that wwe actually predict a onehot or something that works iwth torch max
            print("predicted %d positive labels" % num_inset_predicted)
            return total_errors

        total_errors = test_model(base_model, validation_generator) 
        total_errors += test_model(base_model, test_generator)
        print("base model had %d errors total" % total_errors)

        total_errors = test_model(model, validation_generator) 
        total_errors += test_model(model, test_generator)
        print("export model had %d errors total" % total_errors)

        total_errors = test_model(traced_script_module, validation_generator) 
        total_errors += test_model(traced_script_module, test_generator)
        print("traced_script_module had %d errors total" % total_errors)

        tensor_list, label_list = extract_data(validation_set)
        tensor_list, label_list = extract_data(test_set, tensor_list=tensor_list, label_list=label_list)
        tensors = torch.stack(tensor_list, dim=0)
        labels = torch.FloatTensor(label_list)

        print("final exported tensor dims", tensors.shape)
        print("final exported label dims", labels.shape)
        print("with positive labels", np.sum(labels.detach().numpy()))

        export_values = {
            'data': tensors,
            'labels': labels,
            'model_path': model_path
        }
        export_container = torch.jit.script(container.Container(export_values))
        export_container.save(os.path.join(args.root, "python/modelsaves/%s_container.pt" % model_save_name))
        print("container saved successfully")


def export_blank_model(args):
    model_path = os.path.join(args.root, 'python/modelsaves/explicit_lstm_1.pth')
    model_save_name = "explicit_lstm_1"
    model, base_model = load_lstm(args, model_path)
    empty_model = embedding_lstm.LSTMBasic(args, 2, built_in_dropout=False)
    empty_model.eval()

    model_size = utils.get_model_size(base_model, args)
    export_size = utils.get_model_size(model, args)
    empty_size = utils.get_model_size(empty_model, args)
    a = utils.sizeof_fmt(model_size, suffix="b")
    b = utils.sizeof_fmt(export_size, suffix="b")
    # a,b = 0,0
    c = utils.sizeof_fmt(empty_size, suffix="b")

    print(f"the size of the empty model is {empty_size}\n the size of the base model trained is {a}\n the size of the exoprt model is {b}\n")