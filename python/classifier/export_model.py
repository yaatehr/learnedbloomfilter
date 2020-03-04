import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from classifier import embedding_cnn, embedding_lstm, embedding_rnn, utils
from data_intake import data_loader, processing

# import torch.jit


def load_lstm(args, model_path):
    model = embedding_lstm.LSTMBasicX(args, 2, model_path)
    model.eval()
    return model

def export_lstm(args):
    model_path = os.path.join(args.root, 'python/modelsaves/second_run_train_test_val_split.pth')
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
    traced_script_module.save(os.path.join(args.root, "python/modelsaves/traced_lstm_non_homogenized.pt"))
    print("MODEL SAVED!")
