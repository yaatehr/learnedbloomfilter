import gc
import sys, os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True
from sklearn.model_selection import train_test_split
from ascii_regression_classifier import *
import numpy as np
import matplotlib.pyplot as plt
from utils import topNError, saveErrorGraph, get_dataset, AsciiStringData, AsciiStringDataCaps
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
import os
import pickle
import re

dirname = os.path.dirname(__file__)
now = datetime.now() # current date and time

multiGPU = torch.cuda.device_count() > 1


def rnnEpoch(model, loader, device, criterion,  epoch, output_period=-1, optimizer=None):

    running_loss = 0.0
    num_batches = len(loader)
    errors = np.zeros(1)
    for batch_num, (features, labels) in enumerate(loader, 1):
        inputs = [features]

        inputs = torch.cat(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print('INPUTS SHAPE:', inputs.shape)
        # print('LABELS SHAPE:', labels.shape)
        # print("OUT SHAPE:", model(inputs))

        outputs = (model(inputs.float())[-1,:]).view(labels.shape[0], -1)
        # print('OUTPUTS SHAPE:', outputs)
        loss = criterion(outputs, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        running_loss += loss.item()

        if batch_num % output_period == 0:
            print('[%d:%.2f] loss: %.3f' % (
                epoch, batch_num*1.0/num_batches,
                running_loss/output_period
                ))
            # print('OUTPUTS:', outputs)
            running_loss = 0.0
        gc.collect()
        errors += topNError(outputs, labels, [1,2], False)
    return errors

def regressionEpoch(model, loader, device, criterion,  epoch, output_period, optimizer=None):

    running_loss = 0.0
    num_batches = len(loader)
    errors = 0
    for batch_num, (features, labels) in enumerate(loader, 1):
        inputs = features

        # inputs = torch.cat(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        # print('INPUTS SHAPE:', inputs.shape)
        # print('LABELS SHAPE:', labels.shape)
        # print("OUT SHAPE:", model(inputs.float()))

        outputs = model(inputs.float())
        # print('OUTPUTS SHAPE:', outputs.shape)
        loss = criterion(outputs, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()

        if batch_num % output_period == 0:
            print('[%d:%.2f] loss: %.3f' % (
                epoch, batch_num*1.0/num_batches,
                running_loss/output_period
                ))
            # print('OUTPUTS:', outputs)
            running_loss = 0.0
        gc.collect()
        classifications = np.where(outputs.detach().numpy() > .5, 1, 0)
        # print((classifications[:,0] + labels.numpy()[:,0]))
        # numCorrect = np.sum(np.where((classifications[:,0] + labels.numpy()) != 1, 1, 0))
        numCorrect = np.sum(np.where((classifications[:,0] + labels.numpy()[:,0]) != 1, 1, 0))
        # print(classifications.shape, labels.numpy().shape, np.where((classifications + labels.numpy()) != 1, 1, 0).shape, numCorrect)

        errors += int(len(labels) - numCorrect)
    # print(errors)
    return errors



def trainRNN():

    # inputDim = 15

    # GX, Y, X = build_features(10000, inputDim)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    num_epochs = 10
    output_period = 100
    batch_size = 1

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # dataset = get_dataset(0) #get 10 character long string dataset
    input_dim = 10
    dataset = AsciiStringData(5000, input_dim)

    # Creating data indices for training and validation splits:
    validation_split = .2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)


    hiddenDim = 6
    # model = LSTMBasic(1, hiddenDim, num_classes=2, num_layers=2, dropout=0.0)
    model = GRUBasic(input_dim, hiddenDim, num_classes=2, num_layers=1, dropout=0.0)
    model = model.float()
    if torch.cuda.is_available():
        if multiGPU:
            model = nn.DataParallel(model)
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_loader, val_loader = dataset.video_train_loader(batch_size), dataset.video_val_loader(batch_size)
    numTrainSamples = len(train_loader) * batch_size
    numValSamples = len(val_loader) * batch_size

    criterion = nn.NLLLoss().to(device)
    epochs = np.arange(1, num_epochs+1)

    trainErrors, valErrors = [], []
    print('Training RNN')
    for epoch in epochs:    
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()
        trainErrors.append(rnnEpoch(model, train_loader, device, criterion, output_period, epoch, optimizer=optimizer)/numTrainSamples)
        gc.collect()
        # save after everay epoch
        #torch.save(model.state_dict(), "models/model" + str(modelNum) + ".%d" % epoch)
        model.eval()
        valErrors.append(rnnEpoch(model, val_loader, device, criterion, output_period, epoch)/numValSamples)
        optimizer.step(valErrors[-1].all())
        gc.collect()
        print('Epoch ' + str(epoch) + ':', 'Train error:', trainErrors[-1], ', Validation error:', valErrors[-1])

    trainErrors = np.array(trainErrors)
    valErrors = np.array(valErrors)
    trainClassificationErrors, trainTop2Errors = trainErrors[:,0], trainErrors[:,1]
    valClassificationErrors, valTop2Errors = valErrors[:,0], valErrors[:,1]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors')
    plt.plot(epochs, trainClassificationErrors, label="Train Classification Error")
    plt.plot(epochs, trainTop2Errors, label="Train Top 2 Error")
    plt.plot(epochs, valClassificationErrors, label="Validation Classification Error")
    plt.plot(epochs, valTop2Errors, label="Validation Top 2 Error")
    plt.legend(loc='best')
    plt.savefig('rnnErrors.png')
    print('Finished training RNN')


def trainLinearClassifier():
    date_time = now.strftime("%m-%d-%H:%M:%S")

    num_epochs = 10
    output_period = 10
    batch_size = 1000

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # dataset = get_dataset(0) #get 10 character long string dataset
    input_dim = 50
    # dataset = AsciiStringData(100000, input_dim)
    dataset = AsciiStringDataCaps(100000, input_dim, noise_ratio=.0001)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)


    hiddenDim = 6
    model = AsciiLinear(input_dim)
    model = model.float()
    if torch.cuda.is_available():
        if multiGPU:
            model = nn.DataParallel(model)
        model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    numTrainSamples = len(train_loader) * batch_size
    numValSamples = len(val_loader) * batch_size
    print(numTrainSamples, numValSamples)

    criterion = nn.BCELoss().to(device)
    epochs = np.arange(1, num_epochs+1)

    trainErrors, valErrors = [], []
    print('Training Regression Model')
    for epoch in epochs:    
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()
        trainErrors.append(regressionEpoch(model, train_loader, device, criterion, epoch, output_period, optimizer=optimizer)/numTrainSamples)
        gc.collect()
        # save after everay epoch
        torch.save(model.state_dict(), "models/model" + date_time + ".%d" % epoch)
        model.eval()
        valErrors.append(regressionEpoch(model, val_loader, device, criterion, epoch, output_period)/numValSamples)
        optimizer.step()
        gc.collect()
        print('Epoch ' + str(epoch) + ':', 'Train error:', trainErrors[-1], ', Validation error:', valErrors[-1])

    trainErrors = np.array(trainErrors)
    valErrors = np.array(valErrors)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors')
    plt.plot(epochs, trainErrors, label="Train Classification Error")
    plt.plot(epochs, valErrors, label="Validation Classification Error")
    plt.legend(loc='best')
    plt.savefig('rnnErrors-insetonly.png')
    print('Finished training Regression Model')

def loadMostRecentModel():
    model = AsciiLinear(50)
    models_path = os.path.join(dirname,'models/')
    paths = os.walk(models_path)
    print(paths)
    dates = []
    for p in paths:
        # print(p)
        for i, path in enumerate(p[2]):
            # print(path[5:])
            date_of_run = datetime.strptime(path[5:].split(".")[0], "%m-%d-%H:%M:%S")
            dates.append((date_of_run, i))
            # print(date_of_run)
        dates = sorted(dates, key=lambda x: x[0], reverse=True)
        most_recent_modelpath = p[2][dates[0][1]]
        break
    # print(paths)
    print(most_recent_modelpath)
    if os.path.isfile(models_path + most_recent_modelpath):
        bestState = torch.load(models_path + most_recent_modelpath)
    else:
        raise Exception('Model file not found.')
    model.load_state_dict(bestState)
    return model

def loadCharCnn():
    model = pickle.load(open(os.path.join(dirname, '../CharLevelCnn.pkl'), 'rb'))
    models_path = os.path.join(dirname,'models/')
    bestState = torch.load(models_path + "CharLevelCnn.pkl")
    model.load_state_dict(bestState)
    return model

def train_main():
    trainRNN()

import string

URL_ALPHABET= string.ascii_letters + string.digits + "_.~-" + ":/?#[]@!$&'()*+,;="
URL_DELIM = ":/?#."
DEFAULT_ALPHABET="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
URL_REGEX = "((http|ftp|https):\\/\\/)?[\\w\\-_]+(\\.[\\w\\-_]+)+([\\w\\-\\.,@?^=%&amp;:/~\\+#]*[\\w\\-\\@?^=%&amp;/~\\+#])?"


trainRNN()
