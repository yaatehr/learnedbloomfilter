import scipy
import numpy as np
import torch
import string 
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

# build a vector of random strings (fixed len)


# neural net architecures




class LSTMBasic(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=1, bi=False, dropout=0.0):
        super(LSTMBasic, self).__init__()

        self.hidden_dim = hidden_dim
        self.bi = False
        self.num_layers = num_layers

        # The LSTM takes video frame sequences as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(in_dim, self.hidden_dim, self.num_layers, bidirectional=self.bi)
        self.dropout = nn.Dropout(dropout)
        # Linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear((2 if self.bi else 1) * hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros((2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim).to(device)
        h1 = torch.zeros((2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim).to(device)
        return (h0, h1)

    def forward(self, sequence):
        # sequence = sequence.view(len(sequence), 1, -1)
        sequence = sequence[0]
        # print('Sequence shape:', sequence.shape)
        # print('Sequence:', sequence)
        # print('Sequence:', sequence.view(len(sequence), 1, -1))
        # lstm_out, self.hidden = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden)
        lstm_out, self.hidden = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden)
        out_space = self.hidden2out(self.dropout(lstm_out.view(len(sequence), -1)))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores

class GRUBasic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=1, dropout=0):
        super(GRUBasic, self).__init__()

        self.hidden_dim = hidden_dim
        self.bi = False
        self.num_layers = num_layers

        self.gru = nn.GRU(in_dim, hidden_size=self.hidden_dim,
                          bidirectional=self.bi, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear((2 if self.bi else 1) * hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros((2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim).to(device)
        h1 = torch.zeros((2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim).to(device)
        return h0

    def forward(self, sequence):
        print('Sequence shape:', sequence.shape)
        gru_out, self.hidden = self.gru(sequence.view(len(sequence), 1, -1), self.hidden)
        out_space = self.hidden2out(self.dropout(gru_out.view(len(sequence), -1)))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores


class RegressionBasic(nn.Module):

    def __init__(self, in_dim):
        super(RegressionBasic, self).__init__()
        self.fc1 = nn.Linear(in_dim,1)

    def forward(self, x):
        #print('Sequence shape:', sequence.shape)
        out_space = self.fc1(x)
        out_scores = torch.sigmoid(out_space)
        # print(out_scores)
        return out_scores



class AsciiRegression(nn.Module):

    def __init__(self, in_dim):
        super(AsciiRegression, self).__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 1)
        # self.out = nn.Linear(2, 1)

    def forward(self, x):
        #print('Sequence shape:', sequence.shape)
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        features = F.relu(self.fc3(features))
        features = F.sigmoid(self.fc4(features))
        # out_scores = torch.sigmoid(out_space)
        # print(out_scores)
        return features



class AsciiLinear(nn.Module):
    def __init__(self, in_dim):
        super(AsciiLinear, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1)
        # self.out = nn.Linear(2, 1)

    def forward(self, x):
        #print('Sequence shape:', sequence.shape)
        features = F.sigmoid(self.fc1(x))
        return features