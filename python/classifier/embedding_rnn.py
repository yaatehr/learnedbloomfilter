import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUBasic(nn.Module):
    def __init__(self, args, num_classes=1, built_in_dropout=False):
        super(GRUBasic, self).__init__()
        dropout = args.dropout_input
        self.hidden_dim = args.hidden_dim
        self.bi = args.bidirectional == 1
        self.num_layers = 1 #TODO remove this param or fix this
        self.num_classes = num_classes
        self.embedding_depth = (
            (args.number_of_characters + len(args.extra_characters))*args.max_embedding_length
            if args.use_char_encoding
            else args.embedding_depth * args.max_embedding_length
        )
        self.built_in_dropout = built_in_dropout

        if not built_in_dropout:
            self.gru = nn.GRU(
                self.embedding_depth,
                hidden_size=self.hidden_dim,
                bidirectional=self.bi,
                batch_first=False,
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.gru = nn.GRU(
                self.embedding_depth,
                hidden_size=self.hidden_dim,
                bidirectional=self.bi,
                batch_first=False,
                dropout=dropout,
            )

        # Linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear(
            (2 if self.bi else 1) * self.hidden_dim, self.num_classes
        )
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(
            (2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim
        ).to(device)
        return h0

    def forward(self, sequence):
        # print('Sequence shape:', sequence.shape)
        gru_out, self.hidden = self.gru(
            sequence.view(len(sequence), 1, -1), self.hidden
        )
        if self.built_in_dropout:
            out_space = self.hidden2out(
                self.dropout(gru_out[:, -1].view(len(sequence), -1))
            )
        else:
            out_space = self.hidden2out(gru_out[:, -1])
        # self.hidden = hidden
        #TODO experiment with adding a relu here, does it increase accuracy?
        out_scores = torch.sigmoid(out_space).squeeze()
        return out_scores


class GRUBasicX(nn.Module):
    def __init__(self, args, model_path, num_classes=1):
        super(GRUBasicX, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.bi = args.bidirectional == 1
        self.num_layers = 1 #TODO remove this param or fix this
        self.num_classes = num_classes
        self.embedding_depth = (
            (args.number_of_characters + len(args.extra_characters))*args.max_embedding_length
            if args.use_char_encoding
            else args.embedding_depth * args.max_embedding_length
        )
        self.projected_input_shape = (1, 1, self.embedding_depth) 

        self.gru = nn.GRU(
            self.embedding_depth,
            hidden_size=self.hidden_dim,
            bidirectional=self.bi,
            batch_first=False,
        )

        # Linear layer that maps from hidden state space to output space
        lin_input_layers = (2 if self.bi else 1) * self.hidden_dim
        self.hidden2out = nn.Linear(
            lin_input_layers, self.num_classes
        )
        self.hidden = self.init_hidden()

        self.load_state_dict(torch.load(model_path))
        #trace loaded modules
        self.gru = torch.jit.trace(self.gru, torch.rand(self.projected_input_shape), self.hidden)
        self.hidden2out = torch.jit.trace(self.hidden2out, torch.rand(lin_input_layers))

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        h0 = torch.zeros(
            (2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim
        ).to(device)
        return h0

    def forward(self, sequence):
        # print('Sequence shape:', sequence.shape)
        sequence = sequence.view(len(sequence), 1, -1)
        gru_out, _hidden = self.gru(#no longer update hidden state
            sequence
        )

        out_space = self.hidden2out(gru_out[:, -1])
        out_scores = torch.sigmoid(out_space).squeeze()
        return out_scores
