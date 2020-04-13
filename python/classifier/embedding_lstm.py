import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBasic(nn.Module):
    """
    Dropout note - lstm won't apply dropout to the last layer, so the only dropout to be apllies is 
    #TODO deprecate the build_in_dropout paramm for LSTM, turn to just a switch for dropout
    """
    def __init__(self, args, num_classes, built_in_dropout=False): 
        super(LSTMBasic, self).__init__()
        dropout = args.dropout_input
        self.hidden_dim = args.hidden_dim
        self.bi = args.bidirectional == 1
        self.num_layers = 1  # TODO remove param or deprecate?
        self.num_classes = num_classes
        self.embedding_size = (
            args.number_of_characters + len(args.extra_characters)
            if args.use_char_encoding
            else args.embedding_size * args.max_length
        )
        self.built_in_dropout = built_in_dropout

        if not built_in_dropout:
            self.lstm = nn.LSTM(
                self.embedding_size,
                hidden_size=self.hidden_dim,
                bidirectional=self.bi,
                batch_first=False,
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.lstm = nn.LSTM(
                self.embedding_size,
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
        h1 = torch.zeros(
            (2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim
        ).to(device)
        return (h0, h1)

    def forward(self, sequence):
        # print('Sequence shape:', sequence.shape)
        lstm_out, self.hidden = self.lstm(
            sequence.view(len(sequence), 1, -1), self.hidden
        )
        if self.built_in_dropout:
            out_space = self.hidden2out(
                self.dropout(lstm_out[:, -1].view(len(sequence), -1))
            )
        else:
            out_space = self.hidden2out(lstm_out[:, -1])

        # TODO experiment with adding a relu here, does it increase accuracy?
        # out_scores = F.log_softmax(out_space, dim=1)
        out_scores = torch.sigmoid(out_space).squeeze()
        return out_scores


class LSTMBasicX(nn.Module):
    """
    LSTM Basic Variant for export. Should be compativle and able to load state dict from the original LSTMBasic Class
    """
    def __init__(self, args, num_classes, model_path):
        super(LSTMBasicX, self).__init__()
        dropout = args.dropout_input
        self.hidden_dim = args.hidden_dim
        self.bi = args.bidirectional == 1
        self.num_layers = 1  # TODO remove param or deprecate?
        self.num_classes = num_classes
        self.embedding_size = (
            args.number_of_characters + len(args.extra_characters)
            if args.use_char_encoding
            else args.embedding_size * args.max_length
        )

        self.hidden = self.init_hidden()
        self.projected_input_shape = (1, 1, args.max_length*args.embedding_size)

        self.lstm = nn.LSTM(
            self.embedding_size,
            hidden_size=self.hidden_dim,
            bidirectional=self.bi,
            batch_first=False,
        )
        lin_input_layers = (2 if self.bi else 1) * self.hidden_dim
        self.hidden2out = nn.Linear(
            lin_input_layers, self.num_classes
        )

        self.load_state_dict(torch.load(model_path))
        #trace loaded modules
        self.lstm = torch.jit.trace(self.lstm, torch.rand(self.projected_input_shape), self.hidden)
        self.hidden2out = torch.jit.trace(self.hidden2out, torch.rand(lin_input_layers))

    def init_hidden(self):
        device = torch.device("cpu")
        h0 = torch.zeros(
            (2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim
        ).to(device)
        h1 = torch.zeros(
            (2 if self.bi else 1) * self.num_layers, 1, self.hidden_dim
        ).to(device)
        return (h0, h1)

    # @torch.jit.script_method
    def forward(self, sequence):
        # print('Sequence shape:', sequence.shape)
        sequence = sequence.view(len(sequence), 1, -1)
        # print("flattened sequence shape: ", sequence.shape)
        lstm_out, _hidden = self.lstm(
             sequence
        ) #Assuming that we can't update the hidden state as traces LSTM no longer takes in a hidden param
        #TODO verify this assumption
        out_space = self.hidden2out(lstm_out[:, -1])
        # print("out_space: ", out_space)
        out_scores = F.log_softmax(out_space, dim=1)
        # print("out scores ", out_scores)

        return out_scores
