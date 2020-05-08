import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, args, num_classes=1):
        super(RNN, self).__init__()
        self.embedding_depth = (
            (args.number_of_characters + len(args.extra_characters))*args.max_embedding_length
            if args.use_char_encoding
            else args.embedding_depth * args.max_embedding_length
        )
        self.hidden_size = args.hidden_size
        self.hidden = initHidden()

        self.i2h = nn.Linear(self.embedding_depth + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.embedding_depth + self.hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        self.hidden = hidden
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



class RNNX(nn.Module):
    def __init__(self, args, num_classes=1):
        super(RNNX, self).__init__()

        self.embedding_depth = (
            (args.number_of_characters + len(args.extra_characters))*args.max_embedding_length
            if args.use_char_encoding
            else args.embedding_depth * args.max_embedding_length
        )

        self.hidden_size = args.hidden_size
        self.hidden = initHidden()

        self.i2h = nn.Linear(input_size + self.hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + self.hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        self.hidden = hidden
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

