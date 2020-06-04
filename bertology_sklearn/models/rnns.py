#   Copyright 2020 trueto

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

# refer to: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
class TextCNN(nn.Module):
    def __init__(self, hidden_size, kernel_num, kernel_sizes):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num,
                                             (K, hidden_size)) for K in kernel_sizes])

    def forward(self, hidden_states):
        # (N,Ci,W,D)
        hidden_states = hidden_states.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)
        hidden_states = [F.relu(conv(hidden_states)).squeeze(3) for conv in self.convs]

        # [(N, Co), ...]*len(Ks)
        hidden_states = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in hidden_states]

        hidden_states = torch.cat(hidden_states, 1)

        return hidden_states

# refer to: https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py
class TextRNN(nn.Module):

    def __init__(self, input_size, num_layers, rnn_model="LSTM", use_first=True):
        super().__init__()
        if rnn_model == "LSTM":
            self.rnn = nn.LSTM(input_size, input_size//2, num_layers=num_layers,
                               dropout=0.5, batch_first=True, bidirectional=True)
        if rnn_model == "GRU":
            self.rnn = nn.GRU(input_size, input_size//2, num_layers=num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)

        self.bn = nn.BatchNorm1d(input_size)
        self.use_first = use_first

    def forward(self, hidden_states):
        rnn_output, _ = self.rnn(hidden_states, None)

        if self.use_first:
            return self.bn(rnn_output[:, 0, :])
        else:
            return self.bn(torch.mean(rnn_output, dim=1))

class LSTM(nn.Module):
    def __init__(self, input_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, input_size // 2, num_layers=num_layers,
                dropout=0.5, batch_first=True, bidirectional=True)

    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states, None)
        return lstm_out