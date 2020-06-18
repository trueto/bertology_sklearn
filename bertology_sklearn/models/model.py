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
from .bertology import BertologyModel
from .rnns import TextCNN, TextRNN, LSTM
from .crf import CRF
from transformers import BertForTokenClassification
class BertologyForClassification(nn.Module):

    def __init__(self, model_name_or_path,
                 num_labels, cache_dir, dropout,
                 classifier_type="Linear", kernel_num=3,
                 kernel_sizes=(3, 4, 5), num_layers=2):

        super().__init__()
        self.bertology_model = BertologyModel(model_name_or_path, cache_dir)

        if classifier_type == "Linear":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bertology_model.config.hidden_size, num_labels)
            )

        elif classifier_type == "TextCNN":
            self.classifier = nn.Sequential(
                TextCNN(self.bertology_model.config.hidden_size, kernel_num, kernel_sizes),
                nn.Dropout(dropout),
                nn.Linear(len(kernel_sizes)*kernel_num, num_labels)
            )

        else:
            self.classifier = nn.Sequential(
                TextRNN(self.bertology_model.config.hidden_size, num_layers=num_layers,
                        rnn_model=classifier_type),
                nn.Dropout(dropout),
                nn.Linear(self.bertology_model.config.hidden_size, num_labels)
            )

        self.classifier_type = classifier_type

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        bertology_output = self.bertology_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        if self.classifier_type == "Linear":
            hidden_states = bertology_output[1]
        else:
            hidden_states = bertology_output[0]

        logits = self.classifier(hidden_states)

        return logits


class BertologyForTokenClassification(nn.Module):

    def __init__(self, model_name_or_path,
                 num_labels, cache_dir, dropout,
                 device, classifier_type="Linear",
                 num_layers=2, lstm_hidden_size=32):

        super().__init__()
        self.bertology_model = BertologyModel(model_name_or_path, cache_dir)

        if classifier_type == "Linear":
            self.token_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bertology_model.config.hidden_size, num_labels)
            )
        elif classifier_type == "CRF":
            self.token_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bertology_model.config.hidden_size, num_labels)
            )
            self.crf = CRF(num_labels, device=device)
        elif classifier_type == "LSTM_CRF":
            self.token_classifier = nn.Sequential(
                LSTM(self.bertology_model.config.hidden_size, lstm_hidden_size, num_layers=num_layers),
                nn.Dropout(dropout),
                nn.Linear(2*lstm_hidden_size, num_labels)
            )
            self.crf = CRF(num_labels, device=device)

        self.classifier_type = classifier_type
        self.num_labels = num_labels
        self.device = device

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None
                ):
        bertology_output = self.bertology_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        hidden_states = bertology_output[0]

        logits = self.token_classifier(hidden_states)

        if self.classifier_type == "Linear":
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = torch.tensor(0, device=self.device)

            sequence_tags = logits.argmax(dim=-1)

        else:
            byte_tensor = torch.empty(1, dtype=torch.uint8,
                                      device=input_ids.device)
            mask = attention_mask.type_as(byte_tensor)

            if labels is not None:
                log_likelihood = self.crf(logits, labels, mask)
                loss = -1 * log_likelihood
            else:
                loss = torch.tensor(0, device=self.device)

            sequence_tags = self.crf.decode(logits, mask)

        return loss, sequence_tags