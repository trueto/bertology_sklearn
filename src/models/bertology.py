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

import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertologyModel(nn.Module):

    def __init__(self, model_name_or_path, cache_dir):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, cache_dir=cache_dir)
        self.bertology_model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                         from_tf=bool(".ckpt" in model_name_or_path),
                                                         config=self.config, cache_dir=cache_dir)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None
                ):

        return self.bertology_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )