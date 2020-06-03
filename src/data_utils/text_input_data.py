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
import os
import copy
import json
import logging
import numpy as np
from .common import to_numpy, unpack_text_pairs

import torch
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for text classify dataset, as loaded from disk.

    Args:
        guid: The example's unique identifier
        text_a: first text
        text_b: second text
        label: the class label
    """
    def __init__(self, guid=None,text_a=None, text_b=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    Single squad example features to be fed to a model.
    """
    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor:

    def __init__(self, X, y=None):
        self.texts_a, self.texts_b = unpack_text_pairs(to_numpy(X))

        if y is not None:
            self.labels = to_numpy(y)
        else:
            self.labels = [None] * len(self.texts_a)

    def get_labels(self):
        return np.unique(self.labels)

    def get_examples(self):
        examples = []
        for i, (text_a, text_b, label) in enumerate(zip(self.texts_a, self.texts_b, self.labels)):
            examples.append(InputExample(guid="text-{}".format(i+1), text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

def load_and_cache_examples(args, tokenizer, processor, evaluate=False):

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(
        'test' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)
    ))

    if os.path.exists(cached_features_file) and not args.overwrite_cache and not evaluate:
        logger.info("Loading dataset from cached file %s", cached_features_file)
        dataset = torch.load(cached_features_file)

    else:
        logger.info("Creating dataset from dataset file at %s", args.data_dir)
        examples = processor.get_examples()
        if not evaluate and args.task_type == "classify":
            label_list = processor.get_labels()
        else:
            label_list = None

        dataset = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training=not evaluate,
            label_list=label_list
        )

    return dataset

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training, label_list):
    with Pool(cpu_count(),initializer=pool_init_fn, initargs=(tokenizer,)) as p:
        part_fn = partial(
            convert_example_to_features,
            max_length=max_seq_length,
            label_list=label_list
        )

        features = list(
            tqdm(
                p.imap(part_fn, examples),
                total=len(examples),
                desc="convert examples to features",
            )
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if not is_training:
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids)
        else:
            if label_list is not None:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels)
        return dataset

def convert_example_to_features(example, max_length,
                                  label_list=None,
                                  pad_on_left=False,
                                  pad_token=0,
                                  pad_token_segment_id=0,
                                  mask_padding_with_zero=True):

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        if label_list is not None:
            label = label_list.index(example.label)
        else:
            label = example.label

        data_index += 1
        if data_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("input_text: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))

        return InputFeatures(input_ids=input_ids,attention_mask=attention_mask,
                             token_type_ids=token_type_ids, label=label)


def pool_init_fn(tokenizer_for_convert):
    global tokenizer, data_index
    data_index = 0
    tokenizer = tokenizer_for_convert
