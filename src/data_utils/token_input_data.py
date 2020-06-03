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
    def __init__(self, guid=None, tokens=None, labels=None):

        self.guid = guid
        self.tokens = tokens
        self.labels = labels

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
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids=None):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label_ids

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
        self.words_list = to_numpy(X)

        if y is not None:
            self.labels_list = to_numpy(y)
        else:
            self.labels_list = np.full_like(self.words_list, fill_value='O')

    def get_labels(self):
        return np.unique(np.ravel(self.labels_list))

    def get_examples(self):
        examples = []
        for i, (tokens, labels) in enumerate(zip(self.words_list, self.labels_list)):
            examples.append(InputExample(guid="tokens-{}".format(i+1), tokens=tokens, labels=labels))
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
        if not evaluate:
            label_list = processor.get_labels()
        else:
            label_list = None

        dataset = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            label_list=label_list
        )

    return dataset

def convert_examples_to_features(examples, tokenizer, max_seq_length, label_list):
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
        all_labels_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels_ids)

        return dataset

def convert_example_to_features(example, max_seq_length,
                                  label_list=None,
                                cls_token_at_end=False,
                                cls_token="[CLS]",
                                cls_token_segment_id=1,
                                sep_token="[SEP]",
                                sep_token_extra=False,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                pad_token_label_id=-100,
                                sequence_a_segment_id=0,
                                mask_padding_with_zero=True):
    tokens = []
    label_ids = []
    for word, label in zip(example.tokens, example.labels):
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_list.index[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
        label_ids = label_ids[:(max_seq_length - special_tokens_count)]


    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]

    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    data_index += 1

    if data_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

    return InputFeatures(input_ids=input_ids,
                         attention_mask=input_mask,
                         token_type_ids=segment_ids,
                         label_ids=label_ids)


def pool_init_fn(tokenizer_for_convert):
    global tokenizer, data_index
    data_index = 0
    tokenizer = tokenizer_for_convert
