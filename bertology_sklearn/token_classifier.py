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
import torch
import logging
import shutil
import numpy as np
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from seqeval.metrics import f1_score

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AdamW, \
    get_cosine_with_hard_restarts_schedule_with_warmup, BertTokenizer, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import random_split, DataLoader, SequentialSampler, RandomSampler

from bertology_sklearn.models import BertologyForTokenClassification
from bertology_sklearn.data_utils import TokenDataProcessor, token_load_and_cache_examples
from bertology_sklearn.data_utils.common import to_numpy

logger = logging.getLogger(__name__)

class BertologyTokenClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_name_or_path="bert-base-chinese",
                 do_lower_case=True, cache_dir="cache_model", data_dir="cache_data",
                 max_seq_length=512, overwrite_cache=False, output_dir="results",
                 dev_fraction=0.1, per_train_batch_size=8, per_val_batch_size=8,
                 no_cuda=False, seed=42, overwrite_output_dir=False,
                 bert_dropout=0.1, lstm_dropout=0.5, classifier_type="Linear", kernel_num=3,
                 kernel_sizes=(3, 4, 5), num_layers=2, weight_decay=1e-3,
                 gradient_accumulation_steps=1, max_epochs=10, learning_rate=2e-5,
                 warmup=0.1, fp16=False, fp16_opt_level='01', patience=3, n_saved=3,
                 do_cv=False, schedule_type="linear", lstm_hidden_size=32, k_fold=5,
                 is_nested=False, multi_label_threshold=0.5):

        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.do_lower_case = do_lower_case
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache
        self.output_dir = output_dir
        self.dev_fraction = dev_fraction
        self.per_train_batch_size = per_train_batch_size
        self.per_val_batch_size = per_val_batch_size
        self.overwrite_output_dir = overwrite_output_dir
        self.no_cuda = no_cuda
        self.classifier_type = classifier_type
        self.bert_dropout = bert_dropout
        self.lstm_dropout = lstm_dropout
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.fp16_opt_level = fp16_opt_level
        self.fp16 = fp16

        self.n_saved = n_saved
        self.patience = patience
        self.do_cv = do_cv
        self.k_fold = k_fold

        self.seed = seed
        self.schedule_type = schedule_type
        self.lstm_hidden_size = lstm_hidden_size
        self.is_nested = is_nested
        self.multi_label_threshold = multi_label_threshold

        device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = max(torch.cuda.device_count() if not self.no_cuda else 1, 1)

        self.device = device

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger.warning("Process device: %s, n_gpu: %s,16-bits training: %s",
                       device, self.n_gpu, self.fp16)

        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        if "CRF" in self.classifier_type and self.is_nested:
            raise ValueError("CRF is not supported for nested NER!")

    def fit(self, X, y, sample_weight=None):

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            # os.mkdir(self.data_dir)

        if not os.path.exists(self.output_dir):
            # os.mkdir(self.output_dir)
            os.makedirs(self.output_dir)

        if self.overwrite_output_dir:
            shutil.rmtree(self.output_dir)

        if os.path.exists(self.output_dir) and os.listdir(
                self.output_dir) and not self.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir))

        ## tokenizer
        if 'chinese' in self.model_name_or_path:
            tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path,
                                                      do_lower_case=self.do_lower_case,
                                                      cache_dir=self.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                  do_lower_case=self.do_lower_case,
                                                  cache_dir=self.cache_dir)

        if self.do_cv:
            kfold = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.seed)
            cv = 0
            X, y = to_numpy(X), to_numpy(y)
            for train_index, dev_index in kfold.split(X, y):
                cv += 1
                ## data
                X_train, X_dev = X[train_index], X[dev_index]
                y_train, y_dev = y[train_index], y[dev_index]

                self.overwrite_cache = True

                train_processor = TokenDataProcessor(X_train, y_train, is_nested=self.is_nested)
                dev_processor = TokenDataProcessor(X_dev, y_dev, is_nested=self.is_nested)

                self.label_list = train_processor.get_labels()
                self.num_labels = len(self.label_list)
                train_ds = token_load_and_cache_examples(self, tokenizer, train_processor)
                dev_ds = token_load_and_cache_examples(self, tokenizer, dev_processor)

                self.single_fit(train_ds, dev_ds, cv=cv)
        else:
            ## data
            processor = TokenDataProcessor(X, y, is_nested=self.is_nested)
            dataset = token_load_and_cache_examples(self, tokenizer, processor)

            self.label_list = processor.get_labels()
            self.num_labels = len(self.label_list)

            ds_len = len(dataset)
            dev_len = int(len(dataset) * self.dev_fraction)
            train_ds, dev_ds = random_split(dataset, [ds_len - dev_len, dev_len])
            self.single_fit(train_ds, dev_ds)

    def single_fit(self, train_ds, dev_ds, cv=None):
        ## data_iter
        batch_size = self.n_gpu * self.per_train_batch_size
        train_sampler = RandomSampler(train_ds)
        train_iter = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)

        dev_sampler = SequentialSampler(dev_ds)
        dev_iter = DataLoader(dev_ds, sampler=dev_sampler, batch_size=batch_size)

        ## model
        model = BertologyForTokenClassification(model_name_or_path=self.model_name_or_path,
                                                num_labels=self.num_labels, cache_dir=self.cache_dir,
                                                device=self.device, bert_dropout=self.bert_dropout,
                                                lstm_dropout=self.lstm_dropout,
                                                classifier_type=self.classifier_type,
                                                num_layers=self.num_layers, lstm_hidden_size=self.lstm_hidden_size)

        model.to(self.device)

        ## optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if 'CRF' in self.classifier_type:
            step_lr = ['GRU', 'LSTM']
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in step_lr)],
                 'weight_decay': 0.0, 'lr': self.learning_rate*10}
            )
        t_total = len(train_iter) // self.gradient_accumulation_steps * self.max_epochs
        warmup_steps = t_total * self.warmup

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        if self.schedule_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
        elif self.schedule_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
        elif self.schedule_type == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif self.schedule_type == "cosine_restarts":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                           num_training_steps=t_total)
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        tb_writer = SummaryWriter()

        def train_fn(engine, batch):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(self.device) for t in batch)
            labels = batch[3]
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": labels,
                "is_nested": self.is_nested
            }

            loss, sequence_tags = model(**inputs)
            if not self.is_nested:
                score = (sequence_tags == labels).float().detach().cpu().numpy()

                condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
                condition_2 = (labels != self.label_list.index("<PAD>")).detach().cpu().numpy()
                patten = np.logical_and(condition_1, condition_2)
                score = score[patten].mean()
            else:
                '''
                y_pred = sequence_tags.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                score = ((y_pred > self.multi_label_threshold) == (labels_np > 0)).mean()
                '''
                score = ((sequence_tags > self.multi_label_threshold) == (labels > 0)).float().detach().cpu().numpy()
                condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
                condition_2 = (labels != self.label_list.index("<PAD>")).detach().cpu().numpy()
                patten = np.logical_and(condition_1, condition_2)
                score = score[patten].mean()

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(engine)(engine, engine.last_event_name)
            tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', loss.item(), global_step)
            tb_writer.add_scalar('train_score', score.item(), global_step)

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            return loss.item(), score.item()

        trainer = Engine(train_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'score')

        def eval_fn(engine, batch):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                optimizer.zero_grad()
                labels = batch[3]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": labels,
                    "is_nested": self.is_nested
                }

                loss, sequence_tags = model(**inputs)
                if not self.is_nested:
                    score = (sequence_tags == labels).float().detach().cpu().numpy()

                    condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
                    condition_2 = (labels != self.label_list.index("<PAD>")).detach().cpu().numpy()
                    patten = np.logical_and(condition_1, condition_2)
                    score = score[patten].mean()
                else:
                    score = ((sequence_tags > self.multi_label_threshold) == (labels > 0)).float().detach().cpu().numpy()
                    '''
                    y_pred = sequence_tags.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    score = ((y_pred > self.multi_label_threshold) == (labels_np > 0)).mean()
                    '''
                    condition_1 = (labels != self.label_list.index("O")).detach().cpu().numpy()
                    condition_2 = (labels != self.label_list.index("<PAD>")).detach().cpu().numpy()
                    patten = np.logical_and(condition_1, condition_2)
                    score = score[patten].mean()

                if self.n_gpu > 1:
                    loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(trainer)(engine, engine.last_event_name)
            tb_writer.add_scalar('dev_loss', loss.item(), global_step)
            tb_writer.add_scalar('dev_score', score.item(), global_step)

            return loss.item(), score.item()

        dev_evaluator = Engine(eval_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(dev_evaluator, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(dev_evaluator, 'score')

        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss', 'score'])
        pbar.attach(dev_evaluator, ['loss', 'score'])

        def score_fn(engine):
            loss = engine.state.metrics['loss']
            score = engine.state.metrics['score']
            return score / (loss + 1e-12)

        handler = EarlyStopping(patience=self.patience, score_function=score_fn, trainer=trainer)
        dev_evaluator.add_event_handler(Events.COMPLETED, handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_dev_results(engine):
            dev_evaluator.run(dev_iter)
            dev_metrics = dev_evaluator.state.metrics
            avg_score = dev_metrics['score']
            avg_loss = dev_metrics['loss']
            logger.info(
                "Validation Results - Epoch: {}  Avg score: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_score, avg_loss))

        l = self.model_name_or_path.split('/')
        if len(l) > 1:
            model_name = l[-1]
        else:
            model_name = self.model_name_or_path

        def model_score(engine):
            score = engine.state.metrics['score']
            return score

        model_prefix = "bertology_{}".format(self.classifier_type.lower()) \
            if cv is None else "bertology_{}_cv_{}".format(self.classifier_type.lower(), cv)

        checkpointer = ModelCheckpoint(self.output_dir, model_prefix, n_saved=self.n_saved,
                                       create_dir=True, score_name="model_score",
                                       score_function=model_score,
                                       global_step_transform=global_step_from_engine(trainer),
                                       require_empty=False)
        dev_evaluator.add_event_handler(Events.COMPLETED, checkpointer,
                                        {model_name: model.module if hasattr(model, 'module') else model})

        # Clear cuda cache between training/testing
        def empty_cuda_cache(engine):
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
        dev_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

        # save config
        @trainer.on(Events.COMPLETED)
        def save_config(engine):
            torch.save(self, os.path.join(self.output_dir, 'fit_args.pkl'))

        trainer.run(train_iter, max_epochs=self.max_epochs)

    def predict(self, X):

        args = torch.load(os.path.join(self.output_dir, 'fit_args.pkl'))


        ## tokenizer
        if 'chinese' in self.model_name_or_path:
            tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path,
                                                      do_lower_case=self.do_lower_case,
                                                      cache_dir=self.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                      do_lower_case=self.do_lower_case,
                                                      cache_dir=self.cache_dir)

        ## data
        processor = TokenDataProcessor(X)
        dataset = token_load_and_cache_examples(args, tokenizer, processor, evaluate=True)

        sampler = SequentialSampler(dataset)
        batch_size = self.per_val_batch_size * max(1, self.n_gpu)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        ## model
        model = BertologyForTokenClassification(model_name_or_path=self.model_name_or_path,
                                                num_labels=args.num_labels, cache_dir=self.cache_dir,
                                                device=self.device, classifier_type=self.classifier_type,
                                                num_layers=self.num_layers,  bert_dropout=self.bert_dropout,
                                                lstm_dropout=self.lstm_dropout,
                                                lstm_hidden_size=self.lstm_hidden_size)

        y_preds = []
        for model_state_path in glob(os.path.join(self.output_dir, '*.pt*')):
            model.load_state_dict(torch.load(model_state_path))
            y_pred,  out_label_ids = self.single_predict(model, dataloader)
            y_preds.append(y_pred)

        if not self.is_nested:
            y_preds = torch.tensor(y_preds)
            y_pred = torch.mode(y_preds, dim=0).values
            y_pred = y_pred.numpy()
        else:
            tmp_y_pred = np.max(y_preds, axis=0)
            y_pred = [[]] * tmp_y_pred.shape[0]

            for i, seq in enumerate(tmp_y_pred):
                for j, label_logits in enumerate(seq):
                    y_ = []
                    for k, pred in enumerate(label_logits):
                        if pred > self.multi_label_threshold:
                            y_.append(args.label_list[k])
                    y_pred[i].append(y_)

        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        pad_token_label_id = nn.CrossEntropyLoss().ignore_index

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    if not self.is_nested:
                        preds_list[i].append(args.label_list[y_pred[i][j]])
                    else:
                        preds_list[i].append(y_pred[i][j])
        return preds_list

    def single_predict(self, model, data_iter):
        model.to(self.device)

        # multi-gpu eval
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Predict
        logger.info("***** Running predict*****")
        logger.info("  Num examples = %d", len(data_iter)*self.per_val_batch_size*self.n_gpu)

        preds = None
        out_label_ids = None
        for batch in tqdm(data_iter, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            label_mask = batch[4]
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "is_nested": self.is_nested
                }
                _, sequence_tags = model(**inputs)
                # if self.classifier_type == "Linear" or self.n_gpu <= 1:
                #    _, sequence_tags = model(**inputs)

                # if self.n_gpu > 1 and "CRF" in self.classifier_type:
                #    _, sequence_tags = model.module.forward(**inputs)

            label_mask = label_mask.detach().cpu().numpy()
            sequence_tags = sequence_tags.detach().cpu().numpy()
            if preds is None:
                preds = sequence_tags
                out_label_ids = label_mask
            else:
                preds = np.append(preds, sequence_tags, axis=0)
                out_label_ids = np.append(out_label_ids, label_mask, axis=0)

        return preds, out_label_ids

    def multi_label_to_id(self, y):
        args = torch.load(os.path.join(self.output_dir, 'fit_args.pkl'))
        label_ids = [ [] ]* len(y)
        for i, seq in enumerate(y):
            for j, label_list in enumerate(seq):
                label_id = [-1] * len(args.label_list)
                for label in label_list:
                    label_id[args.label_list.index(label)] = 1
                label_ids[i].append(label_id)
        return torch.LongTensor(label_ids)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if not self.is_nested:
            score = f1_score(y, y_pred, average="macro")
        else:
            y_pred_ids = self.multi_label_to_id(y_pred)
            y_true_ids = self.multi_label_to_id(y)
            score = (y_pred_ids == y_true_ids).float().mean()
            score = score.item()
        return score