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
import numpy as np
import torch.nn as nn
from glob import glob
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import random_split, DataLoader, SequentialSampler, RandomSampler

from bertology_sklearn.models import BertologyForTokenClassification
from bertology_sklearn.data_utils import TokenDataProcessor, token_load_and_cache_examples

logger = logging.getLogger(__name__)

class BertologyTokenClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_name_or_path="bert-base-chinese",
                 do_lower_case=True, cache_dir="cache_model", data_dir="cache_data",
                 max_seq_length=512, overwrite_cache=False, output_dir="results",
                 dev_fraction=0.1, per_train_batch_size=8, per_val_batch_size=8,
                 no_cuda=False, fp16=False, seed=42, overwrite_output_dir=False,
                 classifier_dropout=0.5, classifier_type="Linear", kernel_num=3,
                 kernel_sizes=(3,4,5), num_layers=2, weight_decay=1e-3,
                 gradient_accumulation_steps=1, max_epochs=10, learning_rate=2e-5,
                 warmup=0.1, fp16_opt_level='01', patience=3, n_saved=3):

        super().__init__()
        self.n_saved = n_saved
        self.patience = patience
        self.fp16_opt_level = fp16_opt_level
        self.warmup = warmup
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.classifier_type = classifier_type
        self.classifier_dropout = classifier_dropout
        self.overwrite_output_dir = overwrite_output_dir
        self.fp16 = fp16
        self.no_cuda = no_cuda
        self.dev_fraction = dev_fraction
        self.per_train_batch_size = per_train_batch_size
        self.per_val_batch_size = per_val_batch_size
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache
        self.model_name_or_path = model_name_or_path
        self.do_lower_case = do_lower_case
        self.cache_dir = cache_dir


        device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count() if not self.no_cuda else 1

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

    def fit(self, X, y, sample_weight=None):

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            # os.mkdir(self.data_dir)

        if not os.path.exists(self.output_dir):
            # os.mkdir(self.output_dir)
            os.makedirs(self.output_dir)

        if os.path.exists(self.output_dir) and os.listdir(
                self.output_dir) and not self.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir))

        ## data
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                  do_lower_case=self.do_lower_case,
                                                  cache_dir=self.cache_dir)

        processor = TokenDataProcessor(X, y)

        dataset = token_load_and_cache_examples(self, tokenizer, processor)

        ds_len = len(dataset)
        dev_len = int(len(dataset) * self.dev_fraction)
        train_ds, dev_ds = random_split(dataset, [ds_len - dev_len, dev_len])
        batch_size = self.n_gpu * self.per_train_batch_size
        train_sampler = RandomSampler(train_ds)
        train_iter = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)

        dev_sampler = SequentialSampler(dev_ds)
        dev_iter = DataLoader(dev_ds, sampler=dev_sampler, batch_size=batch_size)

        ## model
        self.label_list = processor.get_labels()
        self.num_labels = len(self.label_list)

        model = BertologyForTokenClassification(model_name_or_path=self.model_name_or_path,
                                                num_labels=self.num_labels, cache_dir=self.cache_dir,
                                                device=self.device,classifier_type=self.classifier_type,
                                                num_layers=self.num_layers)

        model.to(self.device)

        ## optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = len(train_iter) // self.gradient_accumulation_steps * self.max_epochs

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=t_total*self.warmup,
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
                "labels": labels
            }
            if self.classifier_type == "Linear":
                loss, sequence_tags = model(**inputs)
            else:
                loss, sequence_tags = model.module.forward(**inputs)

            score = (sequence_tags == labels).float()[labels != self.label_list.index("O")].mean()

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step_from_engine(engine))
            tb_writer.add_scalar('train_loss', loss.item(), global_step_from_engine(engine))
            tb_writer.add_scalar('train_score', score.item(), global_step_from_engine(engine))

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
                    "labels": batch[3]
                }
                if self.classifier_type == "Linear":
                    loss, sequence_tags = model(**inputs)
                else:
                    loss, sequence_tags = model.module.forward(**inputs)

                score = (sequence_tags == labels).float()[labels != self.label_list.index("O")].mean()

                if self.n_gpu > 1:
                    loss = loss.mean()
            ## tensorboard
            tb_writer.add_scalar('dev_loss', loss.item(), global_step_from_engine(trainer))
            tb_writer.add_scalar('dev_score', score.item(), global_step_from_engine(trainer))

            return loss.item(), score.item()

        dev_evaluator = Engine(eval_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(dev_evaluator, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(dev_evaluator, 'score')

        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss', 'score'])
        pbar.attach(dev_evaluator, ['loss', 'score'])

        def score_fn(engine):
            loss = engine.state.metrics['loss']
            return -loss

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

        checkpointer = ModelCheckpoint(self.output_dir, "best", n_saved=self.n_saved,
                                       create_dir=True,score_name="model_score",
                                       score_function=model_score,
                                       global_step_transform=global_step_from_engine(trainer))
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
            torch.save(self, os.path.join(self.output_dir, 'fit_args.pt'))

        trainer.run(train_iter, max_epochs=self.max_epochs)

    def predict(self, X):

        args = torch.load(os.path.join(self.output_dir, 'fit_args.pt'))

        ## data
        processor = TokenDataProcessor(X)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                  do_lower_case=self.do_lower_case,
                                                  cache_dir=self.cache_dir)
        dataset = token_load_and_cache_examples(self, tokenizer, processor, evaluate=True)

        sampler = SequentialSampler(dataset)
        batch_size = self.per_val_batch_size * max(1, self.n_gpu)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        ## model
        model = BertologyForTokenClassification(model_name_or_path=self.model_name_or_path,
                                                num_labels=self.num_labels, cache_dir=self.cache_dir,
                                                device=self.device, classifier_type=self.classifier_type,
                                                num_layers=self.num_layers)

        y_preds = []
        for model_state_path in glob(os.path.join(self.output_dir, '*.pth')):
            model.load_state_dict(torch.load(model_state_path))
            y_pred,  out_label_ids = self.single_predict(model, dataloader)
            y_preds.append(y_pred)

        y_preds = torch.tensor(y_preds)
        y_pred = torch.mode(y_preds, dim=0).values
        y_pred = y_pred.numpy()

        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        pad_token_label_id = nn.CrossEntropyLoss().ignore_index

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    preds_list[i].append(args.label_list[y_pred[i][j]])
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
            labels = batch[3]
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                if self.classifier_type == "Linear":
                    _, sequence_tags = model(**inputs)
                    sequence_tags = sequence_tags.detach.cpu().numpy()
                else:
                    _, sequence_tags = model.module.forward(**inputs)
                    sequence_tags = np.array(sequence_tags)

            labels = labels.detach.cpu().numpy()
            if preds is None:
                preds = sequence_tags
                out_label_ids = labels
            else:
                preds = np.append(preds, sequence_tags, axis=0)
                out_label_ids = np.append(out_label_ids, labels, axis=0)

        return preds, out_label_ids

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        score = f1_score(y, y_pred, average="macro")
        return score