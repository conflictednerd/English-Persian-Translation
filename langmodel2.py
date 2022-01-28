from typing import Union
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          DataCollatorWithPadding, M2M100Config,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MBart50TokenizerFast, MBartForConditionalGeneration,
                          Trainer, TrainingArguments)
from sacrebleu.metrics import BLEU, TER

from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from translator import Translator


class LMTranslator2(Translator):
    def __init__(self, args):
        super().__init__()
        self.CLEANIFY = False
        self.MODELS_DIR = os.path.join(args.models_dir, 'lm2')
        self.DATA_PATH = './data/'
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.MODELS_DIR if args.load_model else 'facebook/m2m100_418M').to(self.DEVICE)
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            self.MODELS_DIR if args.load_model else'facebook/m2m100_418M', src_lang='en', tgt_lang='fa')

        self.metric = load_metric('sacrebleu')
        self.dataset = self.load_dataset()
        # if args.train:
        #     self.train(args)

        # if args.test:
        #     self.test(args)

    def load(self, path: str):
        pass

    def save(self):
        self.tokenizer.save_pretrained(self.MODELS_DIR)
        self.model.save_pretrained(self.MODELS_DIR)

    def load_dataset(self):
        '''
        loads from train.tsv, dev.tsv, test.tsv
        '''
        # Load from file
        dset = load_dataset('csv', data_files={'train': os.path.join(self.DATA_PATH, 'train.tsv'), 'val': os.path.join(self.DATA_PATH, 'dev.tsv'), 'test': os.path.join(self.DATA_PATH, 'test.tsv')}, column_names=[
            'en', 'fa', 'type'], delimiter='\t')

        # select rows
        dset = dset.filter(lambda batch: np.array(
            batch['type']) in ['mizan_train_en_fa', 'mizan_dev_en_fa', 'mizan_test_en_fa'])

        dset = dset.filter(lambda batch: isinstance(batch['en'], str) and isinstance(batch['fa'], str))
        dset['train'] = dset['train'].shuffle(seed=23).select(range(10_000)) # ONLY USE 10000 SAMPLES
        # normalize
        dset = dset.map(lambda batch: {'en': [self.clean_en(x) for x in batch['en']], 'fa': [self.clean_fa(x) for x in batch['fa']], 'type': [
                        x if x else 'other' for x in batch['type']]}, batched=True)

        # tokenize
        def tknize(batch):
            model_inputs = self.tokenizer(
                batch['en'], max_length=128, truncation=True, return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch['fa'], max_length=128, truncation=True, return_tensors='pt').input_ids
            model_inputs['labels'] = labels
            return model_inputs
        dset = dset.map(tknize, batched=True)

        return dset

    def translate(self, src_txt: Union[str, List], from_text=True):
        src_txt = [src_txt] if isinstance(src_txt, str) else src_txt
        translated = self.model.generate(
            **self.tokenizer(src_txt, return_tensors='pt', max_length=256, truncation=True, padding=True).to(self.DEVICE) if from_text else src_txt,
            forced_bos_token_id=self.tokenizer.lang_code_to_id['fa'])
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

    def train(self, args):
        # How can we force the first token to be 'fa'?
        self.model.config.bos_token_id = self.tokenizer.lang_code_to_id['fa']
        training_args = Seq2SeqTrainingArguments(
            f'{self.MODELS_DIR}/lm2-finetuned-en-fa',
            evaluation_strategy='epoch',
            learning_rate=args.lm2_lr,
            per_device_train_batch_size=args.lm2_batch_size,
            per_device_eval_batch_size=args.lm2_batch_size,
            weight_decay=5e-3,
            save_total_limit=1,
            num_train_epochs=1,
            predict_with_generate=True,
            generation_num_beams=8,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True)

        dset = self.dataset.remove_columns(['en', 'fa', 'type'])
        trainer = Seq2SeqTrainer(
            self.model,
            training_args,
            train_dataset=dset['train'],
            eval_dataset=dset['val'],  # should I do test here?
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.MODELS_DIR)

    def compute_metrics(self, predictions):
        preds, refs = predictions
        preds = preds[0] if isinstance(preds, tuple) else preds
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        # datacollator pads the labels with -100 which can't be decoded
        refs = np.where(refs != -100, refs, self.tokenizer.pad_token_id)
        decoded_refs = self.tokenizer.batch_decode(
            refs, skip_special_tokens=True)

        result = self.metric.compute(
            predictions=[x.strip() for x in decoded_preds],
            references=[[x.strip()] for x in decoded_refs]
        )
        return {key: round(val, 3) for key, val in result.items()}

    def test(self, args):
        testset = self.dataset['test']
        refs = [[x] for x in testset['fa']]
        preds = self.translate(testset['en'].to(self.DEVICE))

        result = self.metric.compute(predictions=preds, references=refs)
        print(result)
        return result
