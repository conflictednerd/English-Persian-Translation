import os

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

from translator import Translator


class LMTranslator2(Translator):
    def __init__(self, args):
        super().__init__()
        self.CLEANIFY = False
        self.MODELS_DIR = args.models_dir
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.MODELS_DIR if args.load_model else 'facebook/m2m100_418M').to(self.DEVICE)
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            self.MODELS_DIR if args.load_model else'facebook/m2m100_418M', src_lang='en', tgt_lang='fa')

        self.dataset = self.load_dataset()
        if args.train:
            self.train(args)

        if args.test:
            self.test(args)

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
        dset = load_dataset('csv', data_files={'train': 'train.tsv', 'val': 'dev.tsv', 'test': 'test.tsv'}, column_names=[
            'en', 'fa', 'type'], delimiter='\t')

        # select rows
        dset = dset.filter(lambda batch: np.array(
            batch['type']) in ['mizan_train_en_fa', 'mizan_dev_en_fa', 'mizan_test_en_fa'])

        # normalize
        dset = dset.map(lambda batch: {'en': [self.clean_en(x) for x in batch['en']], 'fa': [self.clean_fa(x) for x in batch['fa']], 'type': [
                        x if x else 'other' for x in batch['type']]}, batched=True)

        # tokenize
        def tknize(batch):
            model_inputs = self.tokenizer(
                batch['en'], max_length=256, truncation=True, return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch['fa'], max_length=256, truncation=True, return_tensors='pt').input_ids
            model_inputs['labels'] = labels
            return model_inputs
        dset = dset.map(tknize, batched=True)

        return dset

    def translate(self, src_txt: str):
        pass

    def train(self, args):
        pass

    def test(self, args):
        data_collator = DataCollatorWithPadding(self.tokenizer)
        test_dataloader = DataLoader(self.test_data, batch_size=4, collate_fn=data_collator,
                                     shuffle=False)
        # print(self.test_data[0])
        metric = load_metric("bleu")

        self.model.eval()

        print(len(test_dataloader))

        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            if batch_idx == 5:
                exit(0)
            X1 = batch['input_ids'].to(self.DEVICE)
            # # print(X1.shape)
            y1 = batch['labels'].to(self.DEVICE)
            pred = self.model(X1, attention_mask=batch['attention_mask'].to(
                self.DEVICE))  # TODO: how to manage Seq2Seq batches here?

        # for index, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
        #     encoded_eng = self.tokenizer(row['english'], return_tensors="pt").to(self.DEVICE)
        #
        #     generated_tokens = self.model.generate(**encoded_eng,
        #                                            forced_bos_token_id=self.tokenizer.lang_code_to_id["fa_IR"])
        #
        #     model_predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #     ref = [[row['farsi'].split()]]
        #     pred = [model_predictions[0].split()]
        #     metric.add_batch(predictions=pred, references=ref)
        #
        # final_score = metric.compute()
        #
        # print(final_score)
